import time
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class _State:
    missing: Set[str] = field(default_factory=set)  # last observed missing set
    confirm_streak: int = 0      # consecutive non-compliant cycles
    clear_streak: int = 0        # consecutive compliant cycles
    absent: int = 0              # cycles since this person was last seen
    active: bool = False         # an episode is currently open (start already logged)
    episode_missing: Set[str] = field(default_factory=set)
    episode_start: float = 0.0


class ViolationTracker:
    """Turns noisy per-frame results into stable per-person violation episodes.

    An episode = a continuous stretch where a person is non-compliant. We only
    log its start and its end, never every frame, so the log stays sparse and
    easy for a downstream report generator to consume.

    Debouncing:
      confirm_after - non-compliance must hold this many cycles before a
                      violation_start is emitted (filters single-frame misses).
      clear_after   - compliance must hold this many cycles before an open
                      episode is closed (filters single-frame false positives).
      forget_after  - if a person is unseen this many cycles, drop their state;
                      a returning ID then starts clean. An open episode is
                      closed when the person disappears for good.
    """

    def __init__(self, required, confirm_after: int = 3,
                 clear_after: int = 2, forget_after: int = 15):
        self.required = set(required)
        self.confirm_after = confirm_after
        self.clear_after = clear_after
        self.forget_after = forget_after
        self._states: Dict[int, _State] = {}

    def flush(self, now=None):
        """Close any still-open episodes. Call this when the stream ends."""
        now = time.time() if now is None else now
        events = [self._close(pid, st, now)
              for pid, st in list(self._states.items()) if st.active]
        self._states.clear()
        return events

    def update(self, worn,  now=None) -> List[dict]:
        """Feed one cycle of {track_id: worn_ppe}. Returns the events to log."""
        now = time.time() if now is None else now 
        events: List[dict] = []
        seen = set(worn)

        for pid, worn_ppe in worn.items():
            st = self._states.setdefault(pid, _State())
            st.absent = 0
            missing = self.required - worn_ppe

            if missing:
                st.clear_streak = 0
                st.confirm_streak += 1
                st.missing = missing
                if not st.active and st.confirm_streak >= self.confirm_after:
                    st.active = True
                    st.episode_missing = set(missing)
                    st.episode_start = now
                    events.append({
                        "event": "violation_start",
                        "person_id": pid,
                        "missing_ppe": sorted(missing),
                        "timestamp": now,
                    })
            else:
                st.confirm_streak = 0
                st.clear_streak += 1
                if st.active and st.clear_streak >= self.clear_after:
                    events.append(self._close(pid, st, now))

        # age out people who left the frame
        for pid in list(self._states):
            if pid in seen:
                continue
            st = self._states[pid]
            st.absent += 1
            if st.absent >= self.forget_after:
                if st.active:
                    events.append(self._close(pid, st, now))
                del self._states[pid]

        return events

    def _close(self, pid: int, st: _State, now: float) -> dict:
        st.active = False
        return {
            "event": "violation_end",
            "person_id": pid,
            "missing_ppe": sorted(st.episode_missing),
            "timestamp": now,
            "duration_s": round(now - st.episode_start, 1),
        }
