from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from src.utils.config_loader import load_config

cfg = load_config("configs/config.yaml")

_embeddings = HuggingFaceEmbeddings(
    model_name=cfg["agent"]["embeddings_model"]
)
_vector_store = Chroma(
    embedding_function=_embeddings,
    persist_directory=cfg["agent"]["vectorstore_dir"],
)

@tool
def search_regulations(query: str) -> str:
    """Search German workplace-safety regulations (ArbSchG, PSA-Benutzungsverordnung,
    DGUV rules, BGR) for clauses relevant to a PPE requirement or violation.
    Use a short topical query such as 'head protection employer duty' or
    'eye protection requirements'. Returns the most relevant regulation excerpts."""
    docs = _vector_store.similarity_search(query)
    if not docs:
        return "No matching regulation text found."
    return "\n\n".join(
        f"[{d.metadata.get('regulation', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
