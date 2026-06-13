from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from src.utils.config_loader import load_config
from langchain_chroma import Chroma

cfg = load_config("configs/config.yaml")
folder = Path(cfg["agent"]["regulations_dir"])
emb_model_name = cfg["agent"]["embeddings_model"]
vector_dir = cfg["agent"]["vectorstore_dir"]


docs = []
for pdf_path in sorted(folder.glob("*.pdf")):
    loader = PyMuPDF4LLMLoader(file_path=str(pdf_path), mode="page")
    pdf_docs = loader.load()

    # tag each chunk with a clean regulation name for later citation
    for doc in pdf_docs:
        doc.metadata["regulation"] = pdf_path.stem 

    docs.extend(pdf_docs)
    print(f"{pdf_path.name}: {len(pdf_docs)} pages")

print(f"\nTotal: {len(docs)} pages from {len(list(folder.glob('*.pdf')))} files")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print(f"Length of all_splits: {len(all_splits)}")




embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=vector_dir, 
)


ids = vector_store.add_documents(documents=all_splits)
print(f"Added {len(ids)} chunks to vector store")


results = vector_store.similarity_search(
    "what is the regulation for missing safety helmet"
)

#print(results[0])
