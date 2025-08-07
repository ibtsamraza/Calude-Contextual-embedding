import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

file_path_document = "contextual_texts_with_metadata.pkl"  # Replace with your actual file path

with open(file_path_document, "rb") as file:
    contextual_texts = pickle.load(file)  # Load the pickled data



embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"}
)

vectorstore = FAISS.from_documents(contextual_texts, embedding_model)

vector_embeddings = "./nust_app_db"

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(vector_embeddings):
    os.makedirs(vector_embeddings)
    print(f"Directory '{vector_embeddings}' created.")
else:
    print(f"Directory '{vector_embeddings}' already exists.")

# Save the vector store
vectorstore.save_local(vector_embeddings)
