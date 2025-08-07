import pickle
import json
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize

file_path_document = "./data_chunks.pkl"
with open(file_path_document, "rb") as file:
    documents = pickle.load(file)  # Load the pickled data

# print(type(documents[0]))
# print(documents[10].page_content)



print("BM25 retriever saved successfully!")

retriever = BM25Retriever.from_documents(
    documents,
    k=10,
    preprocess_func=word_tokenize,
)
result = retriever.invoke("how many F grade a student can have")
print(result[0].page_content)

with open("bm25_retriever.pkl", "wb") as f:
    pickle.dump(retriever, f)