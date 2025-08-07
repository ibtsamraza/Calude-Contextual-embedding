import os
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
import pickle

chunk_file_path = "data_chunks.pkl"
document_file_path = "document.pkl"

# Write contextual_texts to a file
def save_contextual_texts(contextual_texts, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(contextual_texts, file)
    print(f"Contextual texts saved to {file_path}")


def list_pdf_files(directory):
    pdf_files = []
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                # Construct the full file path
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    
    return pdf_files

# Specify the directory you want to search
directory_path = './NUST-Documents'

# Get the list of PDF files
pdf_file_paths = list_pdf_files(directory_path)

# Print the list of PDF file paths
for pdf_path in pdf_file_paths:
    print(pdf_path)
    break

print(f"this is the length of the list {len(pdf_file_paths)}")

documents_directory = "./NUST-Documents"
loader = DirectoryLoader(
    documents_directory, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
documents = loader.load()

print("Number of LangChain documents:", len(documents))
save_contextual_texts(documents, document_file_path)


loader = UnstructuredLoader(
    file_path = pdf_file_paths,
    chunking_strategy="basic",
    max_characters=1000,
    overlap_all = True,
    include_orig_elements=False,
)

docs = loader.load()

print("Number of LangChain documents:", len(docs))
print("Length of text in the document:", len(docs[0].page_content))


save_contextual_texts(docs, chunk_file_path)

