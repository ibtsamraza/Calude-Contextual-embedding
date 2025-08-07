from typing import List
import  os
#from together import Together
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pickle
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama


# client = Together(api_key = TOGETHER_API_KEY)

model = ChatOllama(model="llama3.2:3b", temprature = 1)


file_path_document = "document.pkl"  # Replace with your actual file path

with open(file_path_document, "rb") as file:
    documents = pickle.load(file)  # Load the pickled data

print(f"Created documents and the length of documents are {len(documents)}")

file_path_chunk = "data_chunks.pkl"  # Replace with your actual file path

with open(file_path_chunk, "rb") as file:
    texts = pickle.load(file)  # Load the pickled data

print(f"Create chunks and the length of chunks are {len(texts)}")

# file_path = "contextual_texts_with_metadata.pkl"

# # Write contextual_texts to a file
# def save_contextual_texts(contextual_texts, file_path):
#     with open(file_path, "wb") as file:
#         pickle.dump(contextual_texts, file)
#     print(f"Contextual texts saved to {file_path}")

file_path = "test_texts_with_metadata.pkl"

# Write contextual_texts to a file
def save_contextual_texts(contextual_texts, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(contextual_texts, file)
    print(f"Contextual texts saved to {file_path}")


# CONTEXTUAL_RAG_PROMPT = """
# Given the document below, we want to explain what the chunk captures in the document.

# {WHOLE_DOCUMENT}

# Here is the chunk we want to explain:

# {CHUNK_CONTENT}

# Answer ONLY with a succinct explaination of the meaning of the chunk in the context of the whole document above.
# """
CONTEXTUAL_RAG_PROMPT = """
Given the document below, analyze and contextualize the provided chunk within the broader content. Identify key themes, relationships, and any critical background information that enhances understanding of the chunk.

{WHOLE_DOCUMENT}

Here is the specific chunk we want to explain:

{CHUNK_CONTENT}

Provide a concise yet informative explanation of the chunk’s meaning, ensuring it reflects its role within the overall document. Highlight any relevant context, important references, or underlying implications that contribute to its significance.
"""



index = 0
prompts = []
loop_out = 0

for text in texts:
    # Avoid index out-of-bounds errors
    if index >= len(documents):
        print("Warning: Index out of bounds. Stopping loop.")
        break

    # Clean the source paths by removing "./" if present
    text_source = text.metadata['source'].lstrip("./")
    doc_source = documents[index].metadata['source']

    #print(text_source, "\n", doc_source)

    # Match metadata sources and generate prompts
    if text_source == doc_source:
        prompt = CONTEXTUAL_RAG_PROMPT.format(WHOLE_DOCUMENT=documents[index].page_content, CHUNK_CONTENT=text.page_content)
        prompts.append(prompt)
    else:
        # Increment index and ensure it's still valid
        index += 1
        if index < len(documents):
            prompt = CONTEXTUAL_RAG_PROMPT.format(WHOLE_DOCUMENT=documents[index].page_content, CHUNK_CONTENT=text.page_content)
            prompts.append(prompt)
        else:
            print("Warning: No more documents to process.")
            break
    loop_out += 1
    if loop_out > 20:
        break

print(f"Prompts are generated and the length of prompts is {len(prompts)}")



def generate_context(prompt: str):
    """
    Generates a contextual response based on the given prompt using the specified language model.
    Args:
        prompt (str): The input prompt to generate a response for.
    Returns:
        str: The generated response content from the language model.
    """
    # response = client.chat.completions.create(
    #     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.5
    # )
    # return response.choices[0].message.content
    response = model.invoke(prompt)
    #print(response.content)
    return response.content


contextual_texts = []
# Let's generate the entire list of contextual chunks and concatenate to the original chunk
for i in range(len(texts)):
    contextual_chunk = generate_context(prompts[i])+' '+texts[i].page_content
    contextual_texts.append(Document(page_content = contextual_chunk, metadata = texts[i].metadata))
    print(contextual_texts[0].metadata)
    print(f"the {i}th chunk is currently contextualize")
    break


print(f"contextulized text is created and the length of the contestualize text is {len(contextual_texts)}")
contextual_chunks = [generate_context(prompts[i])+' '+texts[i].page_content for i in range(len(texts))]
print(contextual_chunks[:10])
#Save to file
save_contextual_texts(contextual_texts, file_path)




