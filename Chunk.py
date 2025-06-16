from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('File\Current Essentials of Medicine.pdf')

documents = loader.load()

""" Printing The length of this documents.You can also see that the length 
is 607 and the pdf page also 607 """
print(len(documents))

""" Every documents have two things, one is page_content and the other 
one is metada. Which discuss in README file """
print(documents[1])

print(documents[1].page_content)

print(documents[1].metadata)

# Initialize the text splitter with desired chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # Max characters per chunk (~token-aware)
    chunk_overlap=50    # Overlap between chunks to preserve context
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

# Print stats and preview
print(f"Total Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):  # Display first 3 chunks
    print(f"\n🔹 Chunk {i+1}:\n{chunk.page_content}")