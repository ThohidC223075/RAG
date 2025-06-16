from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('File\Current Essentials of Medicine.pdf')

documents = loader.load()

print(len(documents))

print(documents[1])

print(documents[1].page_content)

print(documents[1].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # প্রতি chunk হবে ৩০০ ক্যারেক্টারের
    chunk_overlap=50      # আগের chunk এর ৫০ ক্যারেক্টার overlap থাকবে
)

# 3️⃣ এখন চাংক তৈরি করো
chunks = text_splitter.split_documents(documents)

# 4️⃣ চাংকের সংখ্যা এবং কন্টেন্ট দেখাও
print(f"Total Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n🔹 Chunk {i+1}:\n{chunk.page_content}")