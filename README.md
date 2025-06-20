# RAG

### You have to Install
    1.langchain-community
    2.pypdf
### How to Install it?
    pip install langchain-community pypdf



1️⃣ What is a Document Loader and Why Do We Need It?

    A Document Loader is a component or class that helps you load documents from different sources like PDF, Word, CSV, websites, or databases into a standard format (usually text).

🔍 Why do we need it?

    1.To read files from various formats (PDF, HTML, DOCX, etc.)

    2.To convert content into a clean, uniform structure for further processing (like LLM or RAG systems)

    3.To automate loading from external sources like URLs or APIs

2️⃣ What is a Text Splitter and Why Do We Need It?

    A Text Splitter is a tool that breaks long text into smaller parts (chunks). This is important because many LLMs (like GPT) have a token limit (e.g., 4096 tokens).

🔍 Why do we need it?

    1.To avoid exceeding LLM’s token limits

    2.To help the model understand context better by organizing info into manageable pieces

    3.To prepare chunks for vector embedding (for RAG systems)

3️⃣ What is Chunking and Why Do We Need It?

    Chunking is the process of splitting long documents into small pieces (chunks) to make them easier for AI models to process, embed, or retrieve relevant parts from.

🔍 Why do we need it?

    1.LLMs can't process full documents with too many tokens

    2.Chunks are used in Retrieval-Augmented Generation (RAG) to fetch only the relevant part

    3.Better search, summarization, or QA system performance

### Document Structure

    Every document in this system contains two key components:

    page_content: The main textual content of the document (e.g., raw text, extracted paragraphs, or processed data).

    metadata: Additional structured information about the document, such as:

            Source file name

            Author

            Creation/modification dates

            Page numbers

            Custom tags or categories

            documents
            {
            "page_content": "Machine learning is a subset of AI...",  
            "metadata": {
                "source": "ai_handbook.pdf",  
                "page": 42,  
                "author": "Jane Doe"  
            }
            }

