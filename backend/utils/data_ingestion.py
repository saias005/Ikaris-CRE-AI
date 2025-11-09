# backend/utils/data_ingestion.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pdf_processor import process_all_pdfs
from typing import List
import warnings

warnings.filterwarnings("ignore")


def ingest_all_data(
    pdf_folder: str = "./backend/data/pdfs",
    vectorstore_path: str = "./backend/chroma_db"
) -> Chroma:
    """
    Complete data ingestion pipeline:
    1. Extract PDFs
    2. Chunk text
    3. Create embeddings
    4. Store in vector database
    """
    
    print("=" * 60)
    print("🚀 STARTING DATA INGESTION PIPELINE")
    print("=" * 60)
    
    # ========================================
    # STEP 1: Extract PDFs
    # ========================================
    print("\n📖 STEP 1: Extracting PDFs...")
    pdf_docs = process_all_pdfs(pdf_folder)
    
    if not pdf_docs:
        raise ValueError("No PDF documents found!")
    
    print(f"✅ Loaded {len(pdf_docs)} PDF documents")
    
    # ========================================
    # STEP 2: Chunk Documents
    # ========================================
    print("\n✂️  STEP 2: Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    all_chunks = []
    for doc in pdf_docs:
        # Add note about charts (Option 1 approach)
        content = doc["content"]
        
        # Split into chunks
        text_chunks = text_splitter.split_text(content)
        
        # Convert to LangChain Document format
        for i, chunk_text in enumerate(text_chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": doc["source"],
                    "chunk_id": i,
                    "total_chunks": len(text_chunks),
                    "path": doc["path"]
                }
            )
            all_chunks.append(chunk_doc)
        
        print(f"   📄 {doc['source']}: {len(text_chunks)} chunks")
    
    print(f"✅ Created {len(all_chunks)} total chunks")
    
    # ========================================
    # STEP 3: Create Embeddings
    # ========================================
    print("\n🧠 STEP 3: Creating embeddings...")
    print("   (This may take a few minutes...)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("✅ Embedding model loaded")
    
    # ========================================
    # STEP 4: Store in Vector Database
    # ========================================
    print("\n💾 STEP 4: Building vector database...")
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    
    print(f"✅ Vector database saved to: {vectorstore_path}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("✅ DATA INGESTION COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   • PDFs processed: {len(pdf_docs)}")
    print(f"   • Total chunks: {len(all_chunks)}")
    print(f"   • Database location: {vectorstore_path}")
    print(f"   • Ready for queries!")
    print("=" * 60)
    
    return vectorstore


if __name__ == "__main__":
    # Run the complete ingestion pipeline
    vectorstore = ingest_all_data()
    
    # Test a simple query
    print("\n🧪 Testing vector database...")
    results = vectorstore.similarity_search("Dallas properties energy costs", k=3)
    
    print(f"\nTest query returned {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:200]}...")