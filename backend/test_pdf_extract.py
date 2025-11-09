# test_pdf_processor.py
from utils.pdf_processor import process_all_pdfs

if __name__ == "__main__":
    # Make sure you have PDFs in backend/data/pdfs/
    docs = process_all_pdfs("./backend/data/pdfs")
    
    # Print first document preview
    if docs:
        print(f"\n📄 First document: {docs[0]['source']}")
        print(f"📝 Content preview: {docs[0]['content'][:200]}...")
        print(f"📊 Total documents: {len(docs)}")