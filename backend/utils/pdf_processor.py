import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Union
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def process_all_pdfs(pdf_folder: Optional[Union[str, Path]] = None) -> List[Dict]:
    """Extract text from all PDFs in folder"""
    
    # Auto-detect path based on where script is run from
    if pdf_folder is None:
        # Try to find data/pdfs from current location
        if Path("./backend/data/pdfs").exists():
            pdf_folder = Path("./backend/data/pdfs")
        elif Path("./data/pdfs").exists():
            pdf_folder = Path("./data/pdfs")
        else:
            # Try one more - maybe we're in backend already
            if Path("../data/pdfs").exists():
                pdf_folder = Path("../data/pdfs")
            else:
                raise FileNotFoundError(
                    "Could not find data/pdfs folder. "
                    "Tried: ./backend/data/pdfs, ./data/pdfs, ../data/pdfs"
                )
    else:
        pdf_folder = Path(pdf_folder)
    
    # Rest of the code stays the same...
    
    # Debug info
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Looking for PDFs in: {pdf_folder.absolute()}")
    print(f"🔍 Folder exists: {pdf_folder.exists()}")
    
    if not pdf_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {pdf_folder.absolute()}")
    
    documents = []
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    print(f"🔍 Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"  📄 Processing: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        documents.append({
            "content": text,
            "source": pdf_file.name,
            "path": str(pdf_file)
        })
    
    print(f"✅ Processed {len(documents)} PDF files")
    return documents