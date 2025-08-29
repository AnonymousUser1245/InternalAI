#!/usr/bin/env python3
import PyPDF2
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
    return text

def main():
    pdf_folder = Path("/Users/blakeandrews/Desktop/InternalAI/Corporations-Act-2001")
    
    if not pdf_folder.exists():
        print(f"Error: Folder {pdf_folder} does not exist")
        return
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the folder")
        return
    
    for pdf_file in sorted(pdf_files):
        print(f"Processing: {pdf_file.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_file.name}")
            continue
        
        # Create output filename (same name but with .txt extension)
        txt_file = pdf_file.with_suffix('.txt')
        
        # Save text to file
        try:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved: {txt_file.name} ({len(text)} characters)")
        except Exception as e:
            print(f"Error saving {txt_file.name}: {e}")

if __name__ == "__main__":
    main()