#!/usr/bin/env python3
import os
import tiktoken
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

def tokenize_text(text, model="cl100k_base"):
    """Tokenize text using tiktoken."""
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    return tokens

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
        print(f"\n=== Processing: {pdf_file.name} ===")
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            print("No text extracted from this PDF")
            continue
        
        # Tokenize
        tokens = tokenize_text(text)
        
        print(f"Text length: {len(text)} characters")
        print(f"Token count: {len(tokens)} tokens")
        print(f"First 10 tokens: {tokens[:10]}")
        print("-" * 50)

if __name__ == "__main__":
    main()