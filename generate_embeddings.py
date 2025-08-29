#!/usr/bin/env python3
"""
Script to generate embeddings for all text files in the rag_split directory.
Processes each text file in batches, creates embeddings using OpenAI API, 
and stores results in a JSON file.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Configuration
RAG_SPLIT_DIR = "rag_split"
OUTPUT_FILE = "embeddings.json"
OPENAI_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100  # Process files in batches to avoid overwhelming the API

class EmbeddingGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the embedding generator with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.embeddings_data = []
    
    def get_text_files(self) -> List[Path]:
        """Get all text files from the rag_split directory."""
        rag_split_path = Path(RAG_SPLIT_DIR)
        if not rag_split_path.exists():
            raise FileNotFoundError(f"Directory {RAG_SPLIT_DIR} not found")
        
        text_files = []
        for folder in rag_split_path.iterdir():
            if folder.is_dir():
                for file_path in folder.glob("*.txt"):
                    text_files.append(file_path)
        
        print(f"Found {len(text_files)} text files to process")
        return text_files
    
    def read_file_content(self, file_path: Path) -> str:
        """Read content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts using OpenAI API."""
        payload = {
            "model": OPENAI_MODEL,
            "input": texts,
            "encoding_format": "float"
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result.get("data", [])]
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return [[] for _ in texts]
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return [[] for _ in texts]

    def process_all_files(self):
        """Process all text files and generate embeddings in batches."""
        text_files = self.get_text_files()
        print(f"Starting to process {len(text_files)} files in batches of {BATCH_SIZE}...")

        # Collect contents and filenames
        file_contents = []
        filenames = []
        for file_path in text_files:
            content = self.read_file_content(file_path)
            filenames.append(str(file_path.relative_to(RAG_SPLIT_DIR)))
            file_contents.append(content)

        # Process in batches
        for i in tqdm(range(0, len(file_contents), BATCH_SIZE), desc="Processing batches"):
            batch_texts = file_contents[i:i + BATCH_SIZE]
            batch_files = filenames[i:i + BATCH_SIZE]
            embeddings = self.create_embeddings_batch(batch_texts)

            for fname, content, emb in zip(batch_files, batch_texts, embeddings):
                self.embeddings_data.append({
                    "filename": fname,
                    "content_length": len(content),
                    "embedding": emb,
                    "has_embedding": len(emb) > 0
                })

            # Small delay to respect rate limits
            time.sleep(0.5)
    
    def save_embeddings(self):
        """Save all embeddings to JSON file."""
        output_data = {
            "metadata": {
                "total_files": len(self.embeddings_data),
                "model_used": OPENAI_MODEL,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "successful_embeddings": sum(1 for item in self.embeddings_data if item.get("has_embedding", False))
            },
            "embeddings": self.embeddings_data
        }
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nEmbeddings saved to {OUTPUT_FILE}")
        print(f"Total files processed: {output_data['metadata']['total_files']}")
        print(f"Successful embeddings: {output_data['metadata']['successful_embeddings']}")

def main():
    try:
        generator = EmbeddingGenerator()
        generator.process_all_files()
        generator.save_embeddings()
        print("\nEmbedding generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
