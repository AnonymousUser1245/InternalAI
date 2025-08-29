#!/usr/bin/env python3
"""
Similarity search program that takes a text string input and returns 
the top 3 most similar documents using cosine similarity.
"""

import os
import json
import sys
import argparse
import numpy as np
import requests
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBEDDINGS_FILE = "embeddings.json"
RAG_SPLIT_DIR = "rag_split"
OPENAI_MODEL = "text-embedding-3-large"
TOP_K = 3

class SimilaritySearcher:
    def __init__(self, api_key: str = None):
        """Initialize the similarity searcher with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load embeddings data
        self.embeddings_data = self.load_embeddings()
        print(f"Loaded {len(self.embeddings_data)} embeddings from {EMBEDDINGS_FILE}")
    
    def load_embeddings(self) -> List[Dict[str, Any]]:
        """Load embeddings from the JSON file."""
        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Embeddings file {EMBEDDINGS_FILE} not found. Run generate_embeddings.py first.")
        
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter out entries without embeddings
        valid_embeddings = [item for item in data.get("embeddings", []) if item.get("has_embedding", False)]
        
        return valid_embeddings
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for the query text using OpenAI API."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        payload = {
            "model": OPENAI_MODEL,
            "input": query,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"No embedding data returned for query: {query[:100]}...")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except KeyError as e:
            raise RuntimeError(f"Unexpected API response format: {e}")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def read_file_content(self, filename: str) -> str:
        """Read the actual content from the text file."""
        file_path = Path(RAG_SPLIT_DIR) / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read content from {file_path}: {e}")
            return "Content not available"
    
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Search for the most similar documents to the query.
        
        Args:
            query: The search query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing filename, similarity score, and content
        """
        print(f"Searching for: '{query}'")
        print("Generating query embedding...")
        
        # Create embedding for the query
        query_embedding = self.create_query_embedding(query)
        
        print("Calculating similarities...")
        
        # Calculate similarities with all documents
        similarities = []
        for item in self.embeddings_data:
            if not item.get("embedding"):
                continue
            
            similarity = self.cosine_similarity(query_embedding, item["embedding"])
            similarities.append({
                "filename": item["filename"],
                "similarity": similarity,
                "content_length": item.get("content_length", 0)
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get top k results
        top_results = similarities[:top_k]
        
        # Add actual content to results
        results = []
        for result in top_results:
            content = self.read_file_content(result["filename"])
            results.append({
                "filename": result["filename"],
                "similarity": result["similarity"],
                "content": content,
                "content_length": len(content)
            })
        
        return results
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Print search results in a formatted way with full content."""
        if not results:
            print("No results found.")
            return
        
        print(f"\nTop {len(results)} most similar documents:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Content length: {result['content_length']} characters")
            print(f"\n   Full Content:")
            print("   " + "-" * 76)
            # Print full content with proper indentation
            content_lines = result['content'].split('\n')
            for line in content_lines:
                print(f"   {line}")
            print("   " + "-" * 76)
            if i < len(results):
                print("\n" + "=" * 80)

def main():
    """Main function to run similarity search with command line arguments."""
    parser = argparse.ArgumentParser(description="Search for similar documents using embeddings")
    parser.add_argument("query", nargs='?', help="Search query text")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K, help="Number of top results to return")
    
    args = parser.parse_args()
    
    try:
        searcher = SimilaritySearcher()
        
        # Command line mode
        if args.query:
            try:
                results = searcher.search(args.query, args.top_k)
                searcher.print_results(results)
                return 0
                
            except Exception as e:
                print(f"Search error: {e}")
                return 1
        
        # Interactive mode (default if no query provided)
        else:
            print("Similarity Search Tool")
            print("=" * 30)
            print("Enter a search query to find the most similar documents.")
            print("Type 'quit' or 'exit' to stop.\n")
            
            while True:
                query = input("Enter your search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    print("Please enter a valid query.")
                    continue
                
                try:
                    results = searcher.search(query, args.top_k)
                    searcher.print_results(results)
                    print()
                    
                except Exception as e:
                    print(f"Search error: {e}")
                    print()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())