# ACT Embeddings and Similarity Search

This project processes Australian Corporations Act (ACT) documents, generates embeddings, and provides semantic search functionality.

## Purpose

**Note:** While a normal AI can answer legal questions just as accurately or even better than this system, this tool serves a specific purpose: **providing exact citations and source references** from the ACT documents. The value lies not in the quality of answers, but in the ability to retrieve specific sections, subsections, and exact text from the legislation for proper legal citation and reference.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### 1. Generate Embeddings

First, generate embeddings for all text files in the `rag_split` directory:

```bash
python generate_embeddings.py
```

This will:
- Process all `.txt` files in `rag_split/` subdirectories
- Generate embeddings using OpenAI's `text-embedding-3-large` model
- Save results to `embeddings.json`

### 2. Similarity Search

Search for similar documents using command line:

```bash
python similarity_search.py "your search query"
```

**Example:**
```bash
python similarity_search.py "ASIC enforcement/administration powers"
```

**Sample Output:**
```
Loaded 2092 embeddings from embeddings.json
Searching for: 'ASIC enforcement/administration powers'
Generating query embedding...
Calculating similarities...

Top 3 most similar documents:
================================================================================

1. C2025C00450VOL04/C2025C00450VOL04__parent__Part_5_of_the_Regulatory_Powers_Act.__NoDivision__NoSubdivis__000.txt
   Similarity: 0.5254
   Content length: 2544 characters

   Full Content:
   ----------------------------------------------------------------------------
   Part 5 of the Regulatory Powers Act.
   
   Infringement officer
   (2) For the purposes of Part 5 of the Regulatory Powers Act, each staff 
   member of ASIC who holds, or is acting in, an office or position 
   that is equivalent to an SES employee is an infringement officer in 
   relation to subsection 908CF(1) of this Act.
   [... full content displayed ...]
   ----------------------------------------------------------------------------
```

**Options:**
- `--top-k 5` - Return top 5 results instead of 3
- `--interactive` - Run in interactive mode

**Interactive mode:**
```bash
python similarity_search.py
```

## Files

- `generate_embeddings.py` - Generates embeddings for all text files
- `similarity_search.py` - Performs semantic search using cosine similarity
- `requirements.txt` - Python dependencies
- `embeddings.json` - Generated embeddings data (created after running generate_embeddings.py)
- `rag_split/` - Directory containing split ACT text files