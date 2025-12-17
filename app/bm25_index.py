"""
BM25 Index Module for Hybrid Retrieval

Implements sparse keyword search using BM25 algorithm for government documents.
Particularly effective for:
- Acronyms (MoRTH, PMGSY, NITI)
- Exact scheme names
- Budget numbers and tables
- Legal/formal language

Used alongside dense embeddings for hybrid retrieval.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text while preserving important patterns for government documents.
    
    Preserves:
    - Acronyms (e.g., "MoRTH", "PMGSY")
    - Numbers with units (e.g., "₹10,000", "10000 crore")
    - Year formats (e.g., "2024-25", "2023-2024")
    - Hyphenated terms (e.g., "road-transport")
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Lowercase for matching but preserve structure
    text_lower = text.lower()
    
    # Replace special currency symbols with text
    text_lower = text_lower.replace('₹', 'rs ')
    
    # Split on whitespace and common separators, but keep hyphens in numbers/years
    # This regex splits on whitespace and punctuation except hyphens in specific contexts
    tokens = re.findall(r'\b[\w]+-[\w]+\b|\b\w+\b', text_lower)
    
    # Additional processing
    processed_tokens = []
    for token in tokens:
        # Keep token as-is
        processed_tokens.append(token)
        
        # For numbers with commas, also add version without commas
        if ',' in token:
            no_comma = token.replace(',', '')
            if no_comma != token:
                processed_tokens.append(no_comma)
        
        # For hyphenated terms, also add individual parts
        if '-' in token and not re.match(r'^\d{4}-\d{2,4}$', token):  # Not a year
            parts = token.split('-')
            processed_tokens.extend(parts)
    
    return [t for t in processed_tokens if len(t) > 1]  # Filter single chars


class BM25Index:
    """
    BM25-based keyword search index for government budget documents.
    """
    
    def __init__(self, chunks: Optional[List[Dict]] = None):
        """
        Initialize BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' fields
                   If None, creates empty index (for loading from disk)
        """
        self.chunks = chunks or []
        self.bm25 = None
        self.tokenized_corpus = []
        
        if chunks:
            self._build_index()
    
    def _build_index(self):
        """Build BM25 index from chunks"""
        print(f"Building BM25 index from {len(self.chunks)} chunks...")
        
        # Tokenize all chunks
        self.tokenized_corpus = [
            tokenize_for_bm25(chunk.get('text', ''))
            for chunk in self.chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"BM25 index built: {len(self.tokenized_corpus)} documents indexed")
    
    def search(
        self,
        query: str,
        top_k: int = 15,
        year_filter: Optional[str] = None,
        ministry_filter: Optional[str] = None,
        scheme_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for chunks using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            year_filter: Optional year filter (e.g., "2024-25")
            ministry_filter: Optional ministry filter
            scheme_filter: Optional scheme filter
            
        Returns:
            List of chunk dictionaries with BM25 scores
        """
        if not self.bm25 or not self.chunks:
            print("⚠️  BM25 index is empty or not built")
            return []
        
        # Tokenize query
        tokenized_query = tokenize_for_bm25(query)
        
        if not tokenized_query:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create (index, score) pairs and filter by metadata
        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            metadata = chunk.get('metadata', {}) or {}
            
            # Apply metadata filters
            if year_filter and metadata.get('year') != year_filter:
                continue
            if ministry_filter and metadata.get('ministry') != ministry_filter:
                continue
            if scheme_filter:
                if scheme_filter.lower() == "none" and metadata.get('scheme') != "None":
                    continue
                elif scheme_filter.lower() != "none" and metadata.get('scheme') != scheme_filter:
                    continue
            
            # Only include chunks with non-zero scores
            if score > 0:
                scored_chunks.append({
                    **chunk,
                    'bm25_score': float(score),
                    'distance': 1.0 - min(score / 10.0, 1.0)  # Convert to distance-like metric
                })
        
        # Sort by BM25 score (descending)
        scored_chunks.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        # Return top_k results
        return scored_chunks[:top_k]
    
    def save(self, filepath: str):
        """
        Save BM25 index to disk.
        
        Args:
            filepath: Path to save the index
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save everything needed to reconstruct the index
        data = {
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'bm25': self.bm25
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BM25 index saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BM25Index':
        """
        Load BM25 index from disk.
        
        Args:
            filepath: Path to the saved index
            
        Returns:
            Loaded BM25Index instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance and restore state
        instance = cls(chunks=None)
        instance.chunks = data['chunks']
        instance.tokenized_corpus = data['tokenized_corpus']
        instance.bm25 = data['bm25']
        
        print(f"BM25 index loaded from: {filepath}")
        print(f"  {len(instance.chunks)} chunks indexed")
        
        return instance
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the BM25 index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'num_documents': len(self.chunks),
            'avg_document_length': sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus) if self.tokenized_corpus else 0,
            'total_tokens': sum(len(doc) for doc in self.tokenized_corpus),
            'index_built': self.bm25 is not None
        }


# Example usage and testing
if __name__ == "__main__":
    # Test tokenization
    print("Testing BM25 Tokenization\n" + "="*60)
    
    test_texts = [
        "Ministry of Road Transport & Highways (MoRTH)",
        "Bharatmala Pariyojana allocated ₹10,000 crore",
        "Budget 2024-25 shows 15% increase",
        "PMGSY scheme continues with Rs. 5,000 crore allocation",
    ]
    
    for text in test_texts:
        tokens = tokenize_for_bm25(text)
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens}")
    
    # Test BM25 index
    print("\n\nTesting BM25 Index\n" + "="*60)
    
    sample_chunks = [
        {
            'text': "Ministry of Road Transport and Highways (MoRTH) receives ₹2,78,000 crore for 2024-25",
            'metadata': {'year': '2024-25', 'ministry': 'MoRTH', 'scheme': 'General'},
            'id': 'chunk_0'
        },
        {
            'text': "Bharatmala Pariyojana is allocated Rs. 10,000 crore in the union budget",
            'metadata': {'year': '2024-25', 'ministry': 'MoRTH', 'scheme': 'Bharatmala'},
            'id': 'chunk_1'
        },
        {
            'text': "PMGSY (Pradhan Mantri Gram Sadak Yojana) gets 5000 crore allocation",
            'metadata': {'year': '2024-25', 'ministry': 'MoRTH', 'scheme': 'PMGSY'},
            'id': 'chunk_2'
        },
        {
            'text': "Total infrastructure spending increased by 15 percent to 50,000 crore",
            'metadata': {'year': '2024-25', 'ministry': 'Finance', 'scheme': 'General'},
            'id': 'chunk_3'
        },
    ]
    
    # Build index
    index = BM25Index(sample_chunks)
    
    # Test queries
    test_queries = [
        ("MoRTH budget", "Should match acronym"),
        ("Bharatmala allocation", "Should match scheme name"),
        ("10000 crore", "Should match number"),
        ("PMGSY", "Should match acronym exactly"),
        ("infrastructure spending", "Should match semantic terms"),
    ]
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}' - {description}")
        results = index.search(query, top_k=3)
        print(f"Found {len(results)} results:")
        for i, chunk in enumerate(results, 1):
            print(f"  [{i}] Score: {chunk['bm25_score']:.2f} - {chunk['text'][:80]}...")
    
    # Test filtering
    print("\n\nTesting Metadata Filtering\n" + "="*60)
    results = index.search("allocation", top_k=5, ministry_filter="MoRTH")
    print(f"Query: 'allocation' with ministry_filter='MoRTH'")
    print(f"Found {len(results)} results (should only include MoRTH)")
    for chunk in results:
        print(f"  - {chunk['metadata']['ministry']}: {chunk['text'][:60]}...")
