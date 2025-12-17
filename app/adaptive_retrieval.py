"""
Adaptive Retrieval Module for GovInsight

Dynamically adjusts the number of chunks to retrieve based on query complexity
and characteristics. This improves latency, precision, and explainability.

Query Classification Levels:
- Minimal (3 chunks): Simple factual queries
- Low (5 chunks): Single-entity queries
- Medium (7 chunks): Comparative or multi-entity queries
- High (10 chunks): Broad policy/trend analysis queries
"""

import re
from typing import Tuple


class QueryComplexity:
    """Query complexity levels with corresponding retrieval counts"""
    MINIMAL = "minimal"  # 3 chunks
    LOW = "low"          # 5 chunks
    MEDIUM = "medium"    # 7 chunks
    HIGH = "high"        # 10 chunks


def classify_query_complexity(query: str, max_top_k: int = 10) -> int:
    """
    Classify query complexity and return optimal number of chunks to retrieve.
    
    Uses multiple signals to determine query complexity:
    1. Query length (words/characters)
    2. Question type (who/what/when vs how/why/analyze)
    3. Comparative keywords (compare, trend, analysis)
    4. Temporal scope (single year vs multi-year)
    5. Entity count (single vs multiple ministries/schemes)
    
    Args:
        query: User query string
        max_top_k: Maximum chunks allowed (respects user-specified limit)
        
    Returns:
        Optimal number of chunks to retrieve (3-10)
        
    Examples:
        >>> classify_query_complexity("What is the Bharatmala allocation?")
        3
        >>> classify_query_complexity("Compare infrastructure spending across 3 years")
        8
        >>> classify_query_complexity("Analyze overall capital expenditure trends")
        10
    """
    query_lower = query.lower().strip()
    
    # Calculate various signals
    complexity_score = 0
    
    # Signal 1: Query length
    word_count = len(query_lower.split())
    char_count = len(query_lower)
    
    if word_count <= 6 or char_count <= 40:
        complexity_score += 0  # Short query
    elif word_count <= 12 or char_count <= 80:
        complexity_score += 1  # Medium query
    else:
        complexity_score += 2  # Long query
    
    # Signal 2: Question type
    factual_patterns = [
        r'\bwhat is\b', r'\bwho is\b', r'\bwhen\b', r'\bwhich\b',
        r'\bhow much\b', r'\bhow many\b'
    ]
    exploratory_patterns = [
        r'\bhow\b(?! much| many)', r'\bwhy\b', r'\bexplain\b',
        r'\bdescribe\b', r'\bdiscuss\b'
    ]
    
    if any(re.search(p, query_lower) for p in factual_patterns):
        complexity_score += 0  # Factual question
    elif any(re.search(p, query_lower) for p in exploratory_patterns):
        complexity_score += 2  # Exploratory question
    else:
        complexity_score += 1  # Other
    
    # Signal 3: Comparative/analytical keywords
    comparative_keywords = [
        'compare', 'comparison', 'versus', 'vs', 'difference', 'differences',
        'trend', 'trends', 'pattern', 'patterns', 'analysis', 'analyze',
        'breakdown', 'overview', 'summary', 'across', 'between',
        'change', 'changes', 'growth', 'decline', 'increase', 'decrease',
        'evolution', 'development', 'progression'
    ]
    
    comparative_count = sum(1 for kw in comparative_keywords if kw in query_lower)
    if comparative_count >= 2:
        complexity_score += 3  # Strong comparative signal
    elif comparative_count == 1:
        complexity_score += 2  # Moderate comparative signal
    
    # Signal 4: Temporal scope (multi-year queries need more context)
    year_mentions = len(re.findall(r'\b\d{4}(?:-\d{2,4})?\b', query))
    time_range_keywords = ['years', 'last', 'past', 'previous', 'since', 'from', 'to']
    has_time_range = any(kw in query_lower for kw in time_range_keywords)
    
    if year_mentions >= 2 or has_time_range:
        complexity_score += 2  # Multi-year scope
    elif year_mentions == 1:
        complexity_score += 0  # Single year
    
    # Signal 5: Entity count (multiple ministries/schemes need more context)
    entity_keywords = ['ministry', 'ministries', 'scheme', 'schemes', 'department', 'departments']
    entity_mentions = sum(1 for kw in entity_keywords if kw in query_lower)
    
    # Check for plural forms or multiple entities
    has_multiple_entities = any(kw in query_lower for kw in ['ministries', 'schemes', 'departments'])
    if has_multiple_entities or entity_mentions >= 2:
        complexity_score += 2  # Multiple entities
    
    # Signal 6: Specific allocation queries (usually need fewer chunks)
    allocation_patterns = [
        r'\ballocation\b', r'\bbudget\b', r'\bexpenditure\b',
        r'\bcrore\b', r'\blakh\b', r'\bamount\b', r'\bfunding\b'
    ]
    is_allocation_query = any(re.search(p, query_lower) for p in allocation_patterns)
    
    # If it's a simple allocation query without comparative elements, reduce score
    if is_allocation_query and comparative_count == 0 and word_count <= 10:
        complexity_score = max(0, complexity_score - 1)
    
    # Map complexity score to retrieval count
    # Score ranges: 0-2 (minimal), 3-5 (low), 6-8 (medium), 9+ (high)
    if complexity_score <= 2:
        optimal_top_k = 3  # Minimal complexity - factual queries
    elif complexity_score <= 5:
        optimal_top_k = 5  # Low complexity - single entity queries
    elif complexity_score <= 8:
        optimal_top_k = 7  # Medium complexity - comparative queries
    else:
        optimal_top_k = 10  # High complexity - broad analysis queries
    
    # Respect maximum limit
    final_top_k = min(optimal_top_k, max_top_k)
    
    return final_top_k


def get_complexity_level(top_k: int) -> str:
    """
    Convert top_k count to human-readable complexity level.
    
    Args:
        top_k: Number of chunks to retrieve
        
    Returns:
        Complexity level string
    """
    if top_k <= 3:
        return QueryComplexity.MINIMAL
    elif top_k <= 5:
        return QueryComplexity.LOW
    elif top_k <= 7:
        return QueryComplexity.MEDIUM
    else:
        return QueryComplexity.HIGH


def explain_classification(query: str, top_k: int, max_top_k: int) -> str:
    """
    Generate explanation for why a query was classified at a certain level.
    Useful for debugging and transparency.
    
    Args:
        query: User query string
        top_k: Classified retrieval count
        max_top_k: Maximum allowed
        
    Returns:
        Human-readable explanation string
    """
    complexity_level = get_complexity_level(top_k)
    query_lower = query.lower()
    
    reasons = []
    
    # Check length
    word_count = len(query.split())
    if word_count <= 6:
        reasons.append("short query")
    elif word_count > 15:
        reasons.append("long query")
    
    # Check for comparative keywords
    comparative_keywords = ['compare', 'trend', 'analysis', 'across', 'between']
    if any(kw in query_lower for kw in comparative_keywords):
        reasons.append("comparative analysis")
    
    # Check for multi-year
    year_mentions = len(re.findall(r'\b\d{4}(?:-\d{2,4})?\b', query))
    if year_mentions >= 2:
        reasons.append("multi-year scope")
    
    # Check for factual
    if re.search(r'\bwhat is\b|\bhow much\b', query_lower):
        reasons.append("factual question")
    
    reason_str = ", ".join(reasons) if reasons else "general query"
    
    explanation = (
        f"Query classified as '{complexity_level}' complexity ({reason_str}). "
        f"Retrieving {top_k} chunks (max allowed: {max_top_k})."
    )
    
    return explanation


# Example usage and test cases
if __name__ == "__main__":
    # Test cases
    test_queries = [
        ("What is the Bharatmala allocation?", 3),
        ("Budget for Ministry of Road Transport in 2024-25", 5),
        ("How much was allocated to education sector?", 3),
        ("Compare infrastructure spending between 2023-24 and 2024-25", 7),
        ("Analyze capital expenditure trends across all ministries", 10),
        ("What are the major policy changes in the Union Budget?", 8),
        ("Show me the breakdown of defense budget", 5),
        ("Explain the overall fiscal deficit trend over the last 5 years", 10),
    ]
    
    print("Testing Query Complexity Classification\n" + "="*60)
    for query, expected in test_queries:
        result = classify_query_complexity(query)
        status = "✓" if result == expected else "✗"
        explanation = explain_classification(query, result, 10)
        
        print(f"\n{status} Query: '{query}'")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  {explanation}")
