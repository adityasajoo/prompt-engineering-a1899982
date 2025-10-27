from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_semantic_score(response: str, target: str) -> float:
    """Calculate similarity score using TF-IDF and cosine similarity"""
    response = response.lower().strip()
    target = target.lower().strip()
    
    if not response or not target:
        return 0.0
    
    if response == target:
        return 1.0
    
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([response, target])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)
    except:
        response_tokens = set(response.split())
        target_tokens = set(target.split())
        
        if len(target_tokens) == 0:
            return 0.0
        
        overlap = len(response_tokens & target_tokens) / len(target_tokens)
        return min(overlap, 1.0)
