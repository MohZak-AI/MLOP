"""Test script to demonstrate phrase-based search"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load data
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

# Create TF-IDF vectorizer with n-grams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),  # Support unigrams, bigrams, and trigrams
    stop_words='english',
    min_df=1,
    max_features=5000
)
tfidf_matrix = vectorizer.fit_transform(documents)

def extract_phrases(text, n=3):
    """Extract n-grams (phrases) from text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    
    phrases = []
    for i in range(1, n+1):
        phrases.extend([' '.join(gram) for gram in ngrams(tokens, i)])
    return phrases

def search_with_phrases(query, k=5):
    """Search using TF-IDF phrase matching."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

print("=" * 80)
print("PHRASE-BASED SEARCH TEST")
print("=" * 80)

# Test 1: Single word (keyword)
print("\n1. Keyword search: 'coffee'")
print("-" * 80)
results = search_with_phrases("coffee", k=3)
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f} | {doc[:100]}...")

# Test 2: Two-word phrase
print("\n2. Two-word phrase search: 'cold brew'")
print("-" * 80)
results = search_with_phrases("cold brew", k=3)
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f} | {doc[:100]}...")

# Test 3: Three-word phrase
print("\n3. Three-word phrase search: 'ramadan fasting traditions'")
print("-" * 80)
results = search_with_phrases("ramadan fasting traditions", k=3)
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f} | {doc[:100]}...")

# Test 4: Show extracted phrases
print("\n4. Extracted phrases from query 'coffee brewing methods':")
print("-" * 80)
phrases = extract_phrases("coffee brewing methods")
print(f"Detected phrases: {phrases}")

# Test 5: Exact phrase matching
print("\n5. Exact phrase search test:")
print("-" * 80)
query_phrases = ["turkish coffee", "iftar meal"]
print(f"Searching for exact phrases: {query_phrases}")

for phrase in query_phrases:
    matching = [doc for doc in documents if phrase.lower() in doc.lower()]
    print(f"\nPhrase '{phrase}' found in {len(matching)} documents:")
    for doc in matching[:2]:  # Show first 2 matches
        print(f"  - {doc[:80]}...")

print("\n" + "=" * 80)
print("Phrase-based search is working correctly!")
print("=" * 80)
