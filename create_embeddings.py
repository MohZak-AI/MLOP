import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Read documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

print(f"Loaded {len(documents)} documents")

# Preprocess and tokenize
def preprocess(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove short words
    tokens = [t for t in tokens if len(t) > 2]
    return tokens

# Tokenize all documents
tokenized_docs = [preprocess(doc) for doc in documents]

print("Training Word2Vec model...")
# Train Word2Vec model
model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=384,  # 384-dimensional embeddings
    window=5,         # Context window size
    min_count=1,      # Include all words
    workers=4,
    sg=1,             # Skip-gram model
    epochs=100        # More epochs for better training
)

print(f"Vocabulary size: {len(model.wv)}")

# Create document embeddings by averaging word vectors
def get_document_embedding(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if len(vectors) == 0:
        # Return zero vector if no words found
        return np.zeros(model.vector_size)
    
    # Average the word vectors
    doc_vector = np.mean(vectors, axis=0)
    return doc_vector

# Generate embeddings for all documents
embeddings = np.array([get_document_embedding(tokens, model) for tokens in tokenized_docs])

# Normalize embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings = embeddings / norms

# Save the embeddings
np.save("embeddings.npy", embeddings.astype(np.float32))

print(f"\nCreated Word2Vec embeddings with shape: {embeddings.shape}")
print(f"Sample document 0: {documents[0][:80]}...")
print(f"Sample embedding (first 10 values): {embeddings[0][:10]}")
print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")

# Test similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings[0:1], embeddings[1:5])
print(f"\nSimilarities of doc 0 with docs 1-4:")
for i, sim in enumerate(similarities[0], 1):
    print(f"  Doc {i}: {sim:.4f}")
print("\nWord2Vec embeddings created successfully!")
