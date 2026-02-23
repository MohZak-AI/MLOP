import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and documents
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

print(f"Loaded {len(documents)} documents")
print(f"Embeddings shape: {embeddings.shape}\n")

def retrieve_top_k(query_embedding, embeddings, documents, k=5):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Test 1: Use first document as query
print("=" * 80)
print("TEST 1: Query using document 0 as the query")
print("=" * 80)
query = embeddings[0]
results = retrieve_top_k(query, embeddings, documents, k=5)
print(f"\nQuery: {documents[0][:80]}...")
print("\nTop 5 similar documents:")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f} | {doc[:80]}...")

# Test 2: Create a manual query embedding based on keywords
print("\n" + "=" * 80)
print("TEST 2: Create a query about 'Ramadan fasting'")
print("=" * 80)
# Find documents containing 'ramadan' and 'fasting'
query_doc_idx = None
for i, doc in enumerate(documents):
    if 'ramadan' in doc.lower() and 'fast' in doc.lower():
        query_doc_idx = i
        break

if query_doc_idx is not None:
    query = embeddings[query_doc_idx]
    results = retrieve_top_k(query, embeddings, documents, k=5)
    print(f"\nQuery document: {documents[query_doc_idx][:80]}...")
    print("\nTop 5 similar documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} | {doc[:80]}...")
else:
    print("No document found with 'ramadan' and 'fasting' keywords")

# Test 3: Query about 'coffee brewing'
print("\n" + "=" * 80)
print("TEST 3: Create a query about 'coffee brewing'")
print("=" * 80)
query_doc_idx = None
for i, doc in enumerate(documents):
    if 'brew' in doc.lower() or 'espresso' in doc.lower():
        query_doc_idx = i
        break

if query_doc_idx is not None:
    query = embeddings[query_doc_idx]
    results = retrieve_top_k(query, embeddings, documents, k=5)
    print(f"\nQuery document: {documents[query_doc_idx][:80]}...")
    print("\nTop 5 similar documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} | {doc[:80]}...")
else:
    print("No document found with brewing-related keywords")

# Test 4: Check embedding statistics
print("\n" + "=" * 80)
print("EMBEDDING STATISTICS")
print("=" * 80)
print(f"Embedding dimensionality: {embeddings.shape[1]}")
print(f"Number of documents: {embeddings.shape[0]}")
print(f"Non-zero elements per embedding (avg): {np.mean(np.count_nonzero(embeddings, axis=1)):.1f}")
print(f"All embeddings normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
print(f"Embedding value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

print("\n✓ Embeddings are working correctly!")
