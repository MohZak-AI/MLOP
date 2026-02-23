import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="Coffee & Ramadan Document Search",
    page_icon="☕",
    layout="wide"
)

# Load precomputed document embeddings
@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
    return embeddings, documents

# Load TF-IDF vectorizer for phrase matching
@st.cache_resource
def load_tfidf_vectorizer():
    """Create TF-IDF vectorizer for phrase-based search."""
    _, documents = load_data()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Support unigrams, bigrams, and trigrams
        stop_words='english',
        min_df=1,
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Load Word2Vec model for query embedding
@st.cache_resource
def load_word2vec_model():
    """Recreate Word2Vec model from documents for query embedding."""
    embeddings, documents = load_data()
    
    def preprocess(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    tokenized_docs = [preprocess(doc) for doc in documents]
    
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=384,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=100
    )
    return model

embeddings, documents = load_data()
word2vec_model = load_word2vec_model()
tfidf_vectorizer, tfidf_matrix = load_tfidf_vectorizer()

def extract_phrases(text, n=3):
    """Extract n-grams (phrases) from text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    
    phrases = []
    for i in range(1, n+1):
        phrases.extend([' '.join(gram) for gram in ngrams(tokens, i)])
    return phrases

def get_query_embedding(query):
    """Convert query text to embedding using Word2Vec."""
    def preprocess(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    tokens = preprocess(query)
    vectors = []
    for token in tokens:
        if token in word2vec_model.wv:
            vectors.append(word2vec_model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(384)
    
    query_vector = np.mean(vectors, axis=0)
    # Normalize
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm
    return query_vector

def search_with_phrases(query, documents, k=10):
    """Search using TF-IDF phrase matching."""
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

def hybrid_search(query, documents, semantic_embeddings, k=10, semantic_weight=0.5):
    """Combine phrase-based (TF-IDF) and semantic (Word2Vec) search."""
    # Phrase-based scores
    phrase_results = search_with_phrases(query, documents, k=len(documents))
    phrase_scores = {i: score for i, (doc, score) in enumerate(phrase_results)}
    
    # Semantic scores
    query_embedding = get_query_embedding(query)
    if np.all(query_embedding == 0):
        # If no semantic match, use phrase-only
        return phrase_results[:k]
    
    semantic_sims = cosine_similarity(query_embedding.reshape(1, -1), semantic_embeddings).flatten()
    
    # Combine scores
    combined_scores = []
    for i in range(len(documents)):
        combined_score = (semantic_weight * semantic_sims[i] + 
                         (1 - semantic_weight) * phrase_scores.get(i, 0))
        combined_scores.append((i, combined_score))
    
    # Sort by combined score
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [(documents[i], score) for i, score in combined_scores[:k]]

def retrieve_top_k(query_embedding, embeddings, documents, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Main UI
st.title("☕ Coffee & Ramadan Document Search")
st.markdown("Search through 50 documents about coffee culture and Ramadan traditions using semantic similarity.")

# Sidebar for search options
st.sidebar.header("Search Options")
search_type = st.sidebar.radio(
    "Search by:",
    ["Enter Query", "Select Document", "Search by Phrase"]
)

num_results = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)

# Search mode selection
search_mode = st.sidebar.radio(
    "Search mode:",
    ["Hybrid (Phrases + Semantic)", "Semantic Only (Word2Vec)", "Phrase Only (TF-IDF)"]
)

semantic_weight = 0.5
if search_mode == "Hybrid (Phrases + Semantic)":
    semantic_weight = st.sidebar.slider("Semantic weight", 0.0, 1.0, 0.5, 0.1)

# Main content area
if search_type == "Enter Query":
    st.subheader("Enter your search query:")
    st.markdown("💡 **Tip:** You can search using phrases like 'coffee brewing methods' or 'ramadan fasting traditions'")
    
    query = st.text_input("Type your query:", placeholder="e.g., 'coffee brewing methods', 'ramadan fasting tips'")
    
    if query and st.button("Search"):
        with st.spinner("Searching..."):
            # Show extracted phrases
            phrases = extract_phrases(query)
            st.caption(f"Detected phrases: {', '.join(phrases[:5])}...")
            
            # Perform search based on selected mode
            results = None
            if search_mode == "Semantic Only (Word2Vec)":
                query_embedding = get_query_embedding(query)
                if np.all(query_embedding == 0):
                    st.error("Query doesn't match any known words in the vocabulary. Try different keywords.")
                else:
                    results = retrieve_top_k(query_embedding, embeddings, documents, k=num_results)
                    st.success(f"Found {num_results} most relevant documents (Semantic Search):")
            elif search_mode == "Phrase Only (TF-IDF)":
                results = search_with_phrases(query, documents, k=num_results)
                st.success(f"Found {num_results} most relevant documents (Phrase Search):")
            else:  # Hybrid
                results = hybrid_search(query, documents, embeddings, k=num_results, semantic_weight=semantic_weight)
                st.success(f"Found {num_results} most relevant documents (Hybrid Search):")
            
            # Display results
            if results is not None:
                for i, (doc, score) in enumerate(results, 1):
                    with st.container():
                        col1, col2 = st.columns([1, 10])
                        with col1:
                            st.metric(f"Rank {i}", f"{score:.4f}")
                        with col2:
                            # Highlight matching phrases
                            highlighted_doc = doc
                            for phrase in phrases[:3]:  # Highlight top 3 phrases
                                if phrase.lower() in doc.lower():
                                    highlighted_doc = highlighted_doc.replace(
                                        phrase, f"**{phrase}**"
                                    ).replace(
                                        phrase.capitalize(), f"**{phrase.capitalize()}**"
                                    )
                            st.info(highlighted_doc)
                        st.divider()

elif search_type == "Select Document":
    st.subheader("Select a document to find similar ones:")
    
    # Create a selectbox with document previews
    doc_options = [f"{i+1}. {doc[:80]}..." for i, doc in enumerate(documents)]
    selected_idx = st.selectbox("Choose a document:", range(len(documents)), format_func=lambda x: doc_options[x])
    
    if st.button("Find Similar Documents"):
        with st.spinner("Searching..."):
            query_embedding = embeddings[selected_idx]
            results = retrieve_top_k(query_embedding, embeddings, documents, k=num_results)
            
            st.success(f"Found {num_results} most similar documents:")
            
            for i, (doc, score) in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        st.metric(f"Rank {i}", f"{score:.4f}")
                    with col2:
                        if i == 1 and doc == documents[selected_idx]:
                            st.success(f"**QUERY DOCUMENT:** {doc}")
                        else:
                            st.info(doc)
                    st.divider()

else:  # Search by Phrase
    st.subheader("Search by Specific Phrases")
    st.markdown("💡 **Tip:** Enter exact phrases you want to find. Use quotes for multi-word phrases like \"cold brew coffee\"")
    
    phrase_query = st.text_input("Enter phrase(s):", placeholder="e.g., 'cold brew', 'ramadan fasting', 'espresso bar'")
    exact_match = st.checkbox("Require exact phrase match", value=False)
    
    if phrase_query and st.button("Search by Phrase"):
        with st.spinner("Searching for phrases..."):
            # Extract phrases from query
            query_phrases = [p.strip().lower() for p in phrase_query.split(',')]
            
            # Search for documents containing these phrases
            matching_docs = []
            for i, doc in enumerate(documents):
                doc_lower = doc.lower()
                match_count = 0
                matched_phrases = []
                
                for phrase in query_phrases:
                    if exact_match:
                        # Require exact phrase match
                        if phrase in doc_lower:
                            match_count += 1
                            matched_phrases.append(phrase)
                    else:
                        # Check if all words in phrase appear in document
                        words = phrase.split()
                        if all(word in doc_lower for word in words):
                            match_count += 1
                            matched_phrases.append(phrase)
                
                if match_count > 0:
                    score = match_count / len(query_phrases)
                    matching_docs.append((i, doc, score, matched_phrases))
            
            # Sort by score
            matching_docs.sort(key=lambda x: x[2], reverse=True)
            
            if matching_docs:
                st.success(f"Found {len(matching_docs)} documents with matching phrases:")
                
                for rank, (idx, doc, score, phrases) in enumerate(matching_docs[:num_results], 1):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 8, 2])
                        with col1:
                            st.metric(f"Rank {rank}", f"{score:.1%}")
                        with col2:
                            # Highlight matched phrases
                            highlighted = doc
                            for phrase in phrases:
                                highlighted = highlighted.replace(
                                    phrase, f"**{phrase}**"
                                ).replace(
                                    phrase.capitalize(), f"**{phrase.capitalize()}**"
                                )
                            st.info(highlighted)
                        with col3:
                            st.caption(f"Matched: {', '.join(phrases)}")
                        st.divider()
            else:
                st.warning("No documents found with the specified phrases. Try different phrases or disable exact match.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app supports multiple search methods:")
st.sidebar.markdown("• **Semantic Search** - Word2Vec embeddings for meaning")
st.sidebar.markdown("• **Phrase Search** - TF-IDF with n-grams (1-3 words)")
st.sidebar.markdown("• **Hybrid Search** - Combines both approaches")
st.sidebar.markdown(f"**Total Documents:** {len(documents)}")
st.sidebar.markdown(f"**Embedding Dimension:** {embeddings.shape[1]}")

# Display sample documents
with st.sidebar.expander("View Sample Documents"):
    for i in range(min(5, len(documents))):
        st.write(f"{i+1}. {documents[i][:60]}...")
