import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

print("1. Loading and cleaning data...")
df = pd.read_csv('data/raw_reviews.csv')

# Drop empty reviews
df = df.dropna(subset=['content'])

# Drop extremely short reviews (e.g., "ok", "bad app") that don't add value
df = df[df['content'].str.len() > 15]
print(f"Cleaned data down to {len(df)} useful reviews.")

print("2. Initializing Chroma Vector Database...")
# Store the database locally in a folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# We use a free, lightweight open-source embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create a collection (like a table in a relational database)
collection = chroma_client.get_or_create_collection(
    name="app_reviews",
    embedding_function=sentence_transformer_ef
)

print("3. Embedding and indexing reviews (This will take a few minutes)...")
# Prepare data arrays for Chroma
documents = df['content'].tolist()
# Create unique IDs using the app name and review ID
ids = [f"{row['app_name']}_{row['review_id']}" for _, row in df.iterrows()]
# Store metadata so we can filter searches later (e.g., "only search ChatGPT reviews")
metadatas = [{'app_name': row['app_name'], 'score': row['score'], 'date': row['date']} for _, row in df.iterrows()]

# Add to the database in batches to prevent your computer from freezing or running out of memory
batch_size = 500
for i in range(0, len(documents), batch_size):
    print(f"   Indexing batch {i} to {min(i + batch_size, len(documents))}...")
    collection.add(
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

print("\nSuccess! Vector database built and saved to the 'chroma_db' folder.")