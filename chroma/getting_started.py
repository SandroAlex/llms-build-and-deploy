#!/usr/bin/env python3

# Load packages.
import chromadb

# Create a client to interact with the database.
chroma_client = chromadb.HttpClient(host="chroma-service", port=8878)

# Collections are where you'll store your embeddings, documents, and any additional metadata. 
# Collections index your embeddings and documents, and enable efficient retrieval and filtering. 
# You can create a collection with a name:
collection = chroma_client.create_collection(name="alex-collection")

# Chroma will store your text and handle embedding and indexing automatically. 
# You can also customize the embedding model. 
# You must provide unique string IDs for your documents.
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

# You can query the collection with a list of query texts, and Chroma will return the n most similar results. 
# It's that easy!
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you.
    n_results=2                                            # How many results to return.
)

print(results)