from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

<<<<<<< HEAD
=======
# Menu codebook to store in FAISS
>>>>>>> 7956c3a (Added comments to calculation tools and menu scripts)
documents = [
    "Cheeseburger: grilled beef patty with American cheese.",
    "Bacon Double: two beef patties with bacon and cheese.",
    "Veggie Wrap: fresh vegetables wrapped in a tortilla.",
    "Chicken Tenders: crispy breaded chicken tenders.",
    "Fries: golden crispy fries."
]

<<<<<<< HEAD
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings, dtype="float32"))

=======
# Load embeddding model so menu can be converted to vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert menu to vectors
embeddings = model.encode(documents)

# Add the vectors to FAISS
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings, dtype="float32"))

# Define the function the excecutes menu information retrieval

>>>>>>> 7956c3a (Added comments to calculation tools and menu scripts)

def retrieve_menu_context(query):
    query_embedding = model.encode([query])

    _, indices = index.search(
        np.array(query_embedding, dtype="float32"),
        k=3
    )

    return "\n".join(
        documents[i]
        for i in indices[0]
    )
