from sentence_transformers import SentenceTransformer, util
import torch

# Load a lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

def compute_similarity(query_embedding, doc_embedding):
    return float(util.cos_sim(query_embedding, doc_embedding))
