import os
import json
from datetime import datetime
from pdf_parser import extract_sections
from model_utils import embed_text, compute_similarity

# === INPUTS ===
persona = "Travel Planner"
job = "Plan a trip of 4 days for a group of 10 college friends."
input_dir = "sample_pdfs"
output_file = "output.json"
top_k = 5

# === EMBED QUERY ===
query_text = f"Persona: {persona}. Task: {job}"
query_embedding = embed_text(query_text)

# === LOAD DOCUMENTS ===
documents = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdf")]

# === PROCESS SECTIONS ===
results = []
boost_keywords = ['cities', 'activities', 'friends', 'group', 'beach', 'nightlife', 'cuisine', 'fun', 'travel', 'explore']

for doc_path in documents:
    sections = extract_sections(doc_path)
    print(f"📘 Found {len(sections)} candidate sections in {os.path.basename(doc_path)}")

    for sec in sections:
        sec_embedding = embed_text(sec['section_text'])
        similarity = compute_similarity(query_embedding, sec_embedding)

        # Keyword-based relevance boost
        boost = sum(kw in sec['section_text'].lower() for kw in boost_keywords)
        similarity += 0.01 * boost

        sec['similarity'] = similarity
        results.append(sec)

print(f"🔍 Total relevant sections found: {len(results)}")

# === RANK AND FILTER UNIQUE DOCS ===
ranked = sorted(results, key=lambda x: x['similarity'], reverse=True)

seen_docs = set()
unique_ranked = []
for sec in ranked:
    doc_name = os.path.basename(sec["document"])
    if doc_name not in seen_docs:
        seen_docs.add(doc_name)
        unique_ranked.append(sec)
    if len(unique_ranked) == top_k:
        break

# === FORMAT OUTPUT ===
output = {
    "metadata": {
        "input_documents": [os.path.basename(f) for f in documents],
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.utcnow().isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

for i, sec in enumerate(unique_ranked):
    output["extracted_sections"].append({
        "document": os.path.basename(sec["document"]),
        "section_title": sec["section_title"],
        "importance_rank": i + 1,
        "page_number": sec["page_number"]
    })

    output["subsection_analysis"].append({
        "document": os.path.basename(sec["document"]),
        "refined_text": sec["section_text"].replace('\n', ' ').strip(),
        "page_number": sec["page_number"]
    })

# === SAVE JSON OUTPUT ===
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Analysis complete. Output saved to {output_file}")
