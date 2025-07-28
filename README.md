# Adobe_Connecting_The_Dots
#Round 1A
#  PDF Document Outline Extractor

##  Overview

This project extracts a **structured outline** from PDF documents by identifying the **Title** and **Headings (H1, H2, H3)**. It provides two modes of operation:

1. **Heuristic-Based Parsing** — Lightweight, fast, font-size-based extraction.
2. **Machine Learning-Based Classification** — Trained classifier for higher accuracy on diverse layouts.

Built for **Adobe All India Hackathon 2025**, this solution is designed to be **CPU-only, offline**, and fully compatible with `linux/amd64` Docker environments.

---

##  Problem Statement

> “You’re handed a PDF — instead of simply reading it, you’re tasked with making sense of it like a machine.”

### Your Mission:
- Accept a PDF file (up to 50 pages).
- Extract:
  - **Title**
  - **Headings**: H1, H2, H3 (with level and page number)
# Libraries used are as follows:
PyMuPDF (fitz)	          PDF text + layout extraction
scikit-learn	            Model training + evaluation
imblearn	                SMOTE oversampling for rare heading levels
joblib	                  Save/load trained model pipeline
transformers            	Layout-aware feature extraction (LayoutLMv3)
onnxruntime             	ONNX-based inference if using layout-aware models
#  How It Works
-Heuristic Mode
Extracts title by finding the largest text block on the first page.
Detects heading levels (Title, H1, H2, H3, H4, Body) by:
Font size thresholds
Bold/Italic style detection
Line spacing
Block alignment

- ML-Based Mode 
Training data: Labeled CSV of text blocks + layout metadata.
Model: RandomForestClassifier with TF-IDF + numerical features.
Improves accuracy across inconsistent layouts and fonts
