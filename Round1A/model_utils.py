import numpy as np
import fitz  # PyMuPDF
from transformers import LayoutLMv3FeatureExtractor
import onnxruntime as ort

class PDFOutlineExtractor:
    def _init_(self, onnx_model_path):
        # Line 4: Load ONNX model (quantized)
        self.ort_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.feature_extractor = LayoutLMv3FeatureExtractor()
    
    def _call_(self, pdf_path):
        # Line 5: Parse PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        headings = []
        
        for page in doc:
            # Line 6: Get text blocks with coordinates
            blocks = page.get_text("dict")["blocks"]
            
            # Line 7: Batch process for speed
            texts = [b["text"] for b in blocks]
            boxes = [b["bbox"] for b in blocks]
            
            # Line 8: ML inference
            inputs = self.feature_extractor(texts, boxes=boxes, return_tensors="np")
            outputs = self.ort_session.run(None, {
                "input_ids": inputs["input_ids"],
                "bbox": inputs["bbox"],
                "attention_mask": inputs["attention_mask"]
            })
            
            # Line 9: Filter headings (class 1)
            for i, pred in enumerate(np.argmax(outputs[0], axis=1)):
                if pred == 1:  # Heading class
                    headings.append({
                        "text": texts[i],
                        "page": page.number,
                        "bbox": boxes[i]
                    })
        
        # Line 10: Build hierarchy
        return self._build_hierarchy(headings)
    
    def _build_hierarchy(self, headings):
        """Convert flat list of headings into nested document structure"""
        # Sort headings by page -> Y position -> X position
        headings.sort(key=lambda x: (x["page"], x["bbox"][1], x["bbox"][0]))
        
        # Initialize root node and stack
        root = {
            "title": "Document Root",
            "level": 0,
            "page": 0,
            "children": []
        }
        stack = [root]
        
        for heading in headings:
            # Estimate heading level from font size (bbox height)
            level = self._estimate_heading_level(heading["bbox"])
            
            # Create new node
            new_node = {
                "title": heading["text"],
                "level": level,
                "page": heading["page"],
                "children": []
            }
            
            # Find appropriate parent (nearest ancestor with lower level)
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            
            # Add to parent's children
            stack[-1]["children"].append(new_node)
            stack.append(new_node)
        
        return root
    
    def _estimate_heading_level(self, bbox):
        """Convert bbox dimensions to heading level (1-6)"""
        height = bbox[3] - bbox[1]  # y2 - y1
        width = bbox[2] - bbox[0]   # x2 - x1
        
        # Heuristic rules (adjust based on your document corpus)
        if height > 30: return 1    # H1 (main title)
        if height > 25: return 2    # H2 (section)
        if height > 20: return 3    # H3 (subsection)
        if width < 200: return 4    # Narrow = likely sidebar heading
        return 5                    # Default
    
    def to_markdown(self, outline_node=None, level=0):
        """Convert hierarchy to markdown format"""
        if outline_node is None:
            outline_node = self._current_outline
        
        md = []
        if level > 0:  # Skip root node
            prefix = "#" * min(level, 6)  # Max H6
            md.append(f"{prefix} {outline_node['title']} (p{outline_node['page']+1})")
        
        for child in outline_node["children"]:
            md.extend(self.to_markdown(child, level+1))
        
        return "\n".join(md)
