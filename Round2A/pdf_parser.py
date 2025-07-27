import fitz  # PyMuPDF
import os

def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []

    print(f"📄 Processing: {os.path.basename(pdf_path)}")

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                text = " ".join(span["text"] for span in spans).strip()
                font_size = spans[0]["size"]

                # Heuristic: heading detection
                if (
                    10 < font_size < 25 and
                    10 < len(text) < 120 and
                    text[0].isupper() and
                    len(text.split()) > 2 and
                    not text.endswith(".")
                ):
                    page_text = page.get_text("text")
                    after_title_text = page_text.split(text, 1)[-1].strip()
                    paragraphs = after_title_text.split("\n\n")
                    content = ' '.join(p.strip() for p in paragraphs[:3])
                    content = ' '.join(content.split())

                    if len(content) < 80:
                        continue

                    sections.append({
                        "document": pdf_path,
                        "section_title": text,
                        "section_text": content,
                        "page_number": page_num + 1
                    })

    print(f"✅ Extracted {len(sections)} sections from {os.path.basename(pdf_path)}")
    return sections
