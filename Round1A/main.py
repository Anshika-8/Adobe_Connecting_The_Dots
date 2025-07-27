import fitz  # PyMuPDF
import json
import os


def extract_text_with_features(pdf_path):
    doc = fitz.open(pdf_path)
    outline = []
    title = "Untitled"

    # Extract title from the first page
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    candidate_title = span["text"].strip()
                    if candidate_title:
                        title = candidate_title
                        break
                if title != "Untitled":
                    break
        if title != "Untitled":
            break

    # Process all pages
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue

            block_x0 = block.get("bbox", [0])[0]
            alignment = "left"
            if block_x0 > 100:
                alignment = "center"
            if block_x0 > 200:
                alignment = "right"

            prev_line_y = None
            for line in block["lines"]:
                line_y = line["bbox"][1]
                spacing = None
                if prev_line_y is not None:
                    spacing = round(line_y - prev_line_y, 2)
                prev_line_y = line_y

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    font_size = span["size"]
                    font_name = span["font"].lower()
                    is_bold = "bold" in font_name
                    is_italic = "italic" in font_name

                    # Heuristic: Level classification by size
                    if font_size >= 24:
                        level = "Title"
                    elif font_size >= 20:
                        level = "H1"
                    elif font_size >= 17:
                        level = "H2"
                    elif font_size >= 14:
                        level = "H3"
                    elif font_size >= 11:
                        level = "H4"
                    else:
                        level = "Body"

                    outline.append({
                        "level": level,
                        "text": text,
                        "page": page_num,
                        "font_size": font_size,
                        "font": span["font"],
                        "bold": is_bold,
                        "italic": is_italic,
                        "alignment": alignment,
                        "line_spacing": spacing
                    })

    return {"title": title, "outline": outline}


def process_pdfs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")
            data = extract_text_with_features(pdf_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    input_dir = "tests/inputs"     # Update as needed
    output_dir = "tests/outputs"   # Update as needed
    process_pdfs(input_dir, output_dir)
