import os
import json
import glob
import fitz  
import pandas as pd
import joblib
import unicodedata
import re
from collections import Counter
import numpy as np

MODEL_PATH = "title_h1_h2_h3_model.joblib"
FONT_ENCODER_PATH = "font_encoder.joblib"

INPUT_PDF_DIR = os.path.join(os.getcwd(), "input")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.json")

HEADING_LEVELS = ["title", "H1", "H2", "H3", "H4", "H5", "H6"]

os.makedirs(INPUT_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_pdf_lines(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    bullet_chars = {'•', '-', '*', '·', '‣', '◦', '–', '—', '▪', '‒'}
    font_size_counter = Counter()
    font_counter = Counter()
    all_lines_per_page = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_width = page.rect.width
        page_height = page.rect.height
        page_lines = []
        spans = page.get_text("dict")["blocks"]
        for block in spans:
            if "lines" in block:
                for l in block["lines"]:
                    line_text = "".join(span["text"] for span in l["spans"])
                    clean_line = line_text
                    if clean_line.strip():
                        max_span = max(l["spans"], key=lambda s: s.get("size", 0), default={})
                        font = max_span.get("font", "")
                        size = max_span.get("size", 0)
                        font_size_counter[(font, size)] += 1
                        font_counter[font] += 1
                        page_lines.append({
                            "page": page_num + 1,
                            "text": clean_line.strip(),
                            "font": font,
                            "size": size,
                            "flags": max_span.get("flags", 0),
                            "spans": l["spans"],
                            "page_width": page_width,
                            "page_height": page_height,
                        })
        all_lines_per_page.append(page_lines)

    all_sizes = [line["size"] for page_lines in all_lines_per_page for line in page_lines]
    unique_sizes = sorted(set(all_sizes), reverse=True)
    size_rank_map = {size: rank for rank, size in enumerate(unique_sizes)}
    most_common_font = font_counter.most_common(1)[0][0] if font_counter else ""
    most_common_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else 0

    for page_lines in all_lines_per_page:
        for line in page_lines:
            spans = line["spans"]
            y0 = min(span["bbox"][1] for span in spans)
            x0 = min(span["bbox"][0] for span in spans)
            x1 = max(span["bbox"][2] for span in spans)
            center_x = (x0 + x1) / 2
            width = line["page_width"]
            is_centered = abs(center_x - width/2) < width * 0.15
            is_all_caps = line["text"].isupper()
            is_bold = 'Bold' in line["font"] or 'bold' in line["font"].lower()
            first_char = line["text"][0] if line["text"] else ""
            is_bullet = first_char in bullet_chars
            indent_level = int(x0 / (width+1) * 10)
            font_rank = font_counter[line["font"]]
            size_rank = size_rank_map.get(line["size"], 99)
            is_largest_font = line["size"] == unique_sizes[0]
            is_top3_font = size_rank < 3

            lines.append({
                "page": line["page"],
                "text": line["text"],
                "font": line["font"],
                "size": line["size"],
                "flags": line["flags"],
                "is_bullet": is_bullet,
                "is_centered": is_centered,
                "is_all_caps": is_all_caps,
                "is_bold": is_bold,
                "indent_level": indent_level,
                "font_rank": font_rank,
                "size_rank": size_rank,
                "is_largest_font": is_largest_font,
                "is_top3_font": is_top3_font,
            })
    doc.close()
    return lines

def build_features(pdf_lines, font_encoder):
    df = pd.DataFrame(pdf_lines)
    df["font_encoded"] = df["font"].apply(
        lambda x: font_encoder.transform([x])[0] if x in font_encoder.classes_ else -1
    )

    df['prev_size'] = 0.0
    df['prev_flags'] = 0
    df['prev_is_bullet'] = False
    df['prev_label_is_heading'] = False
    df['prev_indent_level'] = 0
    df['prev_is_bold'] = False
    df['prev_is_all_caps'] = False

    prevs = {
        "page": -1,
        "size": 0.0,
        "flags": 0,
        "is_bullet": False,
        "label": "other",
        "indent_level": 0,
        "is_bold": False,
        "is_all_caps": False,
    }
    for i, row in df.iterrows():
        if prevs["page"] == row["page"]:
            df.at[i, "prev_size"] = prevs["size"]
            df.at[i, "prev_flags"] = prevs["flags"]
            df.at[i, "prev_is_bullet"] = prevs["is_bullet"]
            df.at[i, "prev_label_is_heading"] = prevs["label"] in HEADING_LEVELS
            df.at[i, "prev_indent_level"] = prevs["indent_level"]
            df.at[i, "prev_is_bold"] = prevs["is_bold"]
            df.at[i, "prev_is_all_caps"] = prevs["is_all_caps"]
        prevs = {
            "page": row["page"],
            "size": row["size"],
            "flags": row["flags"],
            "is_bullet": row["is_bullet"],
            "label": "other",  
            "indent_level": row["indent_level"],
            "is_bold": row["is_bold"],
            "is_all_caps": row["is_all_caps"],
        }
    return df

def is_numbered_heading(text):
    return bool(re.match(r"^((\d+[\.\)])+|([ivxlcIVXLC]+\.)+)\s+", text))

def pdf_has_majority_bullet_or_numbering(df, min_pct=0.35):
    """
    Returns True if more than min_pct of the lines are bullets/numbered (counted only among headings).
    """
    bullet_count = 0
    numbered_count = 0
    heading_count = 0
    for _, row in df.iterrows():
        label = row.get("predicted_label")
        if label in HEADING_LEVELS[1:]:
            heading_count += 1
            txt = row["text"]
            first_char = txt[0] if txt else ""
            if row["is_bullet"]:
                bullet_count += 1
            elif is_numbered_heading(txt):
                numbered_count += 1
    if heading_count == 0:
        return False
    bullet_pct = (bullet_count + numbered_count) / heading_count
    return bullet_pct >= min_pct

def convert_to_native_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_type(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.str_)):
        return str(obj)
    else:
        return obj

def trim_outline_at_colon(json_data):
    
    if "outline" in json_data:
        for item in json_data["outline"]:
            text = item.get("text", "")
            if ":" in text:
                before, _ = text.split(":", 1)
                item["text"] = before.strip()
    return json_data

def predict_outline(pdf_path, model, font_encoder):
    print(f"\nPredicting outline for: {pdf_path}")
    pdf_lines = extract_pdf_lines(pdf_path)
    if not pdf_lines:
        print(f"No lines extracted from {pdf_path}. Skipping prediction.")
        return {"title": "", "outline": []}

    df = build_features(pdf_lines, font_encoder)
    features = [
        "text", "font_encoded", "size", "flags", "is_bullet", "is_centered", "is_all_caps", "is_bold",
        "indent_level", "font_rank", "size_rank", "is_largest_font", "is_top3_font",
        "prev_size", "prev_flags", "prev_is_bullet", "prev_label_is_heading", "prev_indent_level",
        "prev_is_bold", "prev_is_all_caps"
    ]
    predicted_labels = model.predict(df[features])
    df["predicted_label"] = predicted_labels

    use_bullet_mode = pdf_has_majority_bullet_or_numbering(df)

    outline = []
    title_text = ""
    bullet_chars = {'•', '-', '*', '·', '‣', '◦', '–', '—', '▪', '‒'}

    if use_bullet_mode:
        i = 0
        while i < len(df):
            row = df.iloc[i]
            line_text = row["text"]
            predicted_label = row["predicted_label"]
            page_num = row["page"]

            if predicted_label == "title" and not title_text:
                title_text = line_text
                i += 1
                continue

            first_char = line_text[0] if line_text else ""
            if (first_char in bullet_chars and len(line_text.strip()) == 1 and i+1 < len(df)):
                next_row = df.iloc[i+1]
                if next_row["page"] == page_num and not next_row["is_bullet"]:
                    heading_text = line_text + " " + next_row["text"]
                    next_predicted_label = next_row["predicted_label"]
                    use_label = next_predicted_label if next_predicted_label in HEADING_LEVELS[1:] else predicted_label
                    outline.append({
                        "level": str(use_label),
                        "text": str(heading_text.strip()),
                        "page": int(page_num),
                    })
                    i += 2
                    continue

            if first_char in bullet_chars and len(line_text.strip()) > 1:
                heading_text = line_text[1:].strip()
            elif is_numbered_heading(line_text):
                heading_text = re.sub(r"^((\d+[\.\)])+|([ivxlcIVXLC]+\.)+)\s*", "", line_text).strip()           
            else:
                heading_text = line_text

            if predicted_label in HEADING_LEVELS[1:]:
                outline.append({
                    "level": str(predicted_label),
                    "text": str(heading_text),
                    "page": int(page_num),
                })
            i += 1
    else:
        
        for idx, row in df.iterrows():
            line_text = row["text"]
            predicted_label = row["predicted_label"]
            page_num = row["page"]

            if predicted_label == "title" and not title_text:
                title_text = line_text
                continue

            if predicted_label in ["H1", "H2", "H3", "H4"]:
                word_count = len(line_text.split())
                if word_count > 5:
                    continue

            if predicted_label in HEADING_LEVELS[1:]:
                clean_text = line_text
                clean_text = re.sub(r'^\s*([\d]+(\.[\d]+)*[\.\)]|[a-zA-Z][\.\)])\s*', '', clean_text)
                if not clean_text.strip():
                    clean_text = line_text  
                outline.append({
                    "level": predicted_label,
                    "text": clean_text,
                    "page": page_num,
                })
                    

    output_data = {
        "title": title_text,
        "outline": outline
    }
    output_data = trim_outline_at_colon(output_data)
    return output_data

if __name__ == "__main__":
    print("--- Starting PDF Heading Prediction ---")
    try:
        model = joblib.load(MODEL_PATH)
        font_encoder = joblib.load(FONT_ENCODER_PATH)
        print(f"Successfully loaded model from {MODEL_PATH} and font encoder from {FONT_ENCODER_PATH}")
    except FileNotFoundError:
        print(f"Error: Model or font encoder not found. Please ensure '{MODEL_PATH}' and '{FONT_ENCODER_PATH}' exist.")
        print("Run the training script first to generate these files.")
        exit()
    except Exception as e:
        print(f"Error loading model or font encoder: {e}")
        exit()

    pdf_files = glob.glob(os.path.join(INPUT_PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_PDF_DIR}. Please add PDFs to test.")
        exit()

    for pdf_file in pdf_files:
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        print(f"Processing: {pdf_file}")
        result_json = predict_outline(pdf_file, model, font_encoder)
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        result_json_native = convert_to_native_type(result_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_json_native, f, indent=2, ensure_ascii=False)
        print(f"Generated predicted outline for '{base_name}.pdf' saved to: {out_path}")