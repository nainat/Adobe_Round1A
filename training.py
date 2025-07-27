import os
import json
import glob
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import unicodedata
import re
import numpy as np
from collections import Counter

INPUT_PDF_DIR = "train_data/input_pdfs"
OUTPUT_JSON_DIR = "train_data/output_jsons"

MODEL_PATH = "title_h1_h2_h3_model.joblib"
FONT_ENCODER_PATH = "font_encoder.joblib"

HEADING_LEVELS = ["title", "H1", "H2", "H3", "H4", "H5", "H6"]

os.makedirs(INPUT_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

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
                            "spans": l["spans"],  # Save spans for bbox info
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
            indent_level = int(x0 / (width+1) * 10)  # 0-10 scale
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
                "x0": x0,
                "y0": y0,
                "center_x": center_x,
                "page_width": line["page_width"],
                "page_height": line["page_height"],
                "is_centered": is_centered,
                "is_all_caps": is_all_caps,
                "is_bold": is_bold,
                "is_bullet": is_bullet,
                "indent_level": indent_level,
                "font_rank": font_rank,
                "size_rank": size_rank,
                "is_largest_font": is_largest_font,
                "is_top3_font": is_top3_font,
                "most_common_font": most_common_font,
                "most_common_size": most_common_size,
            })
    doc.close()
    return lines

def get_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "")
    outlines = data.get("outline", [])
    labels = {}
    for item in outlines:
        labels[(item["page"], item["text"])] = item["level"]
    return title, labels

def build_training_data():
    pdf_files = glob.glob(os.path.join(INPUT_PDF_DIR, "*.pdf"))
    rows = []
    for pdf_file in pdf_files:
        base = os.path.splitext(os.path.basename(pdf_file))[0]
        json_file = os.path.join(OUTPUT_JSON_DIR, base + ".json")
        if not os.path.exists(json_file):
            continue
        title, outline_labels_raw = get_labels(json_file)
        outline_labels = {}
        for (lbl_page, lbl_text_raw), lbl_level in outline_labels_raw.items():
            norm_lbl_text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', lbl_text_raw)).strip().lower()
            outline_labels[(lbl_page, norm_lbl_text)] = lbl_level

        pdf_lines = extract_pdf_lines(pdf_file)
        filename = os.path.basename(pdf_file)

        norm_title = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', title)).strip().lower()

        lines_by_page = {}
        for line_info in pdf_lines:
            lines_by_page.setdefault(line_info["page"], []).append(line_info)
        page1_lines = lines_by_page.get(1, [])
        if page1_lines:
            max_font_size = max(l["size"] for l in page1_lines)
            largest_font_lines = [l for l in page1_lines if l["size"] == max_font_size]
            best_title_line = None
            for l in largest_font_lines:
                if l["is_centered"] and l["y0"] < l["page_height"]*0.25:
                    best_title_line = l
                    break
            if not best_title_line and largest_font_lines:
                best_title_line = largest_font_lines[0]
            if best_title_line:
                pdf_title_candidate = best_title_line["text"]
                norm_pdf_title_candidate = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', pdf_title_candidate)).strip().lower()
            else:
                norm_pdf_title_candidate = ""
        else:
            norm_pdf_title_candidate = ""

        prev_line_info = None
        for line_info in pdf_lines:
            page = line_info["page"]
            line = line_info["text"]
            font = line_info["font"]
            size = line_info["size"]
            flags = line_info["flags"]
            is_bullet = line_info.get("is_bullet", False)

            norm_line = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', line)).strip().lower()
            label = None
            matched = False

            if (page, norm_line) in outline_labels:
                label = outline_labels[(page, norm_line)]
                matched = True

            if not matched:
                if norm_line == norm_title:
                    label = "title"
                    matched = True
                elif page == 1 and norm_line == norm_pdf_title_candidate and norm_pdf_title_candidate:
                    label = "title"
                    matched = True
                elif line_info["is_largest_font"] and line_info["is_centered"] and line_info["y0"] < line_info["page_height"]*0.25:
                    label = "title"
                    matched = True
                else:
                    label = "other"

            prev_size = 0.0
            prev_flags = 0
            prev_is_bullet = False
            prev_label_is_heading = False
            prev_indent_level = 0
            prev_is_bold = False
            prev_is_all_caps = False

            if prev_line_info and prev_line_info["page"] == page:
                prev_size = prev_line_info["size"]
                prev_flags = prev_line_info["flags"]
                prev_is_bullet = prev_line_info["is_bullet"]
                prev_label_is_heading = prev_line_info["label"] in HEADING_LEVELS
                prev_indent_level = prev_line_info["indent_level"]
                prev_is_bold = prev_line_info["is_bold"]
                prev_is_all_caps = prev_line_info["is_all_caps"]

            rows.append({
                "filename": filename,
                "text": line,
                "font": font,
                "size": size,
                "flags": flags,
                "is_bullet": is_bullet,
                "is_centered": line_info["is_centered"],
                "is_all_caps": line_info["is_all_caps"],
                "is_bold": line_info["is_bold"],
                "indent_level": line_info["indent_level"],
                "font_rank": line_info["font_rank"],
                "size_rank": line_info["size_rank"],
                "is_largest_font": line_info["is_largest_font"],
                "is_top3_font": line_info["is_top3_font"],
                "label": label,
                "prev_size": prev_size,
                "prev_flags": prev_flags,
                "prev_is_bullet": prev_is_bullet,
                "prev_label_is_heading": prev_label_is_heading,
                "prev_indent_level": prev_indent_level,
                "prev_is_bold": prev_is_bold,
                "prev_is_all_caps": prev_is_all_caps,
            })

            prev_line_info = {
                "page": page,
                "size": size,
                "flags": flags,
                "is_bullet": is_bullet,
                "label": label,
                "indent_level": line_info["indent_level"],
                "is_bold": line_info["is_bold"],
                "is_all_caps": line_info["is_all_caps"],
            }

            try:
                print(
                    f"Processed {pdf_file}: '{line}' -> Label: {label} | Font: {font}, Size: {size}, "
                    f"Flags: {flags}, Bullet: {is_bullet} | Centered: {line_info['is_centered']}, "
                    f"AllCaps: {line_info['is_all_caps']}, Bold: {line_info['is_bold']} | "
                    f"Indent: {line_info['indent_level']} | Prev Size: {prev_size}, "
                    f"Prev Flags: {prev_flags}, Prev Bullet: {prev_is_bullet}, "
                    f"Prev Heading: {prev_label_is_heading}"
                )
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
    return pd.DataFrame(rows)

def train_model(df):
    df = df[df["label"].isin(HEADING_LEVELS)].copy()
    X = df[
        [
            "text", "font", "size", "flags", "is_bullet", "is_centered", "is_all_caps", "is_bold",
            "indent_level", "font_rank", "size_rank", "is_largest_font", "is_top3_font",
            "prev_size", "prev_flags", "prev_is_bullet", "prev_label_is_heading", "prev_indent_level",
            "prev_is_bold", "prev_is_all_caps"
        ]
    ]
    y = df["label"]

    font_encoder = LabelEncoder()
    X.loc[:, "font_encoded"] = font_encoder.fit_transform(X["font"])
    joblib.dump(font_encoder, FONT_ENCODER_PATH)

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(), "text"),
        ("font", "passthrough", ["font_encoded"]),
        ("size", StandardScaler(), ["size"]),
        ("flags", "passthrough", ["flags"]),
        ("is_bullet", "passthrough", ["is_bullet"]),
        ("is_centered", "passthrough", ["is_centered"]),
        ("is_all_caps", "passthrough", ["is_all_caps"]),
        ("is_bold", "passthrough", ["is_bold"]),
        ("indent_level", StandardScaler(), ["indent_level"]),
        ("font_rank", StandardScaler(), ["font_rank"]),
        ("size_rank", StandardScaler(), ["size_rank"]),
        ("is_largest_font", "passthrough", ["is_largest_font"]),
        ("is_top3_font", "passthrough", ["is_top3_font"]),
        ("prev_size", StandardScaler(), ["prev_size"]),
        ("prev_flags", "passthrough", ["prev_flags"]),
        ("prev_is_bullet", "passthrough", ["prev_is_bullet"]),
        ("prev_label_is_heading", "passthrough", ["prev_label_is_heading"]),
        ("prev_indent_level", StandardScaler(), ["prev_indent_level"]),
        ("prev_is_bold", "passthrough", ["prev_is_bold"]),
        ("prev_is_all_caps", "passthrough", ["prev_is_all_caps"]),
    ], remainder='drop')

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])
    pipeline.fit(X, y)
    return pipeline

if __name__ == "__main__":
    df = build_training_data()
    df.to_csv("training_data_lines.csv", index=False)
    model = train_model(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved as {MODEL_PATH}")