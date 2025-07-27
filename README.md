# 📄 Adobe India Hackathon 2025 – Round 1A Submission

### 🔍 Challenge: “Connecting the Dots” – Understand Your Document  
**Team Name:** `NAN`

---

## 🧠 Objective

Extract a **structured outline** (Title, H1, H2, H3) from a given PDF (≤ 50 pages) with **high precision** and **speed**.  
The output is a **clean, hierarchical JSON format** ready for downstream document intelligence.

---

## 🧰 Models & Libraries Used

- **PyMuPDF (`fitz`)** – PDF layout parsing  
- **scikit-learn** – Feature engineering & `RandomForestClassifier`  
- **joblib** – Model serialization  
- **pandas**, **numpy** – Data manipulation  
- **unicodedata**, **regex** – Text normalization (for multilingual support)

---

## ✨ Features

- Smart PDF structure parsing using **font and layout signals**
- Accurate **hierarchical heading detection** (Title, H1, H2, H3)
- Trained with contextual + sequential layout features
- Supports **multilingual PDFs** (e.g., Japanese, Telugu, Hindi)

---

## 🔧 How It Works

### 📈 Training Pipeline (`training.py`)

1. Parses PDFs using `PyMuPDF`
2. Dataset created with 350+ PDFs and labeled `.json` files
3. Extracts per-line layout features:
   - Font size, boldness, position, centeredness, etc.
4. Normalizes and aligns with ground truth
5. Trains a `RandomForestClassifier` using:
   - TF-IDF features
   - Layout-based features
6. Saves trained artifacts:
   - `title_h1_h2_h3_model.joblib`
   - `font_encoder.joblib`

---

### 🧪 Inference Pipeline (`testing.py`)

1. Loads trained model + font encoder
2. Extracts lines & engineered features from PDFs in `/input`
3. Predicts heading levels: Title, H1, H2, H3
4. Handles:
   - Numbered & bulleted headings
   - Multilingual (UTF-8) PDFs
5. Outputs structured JSONs to `/output`

---

## 📂 Input & Output

### 🔹 Input

- Directory: `/app/input`
- Files: One or more `.pdf` files (each ≤ 50 pages)

### 🔸 Output

- Directory: `/app/output`
- Format:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

**🐳 Docker Setup**

**Dockerfile Highlights:**
- Based on python:3.9-slim-bullseye
- No internet access
- CPU-only (amd64)
- Model size < 200MB
- Offline execution
  
**Build code:**
```bash
docker build --platform linux/amd64 -t mysolution:abc123 .
```
**Run code:**
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolution:abc123
```

## ✅ Constraints Satisfied

| Constraint                   | Status       |
|-----------------------------|--------------|
| ≤ 10s for 50-page PDF       | Optimized  |
| Model size ≤ 200MB          | (~6MB)     |
| CPU-only                    | Yes        |
| No internet access          | Enforced   |
| AMD64 compatibility         | Base image |



**Submission Package**
Includes:
- Dockerfile
- README.md
- training.py
- testing.py
- title_h1_h2_h3_model.joblib
- font_encoder.joblib



