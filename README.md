## Adobe India Hackathon 2025 – Round 1A Submission

### Challenge: “Connecting the Dots”  Understand Your Document
Team Name: NAN
**Objective**
Extract a structured outline (Title, H1, H2, H3) from a given PDF file (≤ 50 pages) with high precision and speed. The output must be a clean, hierarchical JSON format usable for downstream document intelligence.

**Directory Structure
**
 ├──solution/
 │ ├── input
 │ └── output
 ├── title_h1_h2_h3_model.joblib
 ├── font_encoder.joblib
 ├── training.py
 ├── testing.py
 ├── Dockerfile
 ├── README.md
 
**Models and Libraries Used**
**PyMuPDF (fitz):** PDF layout parsing
**scikit-learn:** Feature engineering & RandomForestClassifier
**joblib:** Model serialization
**pandas, numpy:** Data manipulation
**unicodedata, regex: ** Text normalization (multilingual support)

**Features**
-Smart PDF structure parsing with contextual font & layout signals
-Hierarchical heading extraction (Title, H1, H2, H3)
-Trained on ground truth with contextual + sequential features
-Multilingual PDF compatibility (supports Japanese, Telugu,Hindi etc.)

**How It Works**

**Training Pipeline (`training.py`)**
1. Parses PDFs using `PyMuPDF` (fitz).
2. Dataset was made by using existing pdfs and creating .jsons file for 350+ pdfs
3. Extracts per-line features: font size, boldness, position, centeredness, etc.
4. Applies normalization and heuristics to match labeled headings (from ground truth JSONs).
5. Trains a `RandomForestClassifier` inside a `sklearn` pipeline using: TF-IDF features from text
6. Numerical + boolean layout features
7. Saves:
      Trained model: `title_h1_h2_h3_model.joblib`
      Font encoder: `font_encoder.joblib`

**Inference (`testing.py`)**
1. Loads trained model and font encoder.
2, Extracts lines and engineered features from PDFs in `input/`.
3 Predicts headings (title, H1, H2, H3).
4. Handles:
    Numbered/Bulleted formats
    Multilingual PDFs (UTF-8 support)


Generates structured output JSONs in `/output/`.

Input Format

- Input directory: `/app/input`
- Input files: One or more `.pdf` files (each ≤ 50 pages)

Output Format (per file)

- Output directory: `/app/output`
- Output: Corresponding `.json` per input `.pdf` in this format:

{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}

**🐳 Docker Setup**

**Dockerfile Highlights:**
- Based on python:3.9-slim-bullseye
- No internet access
- CPU-only (amd64)
- Model size < 200MB
- Offline execution
- 
Build code:
docker build --platform linux/amd64 -t mysolution:abc123 .
Run code:
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolution:abc123


**Constraints Satisfied**
Constraint
Status 
≤ 10s for 50-page PDF  -  Optimized
Model size ≤ 200MB - (~6MB)
CPU-only - Yes
No internet access - Enforced
AMD64 compatibility - Base image


**Submission Package**
Includes:
- Dockerfile
- README.md
- training.py
- testing.py
- title_h1_h2_h3_model.joblib
- font_encoder.joblib



