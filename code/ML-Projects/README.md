# 🔍 Job Description to ESCO Occupation Code Classification

This project uses **supervised machine learning** and **semantic text embeddings** to classify job descriptions into standardized occupational codes using the [ESCO](https://ec.europa.eu/esco/portal/home) (European Skills, Competences, Qualifications and Occupations) framework.

---

## 🎯 Objective

Automatically predict the correct **ESCO occupation code** from a given job description using **Sentence-BERT embeddings** and a **Support Vector Machine (SVM)** classifier.

---

## 📁 Dataset Used

- **ESCO Occupation Dataset** (v1.2.0, English)
- Source: [EU Open Data Portal](https://ec.europa.eu/esco/portal)
- Key Fields Used: `preferredLabel`, `altLabels`, `definition`, `description`, `code`

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading and preprocessing |
| `scikit-learn` | Model training, encoding, evaluation |
| `sentence-transformers` | Semantic text embeddings |
| `streamlit` | Interactive web interface |
| `numpy`, `collections` | Helper functions and analysis |

---

## 🔄 Pipeline Overview

### 1. Data Preprocessing
- Clean and normalize text (e.g., remove punctuation, lowercase)
- Merge `preferredLabel`, `altLabels`, `definition`, and `description` into one rich job text

### 2. Filtering Top N Classes
- Retain the top `N=500` most frequent ESCO occupation codes for better model performance

### 3. Feature Engineering
- Use `Sentence-BERT` (`all-MiniLM-L6-v2`) to encode job texts into dense vector representations

### 4. Model Training
- Use `LinearSVC` with `class_weight='balanced'` to handle class imbalance effectively

### 5. Evaluation
- Generate a classification report using `sklearn`
- Focus on evaluating the most frequent or predicted classes

### 6. Prediction Function
- Input: Job description text
- Output: ESCO URI and human-readable job label
