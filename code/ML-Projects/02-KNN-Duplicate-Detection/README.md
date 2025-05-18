#  Quora Duplicate Question Detection using KNN

This project demonstrates how to detect whether two questions from the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) are semantically similar (i.e., duplicates) using the **K-Nearest Neighbors (KNN)** algorithm and **text embeddings**.

---

##  Objective

To build a machine learning model that determines whether two questions are duplicates using a KNN classifier and visualize performance metrics.

---

##  Dataset

- Source: [Kaggle - Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data)
- Columns:
  - `question1`, `question2`: The question pairs
  - `is_duplicate`: Label (1 if duplicate, 0 otherwise)

---

## ⚙ Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Seaborn, Matplotlib
- Sentence Transformers (`all-MiniLM-L6-v2`)

---

##  Workflow

1. **Data Loading & Cleaning**
2. **Text Embedding (using Sentence Transformers)**
3. **KNN Model Training**
4. **Prediction & Evaluation**
5. **Performance Metrics Visualization**
6. **Custom Input Prediction**

---

##  Evaluation Metrics

| Metric      | Score |
|-------------|-------|
| Accuracy    | 0.82  |
| Precision   | 0.80  |
| Recall      | 0.74  |
| F1 Score    | 0.77  |

> All metrics are plotted for better interpretability.

---

##  Visualizations

- Confusion Matrix
- Bar Chart of Accuracy, Precision, Recall, and F1 Score

---

## 🤖 Try it Yourself

```python
predict_duplicate("How do I cook pasta?", "What are the steps to cook spaghetti?")

