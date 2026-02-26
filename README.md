# 📊 Multi-Dataset ML Lab

A structured machine learning experimentation project developed after graduation and during my military service, with the goal of maintaining technical sharpness, discipline, and continuous improvement.

This repository contains three independent mini-projects across different machine learning domains:

- 🖼️ **MNIST** — Computer Vision  
- 🧬 **Breast Cancer Wisconsin** — Medical Data Classification  
- 🎬 **IMDB Reviews** — Natural Language Processing  

Each project follows a consistent and reproducible ML pipeline:

- Data Exploration  
- Data Preprocessing  
- Baseline Modeling  
- Model Improvement  
- Evaluation & Analysis  
- Interpretation of Results  

---

# 🎯 Project Motivation

After completing my Integrated Master's degree in Electrical and Computer Engineering (AI & ML specialization), I used structured free time during military service to further strengthen my applied machine learning skills.

Rather than building random experiments, I deliberately selected datasets that:

- Represent different ML domains (Vision, Healthcare, NLP)
- Require different preprocessing strategies
- Demand different modeling approaches
- Enable cross-domain performance comparison
- Reflect real-world industrial applications

This project demonstrates structured thinking, reproducibility, and cross-domain adaptability in machine learning.

---

# 📁 Mini Projects Overview

---

## 🖼️ 1. MNIST Digit Classification

**Dataset Type:** Image Data  
**Task:** Multi-class classification (digits 0–9)  
**Domain:** Computer Vision  

### Objective
Develop models capable of recognizing handwritten digits.

### Pipeline
- Image normalization
- Baseline classifier (Logistic Regression)
- Convolutional Neural Network (CNN)
- Confusion matrix analysis
- Misclassification inspection

### Results
- Baseline model achieved strong performance
- CNN significantly improved classification accuracy
- Misclassifications mainly occurred between visually similar digits (e.g., 4 vs 9)

### Key Insights
- Deep learning clearly outperforms traditional models for image data
- Proper preprocessing is critical
- Error analysis provides deeper understanding than accuracy alone

---

## 🧬 2. Breast Cancer Wisconsin Classification

**Dataset Type:** Structured Numerical Data  
**Task:** Binary classification (Malignant vs Benign)  
**Domain:** Medical AI  

### Objective
Build accurate and interpretable models to assist tumor diagnosis.

### Pipeline
- Exploratory Data Analysis (EDA)
- Feature scaling
- Logistic Regression
- Support Vector Machine (RBF Kernel)
- Random Forest
- Sensitivity-focused evaluation
- Permutation Feature Importance

### Results
- SVM and Random Forest achieved very high accuracy
- Malignant recall (sensitivity) was prioritized due to clinical importance
- Feature importance highlighted the most critical tumor characteristics

### Key Insights
- Interpretability is crucial in healthcare applications
- Sensitivity can be more important than overall accuracy
- Ensemble and kernel methods provide robustness

---

## 🎬 3. IMDB Sentiment Analysis

**Dataset Type:** Text Data  
**Task:** Binary classification (Positive vs Negative)  
**Domain:** Natural Language Processing  

### Objective
Build a sentiment classifier capable of analyzing movie reviews.

### Pipeline
- Text cleaning & tokenization
- TF-IDF vectorization
- Logistic Regression
- Coefficient interpretation (top positive/negative words)
- Performance evaluation

### Results
- Logistic Regression achieved strong classification performance
- Most influential words clearly reflected sentiment polarity
- Model coefficients provided interpretability

### Key Insights
- Linear models remain highly competitive for text classification
- Feature representation (TF-IDF) is critical
- NLP requires fundamentally different preprocessing logic compared to structured or image data

---

# 📊 Cross-Project Comparison

| Dataset        | Data Type   | Best Model                | Main Challenge                     | Key Strength                        |
|---------------|------------|---------------------------|-------------------------------------|--------------------------------------|
| MNIST         | Images     | CNN                       | Visual pattern recognition          | Deep learning representation         |
| Breast Cancer | Structured | SVM / Random Forest       | Sensitivity & interpretability      | Robust classification                |
| IMDB          | Text       | Logistic Regression       | Feature engineering                 | Strong linear modeling               |

### Observations

- Deep learning dominates image-based tasks.
- Kernel and ensemble methods perform strongly on structured medical data.
- Linear models remain powerful for high-dimensional sparse text data.
- Model selection must align with data type and domain constraints.

---

# 🛠️ Hard Skills & Technologies Used

## 💻 Programming & Environment
- Python 3.x  
- Google Colab  
- Jupyter Notebook  
- Git & GitHub  

## 📊 Data Handling & Analysis
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## 🤖 Machine Learning Techniques
- Logistic Regression  
- Support Vector Machines (RBF Kernel)  
- Random Forest  
- Convolutional Neural Networks (CNN)  
- TF-IDF Vectorization  
- Permutation Feature Importance  

## 📈 Evaluation & Model Assessment
- Accuracy  
- Precision / Recall / F1-score  
- Confusion Matrix  
- Sensitivity (Malignant Recall)  
- Feature Importance Analysis  
- Misclassification Analysis  

---

# 🧠 Core Competencies Demonstrated

- End-to-end ML pipeline development  
- Cross-domain adaptability (Vision, Healthcare, NLP)  
- Model comparison & selection strategy  
- Performance interpretation beyond accuracy  
- Domain-aware evaluation (e.g., medical sensitivity)  
- Explainability & feature analysis  
- Clean and reproducible experimentation structure  

---

# 🚀 Conclusion

This project represents disciplined, structured self-development during a transitional professional period.

It demonstrates:

- Adaptability across machine learning domains  
- Strong evaluation awareness  
- Practical implementation skills beyond theory  
- Ability to compare models critically and contextually  

Rather than focusing on a single dataset, this repository highlights versatility — a key requirement for applied AI and machine learning roles.

---

# 🔮 Future Extensions

- Implement deep learning models (LSTM / Transformer) for IMDB  
- Apply SHAP explainability to the medical dataset  
- Extend the lab with additional real-world datasets  
- Deploy selected models as APIs  

---

# 👨‍💻 Author

**Evangelos Papaioannou**  
Electrical & Computer Engineer (AI & Machine Learning)

Recent graduate from Democritus University of Thrace (D.U.Th.), specializing in Artificial Intelligence and Machine Learning.  

This project was developed as part of continuous self-improvement and applied ML practice after graduation and during military service.

🔗 GitHub: https://github.com/PapaEvang01  
🔗 LinkedIn: https://www.linkedin.com/in/evangelos-papaioannou/  

---
