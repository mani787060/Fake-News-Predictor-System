# Fake News Detection System (NLP)

## 📰 Project Overview
In an era of digital misinformation, this project provides a computational approach to identifying "Fake News." By leveraging **Natural Language Processing (NLP)** and **Supervised Machine Learning**, the system analyzes the linguistic patterns, syntax, and vocabulary of news articles to distinguish between factual reporting and fabricated content.

---

## 🛠️ Technical Pipeline

### 1. Text Preprocessing (Linguistic Cleaning)
To reduce noise and dimensionality, the following steps were implemented using the `NLTK` library:
* **Tokenization:** Breaking down sentences into individual words.
* **Stopword Removal:** Filtering out common words (the, is, at) that don't carry significant meaning.
* **Stemming/Lemmatization:** Reducing words to their root form (e.g., "running" to "run") using the Porter Stemmer.
* **Regex Cleaning:** Removing URLs, special characters, and punctuation.

### 2. Feature Extraction (TF-IDF)
Instead of a simple word count (Bag-of-Words), I utilized **Term Frequency-Inverse Document Frequency (TF-IDF)**. 
* **Why TF-IDF?** It penalizes words that appear too frequently across all documents (like "news" or "said") and gives higher weight to unique, descriptive terms that help characterize the truthfulness of an article.

### 3. Model Architecture
I evaluated and implemented the following classifiers to find the optimal balance between bias and variance:
* **Passive Aggressive Classifier:** Ideal for large-scale text streams.
* **Logistic Regression:** Providing a probabilistic baseline for classification.
* **Support Vector Machines (SVM):** Handling high-dimensional sparse data effectively.

---

## 📈 Performance Metrics
The system is evaluated based on:
* **Accuracy:** Overall correctness of the model.
* **Confusion Matrix:** To track False Positives (Real news flagged as Fake).
* **F1-Score:** The harmonic mean of precision and recall, crucial for imbalanced datasets.

---

## 💻 Tech Stack
* **Language:** Python
* **NLP Tools:** NLTK, Regular Expressions (re)
* **Machine Learning:** Scikit-learn
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Seaborn, WordCloud

---

## 🚀 Installation & Usage
1. **Clone the repo:** 
   ```bash
   git clone https://github.com/your-username/fake-news-detector-nlp.git
   
2. **Install dependencies:** 
   ```bash
   pip install -r requirements.txt
   
3. **Run the Notebook:** 
   Plaintext
   Execute `fake_news_detector.ipynb` to train the model and test custom news headlines.   
