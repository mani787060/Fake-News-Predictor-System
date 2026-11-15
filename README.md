# Fake News Predictor System | NLP & Machine Learning

A Natural Language Processing (NLP) and Machine Learning project that explores how news content can be classified as **Fake or Real** using text preprocessing, TF-IDF feature extraction, and Logistic Regression.

The project demonstrates an end-to-end text classification workflow, from dataset exploration and linguistic preprocessing to model training, evaluation, and prediction.

## Project Overview

The spread of online misinformation makes it increasingly important to understand how machine learning can identify patterns in news content.

This project uses supervised machine learning to classify news records based on textual information. It applies NLP techniques to clean and normalize text, converts the processed content into numerical features using TF-IDF, and trains a Logistic Regression model to perform binary classification.

**Project type:** Supervised Machine Learning | Natural Language Processing | Binary Text Classification

## Objectives

- Explore and understand a labelled fake news dataset.
- Perform text cleaning and preprocessing using NLTK.
- Apply stemming and English stopword removal.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate model performance using training and testing accuracy.
- Demonstrate how a trained classifier can generate predictions.

## Dataset

The notebook uses a dataset named `fake_news.csv`.

The dataset contains **20,800 records and 5 columns**.

| Column | Description |
|---|---|
| `id` | Unique identifier for a news record |
| `title` | News article headline |
| `author` | Author or publisher information |
| `text` | News article content |
| `label` | Target class indicating Fake or Real news |

### Target Labels

According to the dataset description:

- `1` — Fake News
- `0` — Real News

### Missing Value Handling

The notebook checks missing values and replaces them with empty strings using Pandas.

The observed missing-value counts are:

| Column | Missing Values |
|---|---:|
| `id` | 0 |
| `title` | 558 |
| `author` | 1,957 |
| `text` | 39 |
| `label` | 0 |

## Machine Learning Workflow

### 1. Data Loading and Exploration

The dataset is loaded into a Pandas DataFrame.

Initial exploration includes:

- Inspecting the first few records.
- Checking dataset dimensions.
- Identifying missing values.
- Understanding the available features and target labels.

### 2. Text Preparation

The notebook combines the `author` and `title` columns into a new text feature named `content`.

This combined feature is used as the input for the text classification workflow.

**Important:** The current notebook does not use the full article `text` column as part of this combined feature.

### 3. Text Preprocessing with NLTK

Text preprocessing reduces unnecessary variation and prepares the content for feature extraction.

The notebook implements:

- **Regex Cleaning:** Removes characters other than English letters.
- **Lowercasing:** Converts text into lowercase.
- **Tokenization:** Splits text into individual words.
- **Stopword Removal:** Removes common English words using NLTK.
- **Stemming:** Uses the Porter Stemmer to reduce words to their root-like forms.

Example:

`"The players are playing football"`

After basic preprocessing and stemming, common words are removed and the remaining words are normalized.

### 4. Feature Extraction using TF-IDF

Machine learning algorithms require numerical input rather than raw text.

The project uses `TfidfVectorizer` from Scikit-learn.

**TF-IDF (Term Frequency-Inverse Document Frequency)** represents text using numerical values based on how important words are within a document relative to the collection of documents.

It helps assign greater importance to informative terms and lower importance to terms that appear frequently across many documents.

The resulting representation is a sparse numerical matrix suitable for text classification.

### 5. Train-Test Split

The dataset is divided into training and testing subsets using Scikit-learn's `train_test_split`.

Current notebook configuration:

- Training data: 80%
- Testing data: 20%
- Random state: 2
- Stratification: Enabled

This creates separate subsets for model training and evaluation.

### 6. Model Training — Logistic Regression

The project uses **Logistic Regression** as its classification algorithm.

Logistic Regression learns patterns from the TF-IDF feature representation and predicts one of the two target classes.

It is a useful baseline for text classification because it works effectively with high-dimensional sparse feature representations.

### 7. Model Evaluation

The notebook evaluates the trained model using accuracy on both training and testing data.

**Reported notebook results:**

| Evaluation Metric | Accuracy |
|---|---:|
| Training Accuracy | 98.64% |
| Testing Accuracy | 97.91% |

These are the results recorded in the current notebook. They should be interpreted cautiously because TF-IDF is fitted before the train-test split in the existing implementation, which can introduce information leakage.

A more reliable evaluation should fit the vectorizer only on training data, then transform the test data.

### 8. Prediction Demonstration

The notebook demonstrates prediction using a sample from the test dataset.

The trained Logistic Regression model generates a class prediction for the selected TF-IDF feature vector.

This demonstrates the basic prediction workflow, rather than a deployed application or an independent user-input interface.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| NLTK | Stopword removal and stemming |
| Regular Expressions | Text cleaning |
| Scikit-learn | TF-IDF, data splitting, Logistic Regression, and evaluation |
| Jupyter Notebook | Interactive development and experimentation |

## Repository Structure

```text
Fake-News-Predictor-System/
│
├── Fake-News-Predictor-System.ipynb
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/Fake-News-Predictor-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Fake-News-Predictor-System
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

The repository currently does not include a `requirements.txt` file. Install the required packages using:

```bash
pip install pandas numpy nltk scikit-learn jupyter
```

### 5. Prepare the Dataset

Place the dataset file named `fake_news.csv` in the project directory.

The expected location is:

```text
Fake-News-Predictor-System/
├── Fake-News-Predictor-System.ipynb
├── fake_news.csv
└── README.md
```

The dataset is referenced by the notebook but is not currently included in the GitHub repository.

### 6. Download NLTK Stopwords

Run the following Python commands:

```python
import nltk
nltk.download("stopwords")
```

### 7. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`Fake-News-Predictor-System.ipynb`

Execute the notebook cells sequentially to explore the dataset, preprocess the text, train the model, and evaluate its predictions.

## Current Implementation Notes

The following points are important when interpreting the current notebook:

- Logistic Regression is the classifier implemented in the notebook. Passive Aggressive Classifier and SVM are not currently implemented.
- The combined text feature uses author and title, rather than the full article body.
- The notebook currently fits TF-IDF before splitting the dataset. For a leakage-safe evaluation, split the raw text first and fit the vectorizer only on training data.
- The dataset label mapping is `1 = Fake` and `0 = Real`. The final prediction display logic should be corrected to match this mapping.
- The current notebook reports accuracy but does not implement a confusion matrix or F1-score evaluation.

## Future Improvements

Potential extensions to make the project more robust include:

- Correcting the TF-IDF fitting order to prevent data leakage.
- Using the full article content alongside headlines.
- Correcting the prediction label mapping.
- Adding precision, recall, F1-score, and confusion matrix evaluation.
- Comparing Logistic Regression with Linear SVM and Passive Aggressive Classifier.
- Saving the trained model and TF-IDF vectorizer for reuse.
- Building a Streamlit interface for interactive predictions.
- Testing performance on independent and more recent news datasets.

## Limitations

This project learns patterns from a labelled dataset. It does not independently verify news claims, investigate sources, or establish whether an article is factually correct.

Its predictions may be affected by dataset quality, historical patterns, language, topic distribution, and misleading correlations in the training data.

The model should therefore be treated as an educational text-classification experiment, not a definitive fact-checking system.

## Author

**Mani Ratan**

- GitHub: [@mani787060](https://github.com/mani787060)
- LinkedIn: [Mani Ratan](https://www.linkedin.com/in/mani-ratan-159963289/)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
