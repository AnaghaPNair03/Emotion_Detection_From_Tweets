# Emotion Detection from Tweets

A Natural Language Processing (NLP) and Deep Learning project that explores emotion classification in text using traditional machine learning algorithms and an LSTM neural network.

## Overview

Emotion Detection from Tweets aims to classify text into emotion categories by learning patterns in natural language. The project covers the end-to-end workflow of a text classification task, including dataset loading, text preprocessing, exploratory data analysis (EDA), feature extraction, model training, and performance evaluation.

Three classification approaches are explored:

- **Logistic Regression** — a linear machine learning classifier.
- **Multinomial Naive Bayes** — a probabilistic classifier commonly used for text classification.
- **Long Short-Term Memory (LSTM)** — a recurrent neural network designed to learn sequential patterns in text.

The project also includes an experimental majority-voting ensemble approach to combine predictions from multiple models.

## Objectives

- Understand and prepare a labelled text dataset for emotion classification.
- Clean and preprocess textual data using NLP techniques.
- Explore emotion distributions and text characteristics through visualizations.
- Train and evaluate traditional machine learning and deep learning models.
- Compare classification performance using accuracy and classification reports.
- Explore ensemble learning as a potential way to combine model predictions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| NLTK | Stopword removal and lemmatization |
| Scikit-learn | Feature extraction, data splitting, classification and evaluation |
| TensorFlow / Keras | LSTM model development and training |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| WordCloud | Visualization of frequently occurring words |

## Project Workflow

### 1. Dataset Loading

The project loads three dataset files:

- `train.txt`
- `val.txt`
- `test.txt`

Each file is read using a semicolon (`;`) delimiter, with two columns: `text` and `emotion`.

The three datasets are subsequently combined for exploratory analysis and model development.

**Expected data format:**

```text
I feel really happy today.;joy
I am worried about what happens next.;fear
```

The examples above illustrate the expected structure; they are not quoted dataset records.

### 2. Text Preprocessing

An NLP preprocessing function is implemented to prepare textual data for analysis. Its operations include:

1. Converting text to lowercase.
2. Removing numbers and non-alphabetic characters.
3. Removing punctuation.
4. Tokenizing text into words.
5. Removing English stopwords.
6. Applying WordNet lemmatization.

A Snowball stemmer is also initialized, although stemming is optional and disabled in the current preprocessing function.

### 3. Exploratory Data Analysis

The notebook examines the dataset's structure and text characteristics through:

- Missing-value checks.
- Duplicate-record identification and removal.
- Dataset information and structure inspection.
- Emotion distribution visualization using a bar chart.
- Text-length distribution analysis using a KDE plot.
- Separate word clouds for each emotion category.

These analyses help characterize the dataset before classification.

### 4. Feature Extraction

For the traditional machine learning models, the notebook uses **CountVectorizer** to convert text into numerical bag-of-words features.

For the LSTM model, the text is converted into integer sequences using the Keras tokenizer. The sequences are padded or truncated to a fixed length of 100 tokens.

### 5. Model Training

#### Logistic Regression

A Logistic Regression classifier is trained on the CountVectorizer features. The model uses `max_iter=1000` and `random_state=100`.

#### Multinomial Naive Bayes

A Multinomial Naive Bayes classifier is trained on the same bag-of-words feature representation, providing a probabilistic baseline for text classification.

#### LSTM Neural Network

The deep learning model uses the following architecture:

- **Embedding layer:** Learns a 128-dimensional representation for tokens.
- **LSTM layer:** Contains 128 units to learn sequential text patterns.
- **Dropout layer:** Uses a dropout rate of 0.5 to help reduce overfitting.
- **Dense output layer:** Uses softmax activation to predict probabilities across the emotion classes.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as its training metric.

Training runs for up to 30 epochs with a batch size of 32. Early stopping monitors validation loss with a patience of three epochs.

### 6. Model Evaluation

The notebook evaluates the trained classifiers using:

- **Accuracy:** The proportion of predictions that match the true labels.
- **Precision:** How many predictions for a class are correct.
- **Recall:** How many actual examples of a class are identified correctly.
- **F1-score:** The harmonic mean of precision and recall.
- **Classification report:** A per-class summary of precision, recall, F1-score, and support.

The notebook also experiments with combining predictions using majority voting. The ensemble implementation requires correction and validation before its output should be used for model comparison.

## Installation

Clone or download this repository and navigate to the project directory.

Install the required libraries:

```bash
pip install numpy pandas seaborn nltk matplotlib wordcloud scikit-learn tensorflow
```

NLTK resources are downloaded in the notebook:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Dataset Setup

Place the three dataset files in the locations expected by the notebook:

```text
/content/train.txt
/content/val.txt
/content/test.txt
```

These paths are configured for a Google Colab environment. If running the notebook locally or in another environment, update the file paths accordingly.

The dataset itself is not included with this notebook, so it must be obtained separately from its original source.

## Running the Project

1. Open `Emotion_Detection_from_tweets.ipynb` in Google Colab or a compatible Jupyter environment.
2. Install the required dependencies.
3. Upload or configure the paths to the three dataset files.
4. Execute the notebook cells in sequence.
5. Review the EDA visualizations and the classification reports generated by the models.
6. Inspect the ensemble section separately after correcting its prediction and voting logic.

## Repository Structure

```text
Emotion-Detection-from-Tweets/
│
├── Emotion_Detection_from_tweets.ipynb
├── README.md
│
└── dataset/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The `dataset/` directory illustrates a suggested local organization. The current notebook uses `/content/` paths instead, and the dataset files are not included in the uploaded notebook.

## Current Limitations and Future Improvements

- **Use the existing dataset splits correctly:** The notebook combines the training, validation, and test datasets before performing a new random train-test split. A more rigorous evaluation should preserve the original splits and keep the test set independent of training and model selection.
- **Apply preprocessing consistently:** The notebook creates a `cleaned_text` column, but the current CountVectorizer and LSTM workflows use the original `text` column. The impact of preprocessing should be evaluated explicitly.
- **Correct the ensemble implementation:** The LSTM already produces multiclass class predictions using `argmax`. The ensemble code incorrectly applies a binary threshold and performs majority voting across the wrong dimension. Predictions should be aligned by sample and class encoding before voting.
- **Compare model performance systematically:** Present accuracy, precision, recall, and F1-score in a single comparison table after validating the experiments.
- **Improve generalization:** Consider class-weighted training, hyperparameter tuning, and additional validation to address class imbalance or overfitting.
- **Support inference on new text:** Add a reusable prediction function that applies the same preprocessing and feature transformation used during training.

## Conclusion

This project provides a practical introduction to NLP-based emotion classification by combining text analysis, traditional machine learning, and deep learning. It demonstrates how textual data can be transformed into numerical representations and used to train classifiers, while offering opportunities to improve evaluation methodology, ensemble learning, and deployment readiness.

The notebook is an experimental implementation; reported performance should be added only after the evaluation pipeline has been verified.

## License

No license has been specified for this project. Add an appropriate license before distributing or reusing the code publicly.
