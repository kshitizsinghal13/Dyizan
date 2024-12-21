# Sentiment Analysis on Movie Reviews

## Overview
This project implements a sentiment analysis model that classifies movie reviews as either positive or negative. Utilizing the IMDb dataset, the model preprocesses the text data, transforms it into numerical vectors using TF-IDF, and trains a Naive Bayes classifier to predict sentiments.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Files Included](#files-included)
- [Model Evaluation](#model-evaluation)
- [Future Improvements](#future-improvements)
- [License](#license)

## Objective
The primary goal of this project is to build a sentiment analysis model that can accurately classify movie reviews based on their sentiment. The tasks involved include:
- Preprocessing text data (tokenization, lemmatization, and vectorization).
- Training a sentiment classification model using a labeled dataset.
- Evaluating the model's performance on unseen reviews.

## Dataset
The project uses the **IMDb Dataset** containing movie reviews labeled with sentiments (positive or negative). The dataset should be placed in the same directory as the project files.

## Installation
To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas numpy nltk scikit-learn joblib
```

Additionally, download the necessary NLTK resources by running:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Usage

### Training the Model
1. Ensure that your dataset (`IMDB Dataset.csv`) is in the same directory as `train.py`.
2. Run the training script:

```bash
python train.py
```

This script will preprocess the data, train a Naive Bayes model, and save both the model and vectorizer for later use.

### Testing the Model
1. Run the testing script:

```bash
python test.py
```

2. Enter a movie review when prompted. The script will output whether the review is classified as "Positive" or "Negative". Type 'exit' to quit.

## Files Included

- `train.py`: Script for training the sentiment analysis model.
- `test.py`: Script for testing predictions with user input.
- `IMDB Dataset.csv`: The dataset containing movie reviews (ensure this file is present in your project directory).

## Model Evaluation
After training, the model's accuracy and performance metrics are printed to the console. The evaluation includes:
- Accuracy score: Percentage of correctly classified reviews.
- Classification report: Detailed metrics including precision, recall, and F1-score for each class.
