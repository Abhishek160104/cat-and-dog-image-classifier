# Spam Text Message Classification 💻📲

This project aims to classify text messages as either spam or legitimate (ham) using various machine learning models. The dataset used is a collection of text messages labeled as spam or ham. 📩

## Dataset 📂

The dataset used in this project is provided in the `spam.csv` file. It contains two columns:

- `v1`: The label indicating whether the message is spam or ham. 🔖
- `v2`: The actual text content of the message. 📝

## Approach 🧭

The project follows these steps:

1. **Data Loading**: The dataset is loaded from the `spam.csv` file using pandas. 📥
2. **Data Cleaning**: The text data is cleaned by converting to lowercase, removing special characters and extra spaces, and removing stop words using NLTK. 🧹
3. **Feature Extraction**: The cleaned text is converted into a numerical feature matrix using TF-IDF vectorization. 🔢
4. **Model Training and Evaluation**: The data is split into training and testing sets. Several machine learning models (Logistic Regression, Multinomial Naive Bayes, Support Vector Machines, and Random Forest) are trained on the training data and evaluated on the test data. The accuracy and classification report are printed for each model. 🔍🧠

## Requirements 📋

To run this project, you'll need the following Python packages:

- pandas
- numpy
- nltk
- sklearn
- re

You can install these packages using pip:
Additionally, you'll need to download the NLTK stopwords corpus:

```python
import nltk
nltk.download('stopwords')

python main.py
