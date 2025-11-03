🧠 **Twitter-Sentiment-Analysis: A Benchmark of Supervised Machine Learning Approaches** Comparative Sentiment Analysis on Twitter Data Using Machine Learning Models

📌 **Project Overview**

- This project performs **sentiment analysis on Twitter data** to classify tweets as positive or negative using multiple supervised machine learning models.

- Two benchmark datasets — **Sentiment140** and **TweetEval** — were utilized to analyze how data quality and noise affects the model accuracies.

- The study compares preprocessing techniques, feature extraction methods, and model performances to identify the most effective approach for real-world Twitter sentiment classification.


🎯 **Objectives**

- Compare performance of different machine learning models for tweet sentiment classification.
Analyze impact of **data quality** (noisy vs. structured datasets).

- Apply and evaluate **feature engineering** techniques such as TF-IDF and Data-Level balancing techniques such as SMOTE.

- Explore **ensemble learning methods** and voting techniques to improve prediction accuracy.

- Visualize results through performance metrics and comparative plots.


## **📊 Dataset Overview**

| **Dataset** | **Source** | **Description** |
|--------------|------------|-----------------|
| **Sentiment140** | [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) | 1.6M noisy tweets with sentiment labels as 0-> negative, 2->neutral, 4->positive. |
| **TweetEval** | [HuggingFace](https://www.kaggle.com/datasets/cardiffnlp/tweeteval) | Structured dataset for binary sentiment analysis (positive/negative). |


Each dataset includes tweet text and corresponding sentiment labels for supervised training.

⚙️ **Methodology**
🧹 **Data Preprocessing**


- Tokenization, lemmatization, and stopword removal.


- Removal of URLs, lowecasing, hashtags, mentions, emojis, and non-ASCII characters.


- Normalization of repeated characters.


Train-test split and vectorization of text for model training.

💡 **Feature Engineering**

- Applied **TF-IDF Vectorization** for Sentiment140 dataset to convert tweets into neumeric data.

- Used **SMOTE(Synthetic Minority Oversampling Technique)**  on TweetEval for handling class imbalance.


🤖 **Model Development**

- Implemented and compared multiple supervised machine learning models:

- Logistic Regression

- Linear SVM

- Multinomial Naïve Bayes

- Random Forest

- XGBoost

- k-Nearest Neighbours(k-NN)

- Decision Tree(DT)

- Hard Voting Ensemble (combining top models)

🧬 **Optimization**

- Hyperparameter tuning using Grid Search and Randomized Search CV.

- applied to XGBoost and Logistic Regression.


📈 **Evaluation Metrics**

- Accuracy

- Precision

- Recall

- F1-Score

- ROC-AUC Score

- Confusion Matrix Visualization

***Results are compared between Sentiment140 and TweetEval datasets to show the impact of data cleanliness and structure on model performance.***

🔍 **Key Insights**

- **Structured datasets (TweetEval)**  outperform ** noisy ones (Sentiment140)**  due to cleaner labeling.

- **TF-IDF** yields strong baseline performance; Hard Voting Ensemble give more acurracy for both dataset.

- Feature imbalance correction with **SMOTE** improves recall on minority sentiment class.

- Comparative analysis reveals data noise as a critical factor influencing sentiment accuracy.

🧠 **Future Work**

- Incorporate **deep learning architectures (LSTM, BERT)**  for contextual sentiment understanding.

- Implement **real-time tweet streaming and live sentiment**  dashboard using Tweepy + Flask.

- Extend analysis to **multi-class sentiment (positive, negative, neutral)** .

- Deploy optimized models through a **REST API**  or lightweight web app.


📦 **Tech Stack**

**Languages:**  Python

**Libraries:**  NumPy, pandas, scikit-learn, XGBoost, matplotlib, seaborn

**Tools:**  Jupyter Notebook, Google Colab

**Data Sources:** Kaggle Datasets and HuggingFace (Sentiment140, TweetEval)
