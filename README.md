# Resume Screening with Machine Learning

An automated Natural Language Processing (NLP) system designed to streamline the HR recruitment process. This tool automatically digests, categorizes, and screens candidate resumes, mapping them to predefined job roles or departments with high computational accuracy.

## 🎯 Objective
To eliminate manual resume sorting by framing the problem as a multi-class text classification task. The model predicts the correct job category (e.g., Data Science, Web Development, HR, Sales) based purely on the textual content of the applicant's resume.

## 🛠️ Technologies & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Text Processing & NLP:** NLTK, Scikit-Learn
* **Machine Learning Classifiers:** Scikit-Learn (K-Nearest Neighbors, RandomForest, or Support Vector Machines)
* **Visualization:** Matplotlib, Seaborn

## 🧠 Methodology
1. **Resume Parsing & Text Cleaning:**
   * Creating custom regex functions to scrub resumes of noise: removing URLs, RTs, hashtags, mentions, special characters, and non-ASCII characters.
   * Transforming text to lowercase and standardizing whitespace.
2. **Categorical Encoding:**
   * Using `LabelEncoder` to convert categorical job roles (target variable) into numerical labels readable by machine learning models.
3. **Text Feature Extraction:**
   * Applying **TF-IDF Vectorization** to convert the cleaned text corpus into a sparse matrix of numerical features, highlighting the most distinguishing keywords for each job category.
4. **Model Training System:**
   * Implementing a multi-class predictive model (e.g., K-Nearest Neighbors coupled with an `OneVsRestClassifier` approach).
   * Splitting data into distinct training and testing cohorts to prevent data leakage.
5. **Evaluation:**
   * Generating highly detailed accuracy reports and classification metrics to ensure the model makes unbiased and accurate screening decisions.

## 🚀 How to Run
1. Install necessary dependencies: `pip install pandas numpy scikit-learn nltk matplotlib seaborn`
2. Open `Resume Screening with machine learning.ipynb` in your Jupyter environment.
3. Execute the workflow to observe the regex cleaning process, the TF-IDF transformation, and the final screening predictions.
