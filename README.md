# README

## Project Title

**Automated Identification of Cipher Algorithms Using Naive Bayes, Neural Networks, LSTM, and Transformer Models**

## About the Project

This project implements two classical cipher algorithms—**Beaufort** and **Bifid**—to generate ciphertext datasets. It trains four machine learning classifiers:

- **Naive Bayes**
- **Neural Network**
- **LSTM Model**
- **Transformer Model**


## Setup Instructions

1. **Install Dependencies**

   Ensure you have Python 3 installed. Install the required libraries using the following command:

   ```bash
   pip install streamlit scikit-learn tensorflow
   pip install matplotlib seaborn
   ```


2. **Run the Streamlit App**

   Execute the following command in your terminal or command prompt:

   ```bash
   streamlit run app.py
   ```


3. **Use the Application**

   - The Streamlit app will open in your default web browser.
   - Enter a ciphertext into the text area provided.
   - Click **"Predict Cipher Type"** to see predictions from all models.
   - Click **"Show Model Performance on Test Data"** to view detailed performance metrics of each model.

## Note

- The first time you run the application, it will generate the dataset and train all models, which may take a few minutes.
- Ensure a stable internet connection during the installation of dependencies and the first run of the app.