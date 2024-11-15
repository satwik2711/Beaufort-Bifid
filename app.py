# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import string
import random
import base64

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# For Transformer Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------------- Cipher Implementations -------------------------

# Beaufort Cipher Implementation
def beaufort_cipher_encrypt(plaintext, key):
    plaintext = ''.join(filter(str.isalpha, plaintext)).upper()
    key = key.upper()
    key_length = len(key)
    ciphertext = ''

    for i, char in enumerate(plaintext):
        p = ord(char) - ord('A')
        k = ord(key[i % key_length]) - ord('A')
        c = (k - p) % 26
        ciphertext += chr(c + ord('A'))

    return ciphertext

# Bifid Cipher Implementation
def bifid_cipher_encrypt(plaintext, key_square):
    plaintext = ''.join(filter(str.isalpha, plaintext)).upper()
    size = 5  # 5x5 Polybius Square
    indices = []

    # Create dictionaries for letter to index mapping
    char_to_index = {}
    index_to_char = {}
    for idx, char in enumerate(key_square):
        char_to_index[char] = (idx // size, idx % size)
        index_to_char[(idx // size, idx % size)] = char

    # Replace 'J' with 'I' if necessary
    plaintext = plaintext.replace('J', 'I')

    # Get row and column indices
    rows = []
    cols = []
    for char in plaintext:
        r, c = char_to_index[char]
        rows.append(r)
        cols.append(c)

    # Combine the indices
    combined = rows + cols

    # Split into pairs and map back to letters
    ciphertext = ''
    for i in range(0, len(combined), 2):
        r = combined[i]
        c = combined[i + 1]
        ciphertext += index_to_char[(r, c)]

    return ciphertext

# ------------------------- Data Generation -------------------------

def generate_dataset():
    # Generate plaintext samples
    samples = 200
    plaintexts = []
    for _ in range(samples):
        length = random.randint(100, 150)
        text = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + ' ', k=length))
        plaintexts.append(text)

    # Generate ciphertexts using Beaufort Cipher
    beaufort_ciphertexts = []
    key = 'CIPHERKEY'
    for text in plaintexts:
        ct = beaufort_cipher_encrypt(text, key)
        beaufort_ciphertexts.append(ct)

    # Generate ciphertexts using Bifid Cipher
    bifid_ciphertexts = []
    key_square = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'  # 'J' is merged with 'I'
    for text in plaintexts:
        ct = bifid_cipher_encrypt(text, key_square)
        bifid_ciphertexts.append(ct)

    # Create DataFrame
    data = pd.DataFrame({
        'Ciphertext': beaufort_ciphertexts + bifid_ciphertexts,
        'Algorithm': ['Beaufort'] * samples + ['Bifid'] * samples
    })

    return data

# ------------------------- Machine Learning Models -------------------------

# Prepare data for ML models
def prepare_data(data):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(data['Ciphertext'])
    sequences = tokenizer.texts_to_sequences(data['Ciphertext'])
    max_seq_length = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    y = data['Algorithm'].map({'Beaufort': 0, 'Bifid': 1}).values
    return X, y, tokenizer, max_seq_length

# Naive Bayes Classifier
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Neural Network Model
def train_neural_network(X_train, y_train, input_dim):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=X_train.shape[1]),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

# LSTM Model
def train_lstm_model(X_train, y_train, input_dim):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=X_train.shape[1]),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

# Transformer Model
def train_transformer_model(X_train, y_train, input_dim):
    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = Sequential(
                [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
            )
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.Input(shape=(X_train.shape[1],))
    embedding_layer = Embedding(input_dim=input_dim, output_dim=embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer)
    x = GlobalAveragePooling1D()(transformer_block)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

# ------------------------- Streamlit GUI -------------------------

def main():
    st.title("Cipher Type Prediction")

    # Check if data is already generated
    if 'data' not in st.session_state:
        st.session_state['data'] = generate_dataset()
        st.session_state['X'], st.session_state['y'], st.session_state['tokenizer'], st.session_state['max_seq_length'] = prepare_data(st.session_state['data'])
        X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'], test_size=0.2, random_state=42)

        # Train Models
        st.session_state['nb_model'] = train_naive_bayes(X_train, y_train)
        st.session_state['nn_model'] = train_neural_network(X_train, y_train, input_dim=len(st.session_state['tokenizer'].word_index) + 1)
        st.session_state['lstm_model'] = train_lstm_model(X_train, y_train, input_dim=len(st.session_state['tokenizer'].word_index) + 1)
        st.session_state['transformer_model'] = train_transformer_model(X_train, y_train, input_dim=len(st.session_state['tokenizer'].word_index) + 1)

        # Store test data
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

    # User Input
    ciphertext_input = st.text_area("Enter Ciphertext Here:", height=200)

    if st.button("Predict Cipher Type"):
        if ciphertext_input:
            # Prepare input
            tokenizer = st.session_state['tokenizer']
            sequence = tokenizer.texts_to_sequences([ciphertext_input])
            X_input = pad_sequences(sequence, maxlen=st.session_state['max_seq_length'], padding='post')

            # Predictions
            nb_pred = st.session_state['nb_model'].predict(X_input)
            nn_pred = st.session_state['nn_model'].predict(X_input)
            lstm_pred = st.session_state['lstm_model'].predict(X_input)
            transformer_pred = st.session_state['transformer_model'].predict(X_input)

            # Convert predictions to labels
            pred_labels = {
                'Naive Bayes': 'Beaufort' if nb_pred[0] == 0 else 'Bifid',
                'Neural Network': 'Beaufort' if nn_pred[0][0] < 0.5 else 'Bifid',
                'LSTM Model': 'Beaufort' if lstm_pred[0][0] < 0.5 else 'Bifid',
                'Transformer Model': 'Beaufort' if transformer_pred[0][0] < 0.5 else 'Bifid'
            }

            # Display Predictions
            st.write("### Prediction Results:")
            for model_name, label in pred_labels.items():
                st.write(f"- **{model_name}** predicts: {label}")

    # Show Model Performance
    if st.button("Show Model Performance on Test Data"):
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        # Naive Bayes
        nb_pred = st.session_state['nb_model'].predict(X_test)
        st.write("#### Naive Bayes Classifier Performance:")
        st.text(classification_report(y_test, nb_pred, target_names=['Beaufort', 'Bifid']))

        # Neural Network
        nn_pred = st.session_state['nn_model'].predict(X_test)
        nn_pred_labels = (nn_pred > 0.5).astype(int)
        st.write("#### Neural Network Performance:")
        st.text(classification_report(y_test, nn_pred_labels, target_names=['Beaufort', 'Bifid']))

        # LSTM Model
        lstm_pred = st.session_state['lstm_model'].predict(X_test)
        lstm_pred_labels = (lstm_pred > 0.5).astype(int)
        st.write("#### LSTM Model Performance:")
        st.text(classification_report(y_test, lstm_pred_labels, target_names=['Beaufort', 'Bifid']))

        # Transformer Model
        transformer_pred = st.session_state['transformer_model'].predict(X_test)
        transformer_pred_labels = (transformer_pred > 0.5).astype(int)
        st.write("#### Transformer Model Performance:")
        st.text(classification_report(y_test, transformer_pred_labels, target_names=['Beaufort', 'Bifid']))

if __name__ == "__main__":
    main()
