import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding

#Sample Data (Batch of 3 sequences, each with 5 words, vocab size 1000)

model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=5),  # Word embeddings
    Bidirectional(LSTM(32, return_sequences=True)),  # BiLSTM Layer
    Dense(10, activation="softmax")  # Output Layer
])

#checkung summary
model.summary()
