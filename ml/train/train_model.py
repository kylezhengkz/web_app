import pickle
import time
import re
import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
import tracemalloc
import tensorflow as tf

tracemalloc.start()

t0 = time.time()
with open("../dataset/dataset.pkl", "rb") as f:
    df = pickle.load(f)
t1 = time.time()
print(f"{round(t1 - t0, 3)} seconds to load dataset")

def tokenize(text):
    text =  text.replace("'", "")
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    text = text.replace("_", " _ ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

t0 = time.time()
with open("../dictionary/embedding_dictionary.pkl", "rb") as f:
    embedding_dictionary = pickle.load(f)
t1 = time.time()
print(f"{round(t1 - t0, 3)} seconds to load embedding dictionary")

max_token_length = 250
max_embedding_length = max_token_length * 300
def load_embeddings(text):
    embeddings = np.empty(max_embedding_length, dtype=np.float32)
    text = tokenize(text)
    tokens = text.split()
    
    for i, token in enumerate(tokens):
        if (i >= max_token_length):
            break
        if (token in embedding_dictionary):
            embeddings[(i*300):((i+1)*300)] = embedding_dictionary[token]
        else:
            embeddings[(i*300):((i+1)*300)] = np.zeros((300,), dtype=np.float32)
    
    if (i < max_token_length - 1):
        room_left = ((max_token_length - 1) - i)*300
        embeddings[((i+1)*300):] = np.zeros((room_left,), dtype=np.float32)
    
    return embeddings

print(f"Bytes of memory used: {tracemalloc.get_traced_memory()[0]}")
    
X = np.empty((len(df) * 5, max_embedding_length), np.float32)
y = []
accumulator = []
chunk_size = 10000
print(f"Rows of data {len(df)}")
print(f"Total data to process {len(df) * 5}")
for i, row in df.iterrows():
    for j in range(1, 6):
        embeddings = load_embeddings(row[f"{j}-Star Reviews"])
        assert embeddings.dtype == np.float32
        assert len(embeddings) == max_embedding_length
        accumulator.append(embeddings)
        y.append(j)
        
    if (len(accumulator) % chunk_size == 0):
        offset_i = (i + 1) * 5
        assert offset_i % chunk_size == 0, offset_i
        start_index = offset_i - chunk_size
        end_index = offset_i
        print(f"Populating X from index {start_index} to {end_index - 1}")
        X[start_index:end_index] = accumulator
        accumulator.clear()
    
if (len(accumulator) > 0):
    print(f"{len(accumulator)} data left to process")
    offset_i = (i + 1) * 5
    start_index = offset_i - len(accumulator)
    end_index = offset_i
    print(f"Populating X from index {start_index} to {end_index - 1}")
    X[start_index:end_index] = accumulator
    accumulator.clear()
        
assert X.dtype == np.float32    
assert (len(X) == len(y))

del df
del embedding_dictionary
import gc
gc.collect()

print("Processing complete")
print(f"Bytes of memory used by X: {sys.getsizeof(X)}")
print(f"Bytes of memory used by y: {sys.getsizeof(y)}")

validation_split_index = int(0.8*len(X))
print(f"Validation split index: {validation_split_index}")

X_train = X[:validation_split_index]
X_val = X[validation_split_index:]

y_train = y[:validation_split_index]
y_val = y[validation_split_index:]

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_train) + len(X_val) == len(X)

dense_layers = [256, 128]
dropouts = [0.5]
dropout_decrements = [0.25]
batch_sizes = [16, 32]
learning_rates = [0.0003, 0.0001]
beta_1s = [0.9]
beta_2s = [0.99]
batch_normalizors = [True]
activations = ["relu"]

hyperparameter_combinations = []

import itertools
iterables = [ dense_layers, dropouts, dropout_decrements, batch_sizes, learning_rates, beta_1s, beta_2s, batch_normalizors, activations ]
for dense_unit, dropout, dropout_decrement, batch_size, learning_rate, beta_1, beta_2, batch_normalizor, activation in itertools.product(*iterables):
    print(f"{dense_unit} {dropout} {dropout_decrement} {batch_size} {learning_rate} {beta_1} {beta_2} {batch_normalizor} {activation}")
    hyperparameter_set = {}
    hyperparameter_set["dense_unit"] = dense_unit
    hyperparameter_set["dropout"] = dropout
    hyperparameter_set["dropout_decrement"] = dropout_decrement
    hyperparameter_set["batch_size"] = batch_size
    hyperparameter_set["learning_rate"] = learning_rate
    hyperparameter_set["beta_1"] = beta_1
    hyperparameter_set["beta_2"] = beta_2
    hyperparameter_set["batch_normalizor"] = batch_normalizor
    hyperparameter_set["activation"] = activation
    hyperparameter_combinations.append(hyperparameter_set)
print(f"Total hyperparameter combinations: {len(hyperparameter_combinations)}")

def build_model(hyperparams):
    first_layer = True
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    dense_unit = hyperparams["dense_unit"]
    dropout = hyperparams["dropout"]
    dropout_decrement = hyperparams["dropout_decrement"]
    while (dense_unit >= 16):
        model.add(Dense(units=dense_unit))
        dense_unit = dense_unit // 2
        if (first_layer and hyperparams["batch_normalizor"]):
            model.add(BatchNormalization())
            first_layer = False
        
        if (activation == "relu"):
            model.add(Activation("relu"))
        
        if (dropout > 0):
            model.add(Dropout(rate=dropout))
            dropout -= dropout_decrement
    
    model.add(Dense(1, activation="linear"))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=hyperparams["learning_rate"], beta_1=hyperparams["beta_1"], beta_2=hyperparams["beta_2"]), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

best_model = None
best_hyperparameters = None
best_val_loss = 1024 # random high value
best_combination_index = 0

for i, hyperparams in enumerate(hyperparameter_combinations):
    tf.keras.backend.clear_session()
    model = build_model(hyperparams)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=300, batch_size=hyperparams["batch_size"], validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    best_epoch = np.argmin(history.history['val_loss']) + 1
    val_loss = history.history['val_loss'][best_epoch - 1]
    print(f"Combination {i + 1}/{len(hyperparameter_combinations)} - val loss {val_loss} - hyperparams {hyperparams}")
    
    if (val_loss < best_val_loss):
        best_model = model
        best_hyperparameters = hyperparams
        best_val_loss = val_loss
        best_combination_index = i + 1
        
print(f"Best combination {best_combination_index} - val loss {best_val_loss} - hyperparams {best_hyperparameters}")
model_name = "test_model"
best_model.save(f"../models/{model_name}.h5")
