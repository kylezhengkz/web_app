import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
import glob
import pickle

class Model:
    __max_token_length = 250
    __max_embedding_length = __max_token_length * 300
    
    @staticmethod
    def __construct_model(npy_files):
        model = Sequential()
        model.add(Input(shape=(Model.__max_embedding_length,)))
        model.add(Dense(units=256))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=128))
        model.add(Activation("relu"))
        model.add(Dropout(rate=0.25))
        model.add(Dense(units=64))
        model.add(Activation("relu"))
        model.add(Dense(units=32))
        model.add(Activation("relu"))
        model.add(Dense(units=16))
        model.add(Activation("relu"))
        model.add(Dense(1, activation="linear"))
        model.summary()

        for i, layer in enumerate(model.layers):
            weights = []
            weight_index = 1
            while True:
                try:
                    weight = np.load(f"resources/npy/layer_{i+1}_weight_{weight_index}.npy")
                    weights.append(weight)
                    weight_index += 1
                except FileNotFoundError:
                    break
            if weights:
                layer.set_weights(weights)
        return model
    
    @staticmethod
    def __construct_embedding_dictionary(dictionary_files):
        embedding_dictionaries = glob.glob("resources/dictionaries/embedding_dictionary_*")
        embedding_dictionary = {}
        for dictionary_path in embedding_dictionaries:
            with open(dictionary_path, "rb") as f:
                embedding_dictionary_chunk = pickle.load(f)
                embedding_dictionary.update(embedding_dictionary_chunk)
        return embedding_dictionary
    
    def __init__(self):
        self.model = Model.__construct_model()
        self.embedding_dictionary = Model.__construct_embedding_dictionary()
    
    @staticmethod
    def __tokenize(text):
        text =  text.replace("'", "")
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        text = text.replace("_", " _ ")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        return text

    def __load_embeddings(self, text):
        embeddings = np.empty(Model.__max_embedding_length, dtype=np.float32)
        text = Model.__tokenize(text)
        tokens = text.split()
        
        for i, token in enumerate(tokens):
            if (i >= Model.__max_token_length):
                break
            if (token in self.embedding_dictionary):
                embeddings[(i*300):((i+1)*300)] = self.embedding_dictionary[token]
            else:
                embeddings[(i*300):((i+1)*300)] = np.zeros((300,), dtype=np.float32)
        
        if (len(tokens) > 0 and i < Model.__max_token_length - 1):
            room_left = ((Model.__max_token_length - 1) - i)*300
            embeddings[((i+1)*300):] = np.zeros((room_left,), dtype=np.float32)
        
        return embeddings
    
    def predict(self, review):
        embeddings = self.__load_embeddings(Model.__tokenize(review))
        X = []
        X.append(embeddings)
        X = np.array(X, dtype=np.float32)
            
        sentiment_score = self.model.predict(X)[0][0]
        sentiment_score = "%.2f" % sentiment_score
        return sentiment_score
