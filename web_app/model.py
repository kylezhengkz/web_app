import re
import numpy as np

class Model:
    __max_token_length = 250
    __max_embedding_length = __max_token_length * 300
    
    def __init__(self, model, embedding_dictionary):
        self.model = model
        self.embedding_dictionary = embedding_dictionary
    
    def __tokenize(text):
        text =  text.replace("'", "")
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        text = text.replace("_", " _ ")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        return text

    def __load_embeddings(text):
        embeddings = np.empty(Model.__max_embedding_length, dtype=np.float32)
        text = Model.__tokenize(text)
        tokens = text.split()
        
        for i, token in enumerate(tokens):
            if (i >= Model.__max_token_length):
                break
            if (token in Model.__embedding_dictionary):
                embeddings[(i*300):((i+1)*300)] = Model.__embedding_dictionary[token]
            else:
                embeddings[(i*300):((i+1)*300)] = np.zeros((300,), dtype=np.float32)
        
        if (len(tokens) > 0 and i < Model.__max_token_length - 1):
            room_left = ((Model.__max_token_length - 1) - i)*300
            embeddings[((i+1)*300):] = np.zeros((room_left,), dtype=np.float32)
        
        return embeddings
    
    def predict(review):
        embeddings = Model.__load_embeddings(Model.__tokenize(review))
        X = []
        X.append(embeddings)
        X = np.array(X, dtype=np.float32)
            
        sentiment_score = Model.model.predict(X)[0][0]
        sentiment_score = "%.2f" % sentiment_score
        return sentiment_score
