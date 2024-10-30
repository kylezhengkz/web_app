import pickle
import time

t0 = time.time()
with open("embedding_dictionary.pkl", "rb") as f:
    data = pickle.load(f)
t1 = time.time()

print(f"{round(t1 - t0, 3)} seconds")
