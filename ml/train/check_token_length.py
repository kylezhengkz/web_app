import pickle
import time
import re

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

x = []
for _, row in df.iterrows():
    x.append(len(tokenize(row["1-Star Reviews"])))
    x.append(len(tokenize(row["2-Star Reviews"])))
    x.append(len(tokenize(row["3-Star Reviews"])))
    x.append(len(tokenize(row["4-Star Reviews"])))
    x.append(len(tokenize(row["5-Star Reviews"])))

import matplotlib.pyplot as plt
range = (0, 2000)
bins = 100
plt.hist(x, bins, range, color = 'green', histtype = "bar", rwidth = 0.8)

plt.xlabel("Token count")
plt.ylabel("Frequency")
plt.title("Graph")
plt.show()
