import json
import pandas as pd
import pickle

start_index = 5000000 # data points seem to be partially sorted by date, and I prefer data that is more recent
category_max = 15000

reviews = {
    "1-Star Reviews": [],
    "2-Star Reviews": [],
    "3-Star Reviews": [],
    "4-Star Reviews": [],
    "5-Star Reviews": []
}

with open("yelp_academic_dataset_review.json", "r") as file:
    for idx, line in enumerate(file):
        if (idx < start_index):
            continue
        
        obj = json.loads(line)
        review = obj.get("text")
        star = obj.get("stars")
        
        if (len(reviews["1-Star Reviews"]) < category_max and star == 1):
            reviews["1-Star Reviews"].append(review)
        elif (len(reviews["2-Star Reviews"]) < category_max and star == 2):
            reviews["2-Star Reviews"].append(review)
        elif (len(reviews["3-Star Reviews"]) < category_max and star == 3):
            reviews["3-Star Reviews"].append(review)
        elif (len(reviews["4-Star Reviews"]) < category_max and star == 4):
            reviews["4-Star Reviews"].append(review)
        elif (len(reviews["5-Star Reviews"]) < category_max and star == 5):
            reviews["5-Star Reviews"].append(review)
        
        if (len(reviews["1-Star Reviews"]) == category_max and len(reviews["2-Star Reviews"]) == category_max and len(reviews["3-Star Reviews"]) == category_max and len(reviews["4-Star Reviews"]) == category_max and len(reviews["5-Star Reviews"]) == category_max):
            break

print(len(reviews["1-Star Reviews"]))
print(len(reviews["2-Star Reviews"]))
print(len(reviews["3-Star Reviews"]))
print(len(reviews["4-Star Reviews"]))
print(len(reviews["5-Star Reviews"]))

df = pd.DataFrame(reviews)

print(df)
df = df.sample(frac=1).reset_index(drop=True) # shuffle
print(df)

with open("dataset.pkl", "wb") as file:
    pickle.dump(df, file)
