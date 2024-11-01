import pickle
import numpy as np
import string

embedding_dictionary = {}

def contains_letter_or_number(token):
    for c in token:
        if ( (ord('a') <= ord(c) <= ord('z')) or (ord('0') <= ord(c) <= ord('9')) ):
            return True
    return False
    
def contains_upper(token):
    for c in token:
        if (ord('A') <= ord(c) <= ord('Z')):
            return True
    return False

# all tokens should be uncased
# token should either be all letters, or a single punctuation
def should_store(token):
    if (contains_upper(token)): # uppercase
        return False
    
    if (not contains_letter_or_number(token) and len(token) > 1): # punctuations with multiple characters (ex. "...")
        return False

    if (len(token.split()) > 1): # multiple tokens in one token (ex. "apple tree") - tokens should be separated by whitespace
        return False
    
    return True
    
with open("glove.840B.300d.txt", "r") as file:
    lines = file.readlines()
    print(f"{len(lines)} tokens to process")
    count = 0
    for line in lines:
        line = line.strip().split(' ')
        token = line[0]
        
        try:
            if (not should_store(token)):
                continue
        except:
            raise RuntimeError(token)
        
        if (contains_letter_or_number(token)): # tokens that contain letters should not contain any punctuations (ex. "he's" should be changed to "hes")
            new_token = token.translate(str.maketrans('', '', string.punctuation))
            if (new_token != token):
                token = new_token
                if (token in embedding_dictionary): # make sure transformed token is not a duplicate
                    continue
        
        embeddings = np.array(line[1:], dtype="float32")
        
        if (len(embeddings) != 300): # some tokens are only whitespace that gets trimmed anyway, which should be ignored
            print(f"Encountered a whitespace token")
            continue
        
        if (token not in embedding_dictionary):
            embedding_dictionary[token] = embeddings # store token
        
        count += 1
        if (count % 100000 == 0):
            print(count)

print(f"{len(embedding_dictionary)} tokens processed")

embedding_dictionarys = []
chunk_size = len(embedding_dictionary) // 15
print(f"Chunk size: {chunk_size}")

for i in range(15):
    if (len(embedding_dictionary) > chunk_size):
        chunk = dict(list(embedding_dictionary.items())[:chunk_size])
        embedding_dictionarys.append(chunk)
        embedding_dictionary = dict(list(embedding_dictionary.items())[chunk_size:])
        print(len(chunk))
        print(len(embedding_dictionary))
    else:
        print(len(embedding_dictionary))
        embedding_dictionarys.append(embedding_dictionary)
        
print(len(embedding_dictionarys))
        
for i, dictionary in enumerate(embedding_dictionarys):
    with open(f"embedding_dictionary_{i+1}.pkl", "wb") as file:
        pickle.dump(dictionary, file)
