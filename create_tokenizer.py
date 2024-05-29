import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the dataset
dataset_path = 'C:/Users/prana/PycharmProjects/hello/tweet_emotions.csv.xls'
df = pd.read_csv(dataset_path)

# Extract text data from the dataset
texts = df['content'].tolist()

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Save the tokenizer
tokenizer_path = 'C:/Users/prana/PycharmProjects/hello/tokenizer.pkl'
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Tokenizer has been saved to {tokenizer_path}")
