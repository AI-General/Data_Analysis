import itertools
import os
import uuid
import dotenv
import pandas as pd
import pinecone
from tqdm import tqdm

from src.parallel import chunks
from src.utils import resample_normalize

dotenv.load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_FULL")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT_FULL")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME_FULL")


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
try:
    pinecone.delete_index(PINECONE_INDEX_NAME)
except Exception as e:
    print(e)
pinecone.create_index(PINECONE_INDEX_NAME, dimension=1000, metric="cosine")
pinecone.describe_index(PINECONE_INDEX_NAME)
print("Collection created")
index = pinecone.Index(PINECONE_INDEX_NAME)
    
dataset_df = pd.read_feather('data/dataset.feather')
#####################################################################################################################
# Functions
#####################################################################################################################
def upsert_vectors(index, df, batch_size=100, window=1000, divide=16):
    step = int(window // divide)
    data_generator = map(lambda i: {
        'id': str(uuid.uuid4()),
        'values': resample_normalize(df['VALUE'][i:i+window].tolist(), 1000),
        'metadata': {
            'ID': int(df['ID'][i]),
            'date': str(df['DATE'][i]),
            'time': str(df['TIME'][i]),
            'window': window
        }
    }, range(0, len(df['VALUE']) - window, step)) # len(df['VALUE'])

    for vectors_chunk in tqdm(chunks(data_generator, batch_size=batch_size), desc=f'Upserting vectors, window: {window}'):
        index.upsert(vectors=vectors_chunk)

#####################################################################################################################
# Upsert
#####################################################################################################################
window = 500
divide = 16
upsert_vectors(index, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
upsert_vectors(index, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
upsert_vectors(index, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)

window = 1000
while window < len(dataset_df):
    upsert_vectors(index, dataset_df, batch_size=100, window=int(window), divide=divide)
    upsert_vectors(index, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
    upsert_vectors(index, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
    upsert_vectors(index, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)
    window *= 2

print("Collection upserted")