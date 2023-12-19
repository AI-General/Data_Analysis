__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import dotenv
import os
import pandas as pd
import chromadb
from tqdm import tqdm
import pinecone
import uuid

from src.parallel import chunks
from src.rhyme import difference_process, savgol_normalize, gaussian_normalize

dotenv.load_dotenv()

dataset_df = pd.read_feather('data/dataset.feather')

#####################################################################################################################
# Create collection
#####################################################################################################################
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_RHYMES")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT_RHYMES")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME_RHYMES")
# Read the Excel file
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

try:
    pinecone.delete_index(PINECONE_INDEX_NAME)
except Exception as e:
    print(e)
pinecone.create_index(PINECONE_INDEX_NAME, dimension=1000, metric='cosine')
pinecone.describe_index(PINECONE_INDEX_NAME)

print("Collection created")

index = pinecone.Index(PINECONE_INDEX_NAME)
#####################################################################################################################
# Functions
#####################################################################################################################
def upsert_vectors_rhymes(index, df, batch_size=100, window=1000, divide=16, sigma=8):
    step = int(window // divide)
    data_generator = map(lambda i: {
        'id': str(uuid.uuid4()),
        'values': gaussian_normalize(df['VALUE'][i:i+window].tolist(), window=1000, sigma=sigma),
        'metadata': {
            'ID': int(df['ID'][i]),
            'date': str(df['DATE'][i]),
            'time': str(df['TIME'][i]),
            'window': window
        }
    }, range(0, len(df['VALUE']) - window, step)) # len(df['VALUE'])

# def upsert_vectors_rhymes(index, df, batch_size=100, window=1000, divide=16, sigma=8):
#     step = int(window // divide)
#     data_generator = map(lambda i: {
#         'id': str(uuid.uuid4()),
#         'values': savgol_normalize(df['VALUE'][i:i+window].tolist(), 1000),
#         'metadata': {
#             'ID': int(df['ID'][i]),
#             'date': str(df['DATE'][i]),
#             'time': str(df['TIME'][i]),
#             'window': window
#         }
#     }, range(0, len(df['VALUE']) - window, step)) # len(df['VALUE'])

    for vectors_chunk in tqdm(chunks(data_generator, batch_size=batch_size), desc=f'Upserting vectors, window: {window}'):
        index.upsert(vectors=vectors_chunk)
        # collection.upsert(
        #     embeddings=[v['value'] for v in vectors_chunk],
        #     ids=[v['id'] for v in vectors_chunk],
        #     metadatas=[v['metadata'] for v in vectors_chunk]
        # )

#####################################################################################################################
# Upsert
#####################################################################################################################
window = 500
divide = 16
upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)
window = 1000

while window < len(dataset_df):
    upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window), divide=divide)
    upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
    upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
    upsert_vectors_rhymes(index, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)
    window *= 2

print("Collection upserted")