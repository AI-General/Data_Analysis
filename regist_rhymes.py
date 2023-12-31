__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import dotenv
import os
import pandas as pd
import chromadb
from tqdm import tqdm
import uuid

from src.parallel import chunks
from src.rhyme import difference_process

dotenv.load_dotenv()
COLLECTION_NAME = "rhymes_8"

#####################################################################################################################
# Create collection
#####################################################################################################################

dataset_df = pd.read_feather('data/dataset.feather')
print("Data loaded")

# dataset_df['Datetime'] = pd.to_datetime(dataset_df['DATE'].astype(str) + ' ' + dataset_df['TIME'].astype(str))
# dataset_df = dataset_df.set_index('Datetime')

#####################################################################################################################
# Create collection
#####################################################################################################################

chroma_client = chromadb.PersistentClient(path="DB_rhymes")

try: 
    chroma_client.delete_collection(COLLECTION_NAME)
    print("Collection deleted")
except Exception as e:
    print(e)
    pass

collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "l2"} # hnsw:space are "l2", "ip, "or "cosine"
)
print("Collection created")

#####################################################################################################################
# Functions
#####################################################################################################################

def upsert_vectors_rhymes(collection, df, batch_size=100, window=1000, divide=16, sigma=8):
    step = int(window // divide)
    data_generator = map(lambda i: {
        'id': str(df['ID'][i]) + '_' + str(window),
        'value': difference_process(df['VALUE'][i:i+window].tolist(), 1000, sigma=sigma),
        'metadata': {
            'ID': int(df['ID'][i]),
            'date': str(df['DATE'][i]),
            'time': str(df['TIME'][i]),
            'window': window
        }
    }, range(0, len(df['VALUE']) - window, step)) # len(df['VALUE'])

    for vectors_chunk in tqdm(chunks(data_generator, batch_size=batch_size), desc=f'Upserting vectors, window: {window}'):
        collection.upsert(
            embeddings=[v['value'] for v in vectors_chunk],
            ids=[v['id'] for v in vectors_chunk],
            metadatas=[v['metadata'] for v in vectors_chunk]
        )

#####################################################################################################################
# Upsert
#####################################################################################################################
window = 500
divide = 16

while window < len(dataset_df):
    upsert_vectors_rhymes(collection, dataset_df, batch_size=100, window=int(window), divide=divide)
    upsert_vectors_rhymes(collection, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
    upsert_vectors_rhymes(collection, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
    upsert_vectors_rhymes(collection, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)
    window *= 2

print("Collection upserted")