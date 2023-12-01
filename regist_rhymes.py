__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import itertools
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import chromadb 
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample
from tqdm import tqdm
import uuid


dotenv.load_dotenv()
COLLECTION_NAME = "rhymes_8"

#####################################################################################################################
# Create collection
#####################################################################################################################

dataset_df = pd.read_excel(os.getenv('DATASET_PATH'))
print("Data loaded")

dataset_df['Datetime'] = pd.to_datetime(dataset_df['DATE'].astype(str) + ' ' + dataset_df['TIME'].astype(str))
dataset_df = dataset_df.set_index('Datetime')

#####################################################################################################################
# Create collection
#####################################################################################################################

chroma_client = chromadb.PersistentClient(path="DB")

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
def resample_non_drop(x, n):
    reflected_signal = np.concatenate((x[::-1], x, x[::-1]))
    resampled_reflected_signal = resample(reflected_signal, 3 * n)
    resampled_signal = list(resampled_reflected_signal[n:2*n])
    return resampled_signal
    
def get_differences(x, sigma=8):
    smoothed_x = gaussian_filter1d(x, sigma)
    return np.where((smoothed_x[1:] - smoothed_x[:-1]) >= 0, 1, 0).tolist()

def difference_process(x, window=1000, sigma=8):
    return get_differences(resample_non_drop(x, window), sigma)

def chunks_rhymes(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def upsert_vectors_rhymes(collection, df, batch_size=100, window=1000, divide=16, sigma=8):
    step = int(window // divide)
    data_generator = map(lambda i: {
        'id': str(uuid.uuid4()),
        'value': difference_process(df['VALUE'][i:i+window].tolist(), 1000, sigma=sigma),
        'metadata': {
            'ID': int(df['ID'][i]),
            'date': str(df['DATE'][i]),
            'time': str(df['TIME'][i]),
            'window': window
        }
    }, range(0, len(df['VALUE']) - window, step)) # len(df['VALUE'])

    for vectors_chunk in tqdm(chunks_rhymes(data_generator, batch_size=batch_size), desc=f'Upserting vectors, window: {window}'):
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