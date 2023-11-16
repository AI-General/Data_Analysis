__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import itertools
import os
import dotenv
import pandas as pd
import chromadb 
from scipy.signal import resample
import seaborn as sns
from tqdm import tqdm

dotenv.load_dotenv()


chroma_client = chromadb.PersistentClient(path="DB")
try: 
    chroma_client.delete_collection("my_collection_scaled")
    print("Collection deleted")
except Exception as e:
    print(e)
    pass

collection = chroma_client.create_collection(
    name="my_collection_scaled",
    metadata={"hnsw:space": "cosine"}
)
print("Collection created")

dataset_df = pd.read_excel(os.getenv('DATASET_PATH'))
print("Data loaded")

dataset_df['Datetime'] = pd.to_datetime(dataset_df['DATE'].astype(str) + ' ' + dataset_df['TIME'].astype(str))
dataset_df = dataset_df.set_index('Datetime')
dataset_df_resampled = dataset_df.resample('D').mean()

# sns.set_theme(style="whitegrid")
# sns.lineplot(x=dataset_df_resampled.index, y=dataset_df_resampled['VALUE'])

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def upsert_vectors_k(collection, df, batch_size=100, window=1000, divide=16):
    step = int(window // divide)
    data_generator = map(lambda i: {
        'id': str(df.index[i]),
        'value': list(resample(df['VALUE'][i:i+window].tolist() - np.mean(df['VALUE'][i:i+window].tolist()), 1000)),
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

window = 1000
divide = 16

while window < len(dataset_df):
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window), divide=divide)
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)
    window *= 2

print("Upsert complete")