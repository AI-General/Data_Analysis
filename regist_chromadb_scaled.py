__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import dotenv
import pandas as pd
import chromadb 
import numpy as np
from tqdm import tqdm
import uuid

from src.parallel import chunks
from src.utils import resample_non_drop

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

# dataset_df = pd.read_excel(os.getenv('DATASET_PATH'))
# print("Data loaded")

# dataset_df['Datetime'] = pd.to_datetime(dataset_df['DATE'].astype(str) + ' ' + dataset_df['TIME'].astype(str))
# dataset_df = dataset_df.set_index('Datetime')
dataset_df = pd.read_feather('data/dataset.feather')
# dataset_df_resampled = dataset_df.resample('D').mean()

# sns.set_theme(style="whitegrid")
# sns.lineplot(x=dataset_df_resampled.index, y=dataset_df_resampled['VALUE'])

def upsert_vectors_k(collection, df, batch_size=100, window=1000, divide=16):
    step = int(window // divide)
    data_generator = map(lambda i: {
        'id': str(uuid.uuid4()),
        'value': list(resample_non_drop(df['VALUE'][i:i+window].tolist() - np.mean(df['VALUE'][i:i+window].tolist()), 1000)),
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

window = 500
divide = 16

while window < len(dataset_df):
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window), divide=divide)
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window*5/4), divide=divide)
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window*6/4), divide=divide)
    upsert_vectors_k(collection, dataset_df, batch_size=100, window=int(window*7/4), divide=divide)
    window *= 2

print("Upsert complete")