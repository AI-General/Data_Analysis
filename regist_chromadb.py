__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import itertools
import os
import dotenv
import pandas as pd
import chromadb 
from tqdm import tqdm

dotenv.load_dotenv()

chroma_client = chromadb.PersistentClient(path="DB")
try: 
    chroma_client.delete_collection("my_collection")
    print("Collection deleted")
except Exception as e:
    print(e)
    pass

collection = chroma_client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}
)
print("Collection created")

df = pd.read_excel('data/DATASET_MASTER.xlsx')
print("Data loaded")
print("length: ", len(df['VALUE']))

data_generator = map(lambda i: {
    'id': str(i),
    'values': df['VALUE'][i:i+1000].tolist(),
}, range(len(df['VALUE']) - 1000)) # len(df['VALUE'])

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


for vectors_chunk in tqdm(chunks(data_generator, batch_size=100), desc='Upserting vectors'):
    collection.upsert(
        embeddings=[v['values'] for v in vectors_chunk],
        ids=[v['id'] for v in vectors_chunk],
    )
    # index.upsert(vectors=vectors_chunk)