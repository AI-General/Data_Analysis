import itertools
import os
import dotenv
import pandas as pd
import pinecone
from tqdm import tqdm

dotenv.load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# Read the Excel file
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

try:
    pinecone.delete_index(PINECONE_INDEX_NAME)
except Exception as e:
    print(e)
pinecone.create_index(PINECONE_INDEX_NAME, dimension=1000)
pinecone.describe_index(PINECONE_INDEX_NAME)
    
df = pd.read_excel('data/DATASET_MASTER.xlsx')

# Print the length of the 'VALUE' column
print(len(df['VALUE']))

# values = [df['VALUE'][i:i+1000].tolist() for i in range(len(df['VALUE']) - 1000)]
# values = [np.array(df['VALUE'][i:i+1000].tolist()) for i in range(len(df['VALUE']) - 1000)]
# means = np.mean(values, axis=1)
# std_devs = np.std(values, axis=1)
# normalized_values = [(values[i] - means[i]) / std_devs[i] if std_devs[i] != 0 else (values[i] - means[i]) for i in range(len(df['VALUE']) - 1000)]

# print("Finished calculating means and standard deviations")

# data_generator = map(lambda i: {
#     'id': str(i),
#     'values': normalized_values[i].tolist(),
# })

data_generator = map(lambda i: {
    'id': str(i),
    'values': df['VALUE'][i:i+1000].tolist(),
}, range(len(df['VALUE']) - 1000))

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
# normalized_values = [(values[i] - means[i]) / std_devs[i] if std_devs[i] != 0 else (values[i] - means[i]) for i in range(len(df['VALUE']) - 1000)]

# vectors = [{
#     'id': str(i), 
#     'values': normalized_values[i].tolist(),
#     } for i in range(len(df['VALUE']) - 1000)]

index = pinecone.Index(PINECONE_INDEX_NAME)
for vectors_chunk in tqdm(chunks(data_generator, batch_size=100), desc='Upserting vectors'):
    index.upsert(vectors=vectors_chunk)