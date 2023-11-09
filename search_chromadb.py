__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from matplotlib import pyplot as plt

import dotenv
import numpy as np
import pandas as pd
import chromadb 

dotenv.load_dotenv()

chroma_client = chromadb.PersistentClient(path="DB")
collection = chroma_client.get_collection(name="my_collection")

sample_df = pd.read_excel('data/SAMPLE_DATASET.xlsx', engine='openpyxl')
sample_data = sample_df.values.tolist()
sample_value = [item[1] for item in sample_data]

assert len(sample_value) == 1000

if sample_value[0] == None:
    sample_value[0] = 0

for i in range(1, 1000):
    if sample_value == None:
        sample_value[i] = sample_value[i-1]

results = collection.query(
    query_embeddings=[sample_value],
    n_results=1
)

x = np.arange(1000)

id = int(results['ids'][0][0])
distance = results['distances'][0][0]

df = pd.read_excel('data/DATASET_MASTER.xlsx')
embedding = df['VALUE'][id:id+1000].tolist()

nor_embedding = np.array(embedding)/np.linalg.norm(np.array(embedding))
nor_sample = np.array(sample_value)/np.linalg.norm(np.array(sample_value))

print("Distance score: ", distance)
print("Date: ", df['DATE'][int(id)])
print("Time: ", df['TIME'][int(id)])

plt.plot(x, nor_sample, color='red', label='sample')
plt.plot(x, nor_embedding, color='blue', label='Search Result')
plt.legend()
plt.grid(True)

text = "Time: From " + str(df['DATE'][int(id)]) + " " + str(df['TIME'][int(id)]) + " to " + str(df['DATE'][int(id)+1000]) + " " + str(df['TIME'][int(id)+1000])

plt.show()
plt.savefig('result.png')