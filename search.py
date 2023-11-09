import os
import dotenv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pinecone

matplotlib.use('Agg') 

dotenv.load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# Read the Excel file
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)

sample_df = pd.read_excel('data/SAMPLE_DATASET.xlsx', engine='openpyxl')
sample_data = sample_df.values.tolist()
sample_value = [item[1] for item in sample_data]

assert len(sample_value) == 1000

if sample_value[0] == None:
    sample_value[0] = 0

for i in range(1, 1000):
    if sample_value == None:
        sample_value[i] = sample_value[i-1]

result = index.query(
  vector=sample_value,
  top_k=1,
  include_values=True
)

df = pd.read_excel('data/DATASET_MASTER.xlsx')
x = [i for i in range(1000)]

print("Similarity score: ", result['matches'][0]['score'])
print("Date: ", df['DATE'][int(result['matches'][0]['id'])])
print("Time: ", df['TIME'][int(result['matches'][0]['id'])])

plt.plot(x, sample_value, color='red', label='sample')
plt.plot(x, result['matches'][0]['values'], color='blue', label='Search Result')
plt.legend()
plt.grid(True)

text = "Time: From " + str(df['DATE'][int(result['matches'][0]['id'])]) + " " + str(df['TIME'][int(result['matches'][0]['id'])]) + " to " + str(df['DATE'][int(result['matches'][0]['id'])+1000]) + " " + str(df['TIME'][int(result['matches'][0]['id'])+1000])


plt.show()
plt.savefig('my_figure.png')


# dataset_df = pd.read_excel('data/DATASET_MASTER.xlsx')

