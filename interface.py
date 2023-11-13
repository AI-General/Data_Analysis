__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import dotenv
import os
import chromadb
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from scipy.signal import resample

dotenv.load_dotenv()

chroma_client = chromadb.PersistentClient(path="DB")
collection = chroma_client.get_collection(name="my_collection_scaled")

dataset_df = pd.read_excel(os.getenv('DATASET_PATH'))
print("Data loaded")
dataset_df['Datetime'] = pd.to_datetime(dataset_df['DATE'].astype(str) + ' ' + dataset_df['TIME'].astype(str))
dataset_df = dataset_df.set_index('Datetime')

# This function will be called when the user uploads an xlsx file
def plot_from_xlsx(file_path):
    # Read the xlsx file with pandas
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    # sample_data = sample_df.values.tolist()
    sample_value = sample_df[sample_df.columns[1]].tolist()
    assert len(sample_value) == 1000

    if sample_value[0] == None:
        sample_value[0] = 0

    for i in range(1, 1000):
        if np.isnan(sample_value[i]):
            sample_value[i] = sample_value[i-1]
    
    results = collection.query(
        query_embeddings=[sample_value],
        n_results=1
    )
    
    id = int(results['metadatas'][0][0]['ID'])
    distance = results['distances'][0][0]
    window = results['metadatas'][0][0]['window']

    embedding = dataset_df['VALUE'][id:id+window].tolist()
    sample = resample(sample_value, window)

    # nor_embedding = np.array(embedding)/np.linalg.norm(np.array(embedding))
    scaled_sample = np.array(sample)/np.linalg.norm(np.array(sample))*np.linalg.norm(np.array(embedding))
    
    start = max(0, id - window)
    end = min(len(dataset_df['VALUE']), id + window * 2)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(start, end), dataset_df['VALUE'][start:end], color='blue', label='Search Result')
    plt.plot(np.arange(id, id+window), scaled_sample, color='red', label='Recaled Sample')
    plt.title("Most relevant search result")
    plt.xlabel("ID")
    plt.ylabel("Value")
    plt.legend()
    
    print("Distance score: ", distance)
    print("Start Datetime: ", dataset_df.index[id])
    print("End Datetime: ", dataset_df.index[id+window])
    
    text = "Distance score: " + str(distance) + "\nStart Datetime: " + str(dataset_df.index[id]) + "\nEnd Datetime: " + str(dataset_df.index[id+window])
    
    # Saving to a temporary file and return its path
    plt.savefig('output.png')
    plt.close()
    return text, 'output.png'  

# Create a File input component on the interface
file_input = gr.File(type="filepath")

text_field = gr.Textbox(label="Result")

# Create the interface with the input and output components
iface = gr.Interface(fn=plot_from_xlsx, inputs=file_input, outputs=[text_field, "image"])

if __name__ == "__main__":
    # Launch the interface
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
