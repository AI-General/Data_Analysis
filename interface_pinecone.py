__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from scipy.signal import resample
import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd
import numpy as np
import os
import dotenv
import pinecone

from src.rhyme import difference_process, savgol_normalize
from src.search import detail_search
from src.utils import clean_list, resample_non_drop, resample_normalize

dotenv.load_dotenv()

TOP_K = 3

dataset_df = pd.read_feather('data/dataset.feather')


def process(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)

    sample = resample_normalize(sample_value, window)
    
    scaled_sample = sample / \
        np.linalg.norm(sample)*embedding_norm + embedding_mean

    start = max(0, id - window)
    end = min(len(dataset_df['VALUE']), id + window * 2)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(start, end),
             dataset_df['VALUE'][start:end], color='blue', label='Search Result')
    plt.plot(np.arange(id, id+window), scaled_sample,
             color='red', label='Recaled Sample')
    plt.title("Most relevant search result")
    plt.xlabel("ID")
    plt.ylabel("Value")
    plt.legend()

    text = "Distance score: " + str(distance) + "\nStart Datetime: " + dataset_df['DATE'][id]+ " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S')

    # Saving to a temporary file and return its path
    plt.savefig(output_path)
    plt.close()
    return text, output_path


def process_80(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)

    # resampled_window_reflected_signal = resample(
    #     reflected_signal, 3*int(5/4*window))
    # sample = resampled_window_reflected_signal[int(
    #     5*window/4): 2*int(5*window/4)]
    sample = resample_non_drop(sample_value, int(5/4*window))

    sample = sample - np.mean(sample[:window])
    scaled_sample = sample / \
        np.linalg.norm(sample[:window])*embedding_norm + embedding_mean

    start = max(0, id - window)
    end = min(len(dataset_df['VALUE']), id + int(window * 5 / 2))

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(start, end),
             dataset_df['VALUE'][start:end], color='blue', label='Search Result')
    plt.plot(np.arange(id, id+window),
             scaled_sample[:window], color='red', label='Recaled Sample')
    plt.plot(np.arange(id+window, id+int(window * 5 / 4)),
             scaled_sample[window:int(window * 5 / 4)], color='green', label='Recaled Sample')
    plt.title("Most relevant search result")
    plt.xlabel("ID")
    plt.ylabel("Value")
    plt.legend()

    text = "Distance score: " + str(distance) + "\nStart Datetime: " + dataset_df['DATE'][id]+ " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S')

    plt.savefig(output_path)
    plt.close()

    return text, output_path


def process_70(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)

    # resampled_window_reflected_signal = resample(
    #     reflected_signal, 3*int(10/7*window))
    # sample = resampled_window_reflected_signal[int(
    #     10/7*window): 2*int(10/7*window)]
    sample = resample_non_drop(sample_value, int(10/7*window))

    sample = sample - \
        np.mean(sample[int(3/14*window): int(3/14*window) + window])
    scaled_sample = sample/np.linalg.norm(sample[int(3/14*window): int(
        3/14*window) + window])*embedding_norm + embedding_mean

    start = max(0, id - window - int(3/14*window))
    end = min(len(dataset_df['VALUE']), id + 2 * int(10/7*window))

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(start, end),
             dataset_df['VALUE'][start:end], color='blue', label='Search Result')
    plt.plot(np.arange(id-int(3/14*window), id),
             scaled_sample[:int(3/14*window)], color='green', label='Recaled Sample')
    plt.plot(np.arange(id, id+window),
             scaled_sample[int(3/14*window):window+int(3/14*window)], color='red', label='Recaled Sample')
    plt.plot(np.arange(id+window, id+window+int(3/14*window)),
             scaled_sample[window+int(3/14*window):window+2*int(3/14*window)], color='green', label='Recaled Sample')
    plt.title("Most relevant search result")
    plt.xlabel("ID")
    plt.ylabel("Value")
    plt.legend()

    text = "Distance score: " + str(distance) + "\nStart Datetime: " + dataset_df['DATE'][id]+ " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S')

    plt.savefig(output_path)
    plt.close()

    return text, output_path


def process_rhymes(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)

    sample = resample_non_drop(sample_value, window)
    sample = sample - np.mean(sample)

    scaled_sample = sample / \
        np.linalg.norm(sample)*embedding_norm + embedding_mean

    expand_window = int(window * 1 / 4)
    start = max(0, id - expand_window)
    end = min(len(dataset_df['VALUE']), id + window + expand_window)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(start, end),
             dataset_df['VALUE'][start:end], color='blue', label='Search Result')
    plt.plot(np.arange(id, id+window), scaled_sample,
             color='red', label='Recaled Sample')
    plt.title("Most relevant search result")
    plt.xlabel("ID")
    plt.ylabel("Value")
    plt.legend()

    text = "Distance score: " + str(distance) + "\nStart Datetime: " + dataset_df['DATE'][id]+ " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S')

    # Saving to a temporary file and return its path
    plt.savefig(output_path)
    plt.close()
    return text, output_path


def process_rhymes_80(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)

    sample = resample_non_drop(sample_value, int(5/4*window))
    sample = sample - np.mean(sample[:window])

    scaled_sample = sample / \
        np.linalg.norm(sample[:window])*embedding_norm + embedding_mean

    expand_window = int(window * 1 / 4)
    start = max(0, id - expand_window)
    end = min(len(dataset_df['VALUE']), id + int(5/4*window) + expand_window)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(start, end),
             dataset_df['VALUE'][start:end], color='blue', label='Search Result')
    plt.plot(np.arange(id, id+window),
             scaled_sample[:window], color='red', label='Recaled Sample')
    plt.plot(np.arange(id+window, id+int(window * 5 / 4)),
             scaled_sample[window:int(window * 5 / 4)], color='green', label='Recaled Sample')
    plt.title("Most relevant search result")
    plt.xlabel("ID")
    plt.ylabel("Value")
    plt.legend()

    text = "Distance score: " + str(distance) + "\nStart Datetime: " + dataset_df['DATE'][id]+ " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S')

    # Saving to a temporary file and return its path
    plt.savefig(output_path)
    plt.close()
    return text, output_path


def plot_from_xlsx(file_path):
    PINECONE_API_KEY_FULL = os.getenv("PINECONE_API_KEY_FULL")
    PINECONE_ENVIRONMENT_FULL = os.getenv("PINECONE_ENVIRONMENT_FULL")
    PINECONE_INDEX_NAME_FULL = os.getenv("PINECONE_INDEX_NAME_FULL")
    pinecone.init(api_key=PINECONE_API_KEY_FULL, environment=PINECONE_ENVIRONMENT_FULL)
    index = pinecone.Index(PINECONE_INDEX_NAME_FULL)
    
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()

    if sample_value[0] == None:
        sample_value[0] = 0

    for i in range(1, len(sample_value)):
        if np.isnan(sample_value[i]):
            sample_value[i] = sample_value[i-1]

    query_signal = resample_normalize(sample_value, 1000)
    results = index.query(
        vector=query_signal, top_k=TOP_K, include_metadata=True
    )
    
    normalized_sample = np.array(sample_value)
    normalized_sample = normalized_sample - np.mean(normalized_sample)
    
    detail_params = []
 
    for i in range(TOP_K):
        id, window, similarity_score = detail_search(
            df_value=dataset_df['VALUE'].tolist(),
            normalized_sample=normalized_sample,
            window_param=int(results['matches'][i]['metadata']['window']),
            id_param=int(results['matches'][i]['metadata']['ID'])
        )
        detail_params.append({
            'id': id,
            'window': window,
            'similarity_score': similarity_score
        })
    
    detail_params.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    result = []
    for i in range(TOP_K):
        text, path = process(
            dataset_df=dataset_df,
            sample_value=sample_value,
            id=detail_params[i]['id'],
            distance=detail_params[i]['similarity_score'],
            window=detail_params[i]['window'],
            output_path=f'image/my_figure{i}.png'
        )
        result.append(text)
        result.append(path)
    return result


def plot_from_xlsx_80(file_path):
    PINECONE_API_KEY_FULL = os.getenv("PINECONE_API_KEY_FULL")
    PINECONE_ENVIRONMENT_FULL = os.getenv("PINECONE_ENVIRONMENT_FULL")
    PINECONE_INDEX_NAME_FULL = os.getenv("PINECONE_INDEX_NAME_FULL")
    pinecone.init(api_key=PINECONE_API_KEY_FULL, environment=PINECONE_ENVIRONMENT_FULL)
    index = pinecone.Index(PINECONE_INDEX_NAME_FULL)
    
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()

    if sample_value[0] == None:
        sample_value[0] = 0

    for i in range(1, len(sample_value)):
        if np.isnan(sample_value[i]):
            sample_value[i] = sample_value[i-1]

    query_signal = resample_normalize(sample_value, 1250)

    results = index.query(
        vector=query_signal[:1000], top_k=TOP_K, include_metadata=True
    )
    
    normalized_sample = np.array(sample_value[:int(len(sample_value)*4/5)])
    normalized_sample = normalized_sample - np.mean(normalized_sample)
    
    detail_params = []
    
    for i in range(TOP_K):
        id, window, similarity_score = detail_search(
            df_value=dataset_df['VALUE'].tolist(),
            normalized_sample=normalized_sample,
            window_param=int(results['matches'][i]['metadata']['window']),
            id_param=int(results['matches'][i]['metadata']['ID'])
        )
        detail_params.append({
            'id': id,
            'window': window,
            'similarity_score': similarity_score
        })
    
    detail_params.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    result = []
    for i in range(TOP_K):
        text, path = process_80(
            dataset_df=dataset_df,
            sample_value=sample_value,
            id=detail_params[i]['id'],
            distance=detail_params[i]['similarity_score'],
            window=detail_params[i]['window'],
            output_path=f'image/my_figure{i}_80.png'
        )
        result.append(text)
        result.append(path)
    return result


def plot_from_xlsx_70(file_path):
    PINECONE_API_KEY_FULL = os.getenv("PINECONE_API_KEY_FULL")
    PINECONE_ENVIRONMENT_FULL = os.getenv("PINECONE_ENVIRONMENT_FULL")
    PINECONE_INDEX_NAME_FULL = os.getenv("PINECONE_INDEX_NAME_FULL")
    pinecone.init(api_key=PINECONE_API_KEY_FULL, environment=PINECONE_ENVIRONMENT_FULL)
    index = pinecone.Index(PINECONE_INDEX_NAME_FULL)
    
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()

    if sample_value[0] == None:
        sample_value[0] = 0

    for i in range(1, len(sample_value)):
        if np.isnan(sample_value[i]):
            sample_value[i] = sample_value[i-1]

    query_signal = resample_normalize(sample_value, int(10000/7))
    results = index.query(
        vector = query_signal[int(3/14*1000):int(3/14*1000)+1000], top_k=TOP_K, include_metadata=True
    )
  
    normalized_sample = np.array(sample_value[int(3/14*len(sample_value)): - int(3/14*len(sample_value))])
    normalized_sample = normalized_sample - np.mean(normalized_sample)
    
    detail_params = []
    
    for i in range(TOP_K):
        id, window, similarity_score = detail_search(
            df_value=dataset_df['VALUE'].tolist(),
            normalized_sample=normalized_sample,
            window_param=int(results['matches'][i]['metadata']['window']),
            id_param=int(results['matches'][i]['metadata']['ID'])
        )
        detail_params.append({
            'id': id,
            'window': window,
            'similarity_score': similarity_score
        })
    
    detail_params.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    result = []
    for i in range(TOP_K):
        text, path = process_70(
            dataset_df=dataset_df,
            sample_value=sample_value,
            id=detail_params[i]['id'],
            distance=detail_params[i]['similarity_score'],
            window=detail_params[i]['window'],
            output_path=f'image/my_figure{i}_70.png'
        )
        result.append(text)
        result.append(path)
    return result


def plot_from_xlsx_rhymes(file_path):
    PINECONE_API_KEY_RHYMES = os.getenv("PINECONE_API_KEY_RHYMES")
    PINECONE_ENVIRONMENT_RHYMES = os.getenv("PINECONE_ENVIRONMENT_RHYMES")
    PINECONE_INDEX_NAME_RHYMES = os.getenv("PINECONE_INDEX_NAME_RHYMES")
    pinecone.init(api_key=PINECONE_API_KEY_RHYMES, environment=PINECONE_ENVIRONMENT_RHYMES)
    index = pinecone.Index(PINECONE_INDEX_NAME_RHYMES)
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()

    clean_list(sample_value)
    difference_list = savgol_normalize(sample_value, window=1000)

    results = index.query(
        vector=difference_list, top_k=TOP_K, include_metadata=True
    )

    text0, path0 = process_rhymes(
        dataset_df=dataset_df,
        sample_value=sample_value,
        id=int(results['matches'][0]['metadata']['ID']),
        distance=results['matches'][0]['score'],
        window=int(results['matches'][0]['metadata']['window']),
        output_path='image/my_figure0_rhymes.png'
    )

    text1, path1 = process_rhymes(
        dataset_df=dataset_df,
        sample_value=sample_value,
        id=int(results['matches'][1]['metadata']['ID']),
        distance=results['matches'][1]['score'],
        window=int(results['matches'][1]['metadata']['window']),
        output_path='image/my_figure1_rhymes.png'
    )

    text2, path2 = process_rhymes(
        dataset_df=dataset_df,
        sample_value=sample_value,
        id=int(results['matches'][2]['metadata']['ID']),
        distance=results['matches'][2]['score'],
        window=int(results['matches'][2]['metadata']['window']),
        output_path='image/my_figure2_rhymes.png'
    )

    return text0, path0, text1, path1, text2, path2


def plot_from_xlsx_rhymes_80(file_path):
    PINECONE_API_KEY_RHYMES = os.getenv("PINECONE_API_KEY_RHYMES")
    PINECONE_ENVIRONMENT_RHYMES = os.getenv("PINECONE_ENVIRONMENT_RHYMES")
    PINECONE_INDEX_NAME_RHYMES = os.getenv("PINECONE_INDEX_NAME_RHYMES")
    pinecone.init(api_key=PINECONE_API_KEY_RHYMES, environment=PINECONE_ENVIRONMENT_RHYMES)
    index = pinecone.Index(PINECONE_INDEX_NAME_RHYMES)
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()

    clean_list(sample_value)
    difference_list = savgol_normalize(sample_value, window=1250)

    results = index.query(
        vector=difference_list[:1000], top_k=TOP_K, include_metadata=True
    )
    
    text0, path0 = process_rhymes_80(
        dataset_df=dataset_df,
        sample_value=sample_value,
        id=int(results['matches'][0]['metadata']['ID']),
        distance=results['matches'][0]['score'],
        window=int(results['matches'][0]['metadata']['window']),
        output_path='image/my_figure0_rhymes_80.png'
    )

    text1, path1 = process_rhymes_80(
        dataset_df=dataset_df,
        sample_value=sample_value,
        id=int(results['matches'][1]['metadata']['ID']),
        distance=results['matches'][1]['score'],
        window=int(results['matches'][1]['metadata']['window']),
        output_path='image/my_figure1_rhymes_80.png'
    )

    text2, path2 = process_rhymes_80(
        dataset_df=dataset_df,
        sample_value=sample_value,
        id=int(results['matches'][2]['metadata']['ID']),
        distance=results['matches'][2]['score'],
        window=int(results['matches'][2]['metadata']['window']),
        output_path='image/my_figure2_rhymes_80.png'
    )

    return text0, path0, text1, path1, text2, path2


with gr.Blocks() as demo:

    with gr.Tab("Full Search"):
        file_input = gr.File(type="filepath")

        text_field0 = gr.Textbox(label="Result0")
        text_field1 = gr.Textbox(label="Result1")
        text_field2 = gr.Textbox(label="Result2")

        image_field0 = gr.Image(label="Result0")
        image_field1 = gr.Image(label="Result1")
        image_field2 = gr.Image(label="Result2")

        iface = gr.Interface(fn=plot_from_xlsx, inputs=file_input, outputs=[
                             text_field0, image_field0, text_field1, image_field1, text_field2, image_field2])

    with gr.Tab("80% Search"):
        file_input_80 = gr.File(type="filepath")

        text_field0_80 = gr.Textbox(label="Result0")
        text_field1_80 = gr.Textbox(label="Result1")
        text_field2_80 = gr.Textbox(label="Result2")

        image_field0_80 = gr.Image(label="Result0")
        image_field1_80 = gr.Image(label="Result1")
        image_field2_80 = gr.Image(label="Result2")

        iface_80 = gr.Interface(fn=plot_from_xlsx_80, inputs=file_input_80, outputs=[
                                text_field0_80, image_field0_80, text_field1_80, image_field1_80, text_field2_80, image_field2_80])

    with gr.Tab("70% Search"):
        file_input_70 = gr.File(type="filepath")

        text_field0_70 = gr.Textbox(label="Result0")
        text_field1_70 = gr.Textbox(label="Result1")
        text_field2_70 = gr.Textbox(label="Result2")

        image_field0_70 = gr.Image(label="Result0")
        image_field1_70 = gr.Image(label="Result1")
        image_field2_70 = gr.Image(label="Result2")

        iface_70 = gr.Interface(fn=plot_from_xlsx_70, inputs=file_input_70, outputs=[
                                text_field0_70, image_field0_70, text_field1_70, image_field1_70, text_field2_70, image_field2_70])

    with gr.Tab("rhymes"):
        file_input_rhymes = gr.File(type="filepath")

        text_field0_rhymes = gr.Textbox(label="Result0")
        text_field1_rhymes = gr.Textbox(label="Result1")
        text_field2_rhymes = gr.Textbox(label="Result2")

        image_field0_rhymes = gr.Image(label="Result0")
        image_field1_rhymes = gr.Image(label="Result1")
        image_field2_rhymes = gr.Image(label="Result2")

        iface_rhymes = gr.Interface(fn=plot_from_xlsx_rhymes, inputs=file_input_rhymes, outputs=[
                             text_field0_rhymes, image_field0_rhymes, text_field1_rhymes, image_field1_rhymes, text_field2_rhymes, image_field2_rhymes])

    with gr.Tab("rhymes 80%"):
        file_input_rhymes_80 = gr.File(type="filepath")

        text_field0_rhymes_80 = gr.Textbox(label="Result0")
        text_field1_rhymes_80 = gr.Textbox(label="Result1")
        text_field2_rhymes_80 = gr.Textbox(label="Result2")

        image_field0_rhymes_80 = gr.Image(label="Result0")
        image_field1_rhymes_80 = gr.Image(label="Result1")
        image_field2_rhymes_80 = gr.Image(label="Result2")

        iface_rhymes_80 = gr.Interface(fn=plot_from_xlsx_rhymes_80, inputs=file_input_rhymes_80, outputs=[
                                text_field0_rhymes_80, image_field0_rhymes_80, text_field1_rhymes_80, image_field1_rhymes_80, text_field2_rhymes_80, image_field2_rhymes_80])


if __name__ == "__main__":
    # Launch the interface
    os.makedirs("image", exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
