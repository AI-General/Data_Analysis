import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd
import numpy as np
import os
import dotenv
import pinecone
from statsmodels.tsa.arima.model import ARIMA

from src.rhyme import rhyme_func
from src.search import detail_search, detail_rhyme_search
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
    plt.grid(True)

    text = "Distance score: " + str(distance) + \
        "\nStart Datetime: " + dataset_df['DATE'][id] + " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + \
        "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S') + \
        "\nID: " + str(id) + \
        "\nWindow: " + str(window)

    # Saving to a temporary file and return its path
    plt.savefig(output_path)
    plt.close()
    return text, output_path


def process_80(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)

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
    plt.grid(True)

    text = "Distance score: " + str(distance) + \
        "\nStart Datetime: " + dataset_df['DATE'][id] + " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + \
        "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S') + \
        "\nID: " + str(id) + \
        "\nWindow: " + str(window)

    plt.savefig(output_path)
    plt.close()

    return text, output_path


def process_70(dataset_df, sample_value, id, distance, window, output_path):
    embedding = dataset_df['VALUE'][id:id+window].tolist()
    embedding_mean = np.mean(embedding)
    embedding_norm = np.linalg.norm(embedding - embedding_mean)
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
    plt.grid(True)

    text = "Distance score: " + str(distance) + \
        "\nStart Datetime: " + dataset_df['DATE'][id] + " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + \
        "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S') + \
        "\nID: " + str(id) + \
        "\nWindow: " + str(window)
        
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
    plt.grid(True)

    text = "Distance score: " + str(distance) + \
        "\nStart Datetime: " + dataset_df['DATE'][id] + " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + \
        "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S') + \
        "\nID: " + str(id) + \
        "\nWindow: " + str(window)

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
    plt.grid(True)

    text = "Distance score: " + str(distance) + \
        "\nStart Datetime: " + dataset_df['DATE'][id] + " " + dataset_df['TIME'][id].strftime('%H:%M:%S') + \
        "\nEnd Datetime: " + dataset_df['DATE'][id+window]+ " " + dataset_df['TIME'][id+window].strftime('%H:%M:%S') + \
        "\nID: " + str(id) + \
        "\nWindow: " + str(window)

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
    query_signal = rhyme_func(sample_value, window=1000)
    # query_signal = gaussian_normalize(sample_value, window=1000, sigma=8)

    results = index.query(
        vector=query_signal, top_k=TOP_K, include_metadata=True
    )
 
    detail_params = []
 
    for i in range(TOP_K):
        id, window, similarity_score = detail_rhyme_search(
            df_value=dataset_df['VALUE'].tolist(),
            query_signal=query_signal,
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
        text, path = process_rhymes(
            dataset_df=dataset_df,
            sample_value=sample_value,
            id=detail_params[i]['id'],
            distance=detail_params[i]['similarity_score'],
            window=detail_params[i]['window'],
            output_path=f'image/my_figure{i}_rhymes.png'
        )
        result.append(text)
        result.append(path)
    return result


def plot_from_xlsx_rhymes_80(file_path):
    PINECONE_API_KEY_RHYMES = os.getenv("PINECONE_API_KEY_RHYMES")
    PINECONE_ENVIRONMENT_RHYMES = os.getenv("PINECONE_ENVIRONMENT_RHYMES")
    PINECONE_INDEX_NAME_RHYMES = os.getenv("PINECONE_INDEX_NAME_RHYMES")
    pinecone.init(api_key=PINECONE_API_KEY_RHYMES, environment=PINECONE_ENVIRONMENT_RHYMES)
    index = pinecone.Index(PINECONE_INDEX_NAME_RHYMES)
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()

    clean_list(sample_value)
    query_signal = rhyme_func(sample_value, window=1250)

    results = index.query(
        vector=query_signal[:1000], top_k=TOP_K, include_metadata=True
    )
    
    detail_params = []
 
    for i in range(TOP_K):
        id, window, similarity_score = detail_rhyme_search(
            df_value=dataset_df['VALUE'].tolist(),
            query_signal=query_signal[:1000],
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
        text, path = process_rhymes_80(
            dataset_df=dataset_df,
            sample_value=sample_value,
            id=detail_params[i]['id'],
            distance=detail_params[i]['similarity_score'],
            window=detail_params[i]['window'],
            output_path=f'image/my_figure{i}_rhymes_80.png'
        )
        result.append(text)
        result.append(path)
    return result


def search(search_type, file_path):
    if search_type == "Full Search":
        return plot_from_xlsx(file_path)
    elif search_type == "80% Search":
        return plot_from_xlsx_80(file_path)
    elif search_type == "70% Search":
        return plot_from_xlsx_70(file_path)
    elif search_type == "Rhyme Search":
        return plot_from_xlsx_rhymes(file_path)
    elif search_type == "80% Rhyme Search":
        return plot_from_xlsx_rhymes_80(file_path)


def arima_process(arima_type, file_path):
    if arima_type == "Forecast":
        return arima_forecast(file_path)
    elif arima_type == "Test":
        return arima_test(file_path)

def arima_forecast(file_path):
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()
    clean_list(sample_value)
    
    pdq = (4, 1, 2)
    model = ARIMA(sample_value, order=pdq)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=int(len(sample_value)/4))
    
    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(0, len(sample_value)), sample_value, color='blue', label='Original')
    plt.plot(np.arange(len(sample_value), len(sample_value)+len(forecast)), forecast, color='red', label='Forecast')
    plt.grid(True)
    plt.savefig('image/arima_forecast.png')
    return "", "image/arima_forecast.png"

def arima_test(file_path):
    sample_df = pd.read_excel(file_path.name, engine='openpyxl')
    sample_value = sample_df[sample_df.columns[1]].tolist()
    clean_list(sample_value)
    
    pdq = (4, 1, 2)
    model = ARIMA(sample_value[:len(sample_value) - int(len(sample_value)/4)], order=pdq)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=int(len(sample_value)/4))
    
    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(0, len(sample_value)), sample_value, color='blue', label='Original')
    plt.plot(np.arange(len(sample_value)-len(forecast), len(sample_value)), forecast, color='red', label='Forecast')
    plt.grid(True)
    plt.savefig('image/arima_test.png')
    return "", "image/arima_test.png"

with gr.Blocks() as demo:

    with gr.Tab("Search"):
        gr.Interface(
            fn = search,
            inputs=[
                gr.Radio(["Full Search", "80% Search", "70% Search", "Rhyme Search", "80% Rhyme Search"], label="Search Type"),
                gr.File(type="filepath", label="File"),
            ],
            outputs=[
                gr.Textbox(label="Result0"), 
                gr.Image(label="Result0"),
                gr.Textbox(label="Result1"), 
                gr.Image(label="Result1"), 
                gr.Textbox(label="Result2"), 
                gr.Image(label="Result2") 
            ],
        )
        
    with gr.Tab("ARIMA"):
        gr.Interface(
            fn = arima_process,
            inputs=[
                gr.Radio(["Forecast", "Test"]),
                gr.File(type="filepath", label="File"),
            ],
            outputs=[
                gr.Textbox(label="Result"),
                gr.Image(label="Result") 
            ],
        )

        

if __name__ == "__main__":
    # Launch the interface
    os.makedirs("image", exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
