import numpy as np

from src.rhyme import rhyme_func
from src.utils import resample_normalize


def similarity(standard, normalized_sample, norm):
    resample = np.array(resample_normalize(standard, len(normalized_sample)))
    return np.dot(resample, normalized_sample) / (np.linalg.norm(resample) * norm)

def similarity_rhyme(standard, query_signal, norm):
    standard_rhyme = rhyme_func(standard, window=1000)
    return np.dot(standard_rhyme, query_signal) / (np.linalg.norm(standard_rhyme) * norm)

def detail_search(df_value, normalized_sample, window_param, id_param):
    step = int(window_param / 32)
    window_step = int(window_param / 8)
    
    ID = id_param
    WINDOW = window_param
    
    similarity_score = 0
    norm = np.linalg.norm(normalized_sample)
    
    for _ in range(5):
        for id in [ID - step, ID, ID + step]:
            for window in [WINDOW - window_step, WINDOW, WINDOW + window_step]:
                similarity_now = similarity(df_value[id:id+window], normalized_sample, norm)
                if similarity_now > similarity_score:
                    similarity_score = similarity_now
                    ID_TEMP = id
                    WINDOW_TEMP = window
        ID = ID_TEMP
        WINDOW = WINDOW_TEMP
        step = int(step / 2)
        window_step = int(window_step / 2)
    return ID, WINDOW, similarity_score


def detail_rhyme_search(df_value, query_signal, window_param, id_param):
    step = int(window_param / 32)
    window_step = int(window_param / 8)
    
    ID = id_param
    WINDOW = window_param
    
    similarity_score = 0
    norm = np.linalg.norm(query_signal)
    
    for _ in range(5):
        for id in [ID - step, ID, ID + step]:
            for window in [WINDOW - window_step, WINDOW, WINDOW + window_step]:
                similarity_now = similarity_rhyme(df_value[id:id+window], query_signal, norm)
                if similarity_now > similarity_score:
                    similarity_score = similarity_now
                    ID_TEMP = id
                    WINDOW_TEMP = window
        ID = ID_TEMP
        WINDOW = WINDOW_TEMP
        step = int(step / 2)
        window_step = int(window_step / 2)
    return ID, WINDOW, similarity_score
