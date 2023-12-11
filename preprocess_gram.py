import pandas as pd
from tqdm import tqdm

from src.gram import get_gram

dataset_df = pd.read_feather('data/dataset.feather')

dataset_df['GRAM'] = None
for i in tqdm(range(len(dataset_df)-5)):
    dataset_df.at[i, 'GRAM'] = get_gram(dataset_df['VALUE'][i:i+5].tolist())

dataset_df.to_feather('data/dataset_gram.feather')