import os
import pandas as pd
import dotenv

dotenv.load_dotenv()

dataset_df = pd.read_excel(os.getenv('DATASET_PATH'))
print("Data loaded")
# dataset_df['Datetime'] = pd.to_datetime(
#     dataset_df['DATE'].astype(str) + ' ' + dataset_df['TIME'].astype(str))
# dataset_df = dataset_df.set_index('Datetime')

dataset_df.to_feather('data/dataset.feather')
