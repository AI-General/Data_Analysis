# DATA_ANALYSIS_PYTHON

## Install requirements
```bash
pip install -r requirements.txt
```

## Setup Environment
```bash
cp .env.example .env
```

## Regist Data
```bash
python3 regist_chromadb.py
```
This command make `DB` directory and regist from `data\DATASET_MASTER.xlsx`

## Search
```bash
python3 search_chromadb.py
```
This command search most similar recode with `data\SAMPLE_DATASET.xlsx`
And print `distance` and `Start time`
and plot graph of `sample` and `searched result` records
