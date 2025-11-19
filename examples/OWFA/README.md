# Open-Domain Wikipedia Fact Audit (OWFA): Traceability from Training Data

This repository provides utilities and workflows for tracing model predictions back to training data sources.  
It includes scripts for downloading datasets, preprocessing them, building indices, and running queries.

NOTE: This code particularly contians the flask app demo on WIkipedia data. You can also add more training data. However, you will have to add the scripts to download and process those datasets.
---

## 🚀 Setup

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install --upgrade git+https://github.com/huggingface/datasets.git
pip install -U 'spacy[cuda12x]'
python -m spacy download en_core_web_sm

```
Add your huggingface token in "config.json"

### 3. Install domynclaimalign Library

The system requires the `domynclaimalign` library (in packages folder) for claim verification and provenance tracing:

```bash
# Install from its folder (adjust path as needed)
cd /path/to/domynclaimalign
pip install -e .
cd /path/to/this/repo
```

## 📂 Utils Workflow

All dataset preparation steps are under the utils/ folder.
```bash
cd utils
python download_data.py ## Download Wikipedia data
python convert_arrow_to_jsonl.py ## Convert Arrow files to JSONL
python check_the_jsonl_file.py ## Validate JSONL file (print 5 samples)
python calculate_and_save_unigram_probrability.py
 ## Create unigram probabilities of tokens in vocab
```


## 📦 Indexing

Build an Infini-gram index from the processed Wikipedia data:
```bash
python -m infini_gram.indexing \
  --data_dir /workspace/Traceability/Traceability2.0/data/wiki \
  --save_dir /workspace/Traceability/Traceability2.0/data/wiki_index \
  --tokenizer olmo \
  --cpus 64 \
  --mem 512 \
  --shards 36 \
  --add_metadata \
  --ulimit 1048576
```
You can also use llama or gpt2 tokenizer instead of olmo.

## Running App
Run the app 
```bash
cd app
python flask_app.py
```
In the app, you can ask a query and the system will find traces and claim alignments, which you can check by clicking on the sentences on the answer.

## 🧪 Running Experiments

To run comprehensive experiments and evaluations, follow these steps:

### 1. Create Experiment Dataset
Generate prompts and evaluation datasets for experiments:
```bash
cd experiments
python create_prompt_dataset.py
```

### 2. Run Experiments
Execute the main experiment pipeline:
```bash
python run_experiments.py
```

### 3. Calculate and Save Metrics
Run the metrics calculation and save results:
```bash
chmod +x run.sh
./run.sh
```

The experiments will:
- Generate evaluation prompts for Wikipedia topics
- Run fact-checking and claim verification
- Calculate various metrics (coverage, hit@5, hallucination detection, etc.)
- Save results and metrics for analysis

Results will be saved in the `experiment_data/` directory.


## 🧪 Example Queries

You can run traceability checks with questions such as:
1. What was the Remdesevir drug originally developed for?
2. What are the technical components of natural language processing?
3. Tell me about Amitabh Bachchan, his early life and family.
4. What does Microsoft do?
5. What is a credit rating agency?
6. What is the definition of capital assets for United States Federal government accounting?

## ✅ Notes

- Ensure sufficient system memory and CPUs for large-scale indexing 
- Use the correct tokenizer (olmo in this case).
- All intermediate data is stored in /workspace/Traceability/Traceability2.0/data/.