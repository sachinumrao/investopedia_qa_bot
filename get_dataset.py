import json

import pandas as pd
from huggingface_hub import hf_hub_download


def download_data():
    repo_id = "ksrepo/investopedia-dataset"
    filename = "articles_with_summaries_extended.json"
    
    data = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir="./data/")

    
if __name__ == "__main__":
    download_data()