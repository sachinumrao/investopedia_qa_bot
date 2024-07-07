import pandas as pd
from nltk.tokenize import sent_tokenize

DATA_FILE = "./../data/investopedia_data.csv"


def word_len_splitter(full_text, n_words=200):
    sents = sent_tokenize(full_text)
    
    chunks = []
    curr_len = 0
    curr_chunk = ""
    for item in sents:
        item_len = len(item.split())
        if curr_len + item_len < n_words:
            curr_len += item_len
            curr_chunk += " " + item
        else:
            chunks.append(curr_chunk)
            curr_len = len(item.split())
            curr_chunk = item

    return chunks
    

def chunkify(text):
    
    chunks = text.split("\n")
    chunks = [item.strip() for item in chunks if len(item)>10]
    
    full_text = " ".join(chunks)
    chunks = word_len_splitter(full_text, n_words=128)
    
    return chunks
    

def create_chunks():
    df = pd.read_csv(DATA_FILE)
    data_chunks = []
    
    for i in range(df.shape[0]):
        text = df["Text"].iloc[i]
        
        chunks = chunkify(text)
        for item in chunks:
            data_chunks.append(item)
            
    chunk_df = pd.DataFrame({"Text": data_chunks})
    
    output_chunk_file = "./../data/investopedia_chunks.csv"
    chunk_df.to_csv(output_chunk_file, index=False)


if __name__ == "__main__":
    create_chunks()