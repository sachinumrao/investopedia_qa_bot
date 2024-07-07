import json

import pandas as pd


def parse_data():
    fname = "./data/articles_with_summaries_extended.json"
    
    with open(fname, "r") as f:
        data = json.load(f)
        
    print("Len data: ", len(data))
    
    parsed_data = []
    for item in data:
        title = item["title"]
        text = item["clean_text"]
        parsed_data.append((title, text))

    data_df = pd.DataFrame(parsed_data, columns=["Title", "Text"])
    print("Shape of Data: ", data_df.shape)
    print(data_df.head()) 
    
    output_file = "./data/investopedia_data.csv"
    data_df.to_csv(output_file, index=False)      
        
if __name__ == "__main__":
    parse_data()