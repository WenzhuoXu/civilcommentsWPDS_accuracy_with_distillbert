import pandas as pd
import datetime


def get_time(date_str):
    date_str = date_str[:19] # Remove all the fuss behind
    time = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()
    return time

if __name__ == "__main__":
    raw_dataset = pd.read_csv('datasets/civilcomments/civilcomments_v1.0/all_data_with_identities.csv')

    for index, row in raw_dataset.iterrows():
        row['created_date'] = get_time(row['created_date'])
    
    raw_dataset.sort_values(by=['created_date'], inplace=True)

    raw_dataset.to_csv('datasets/civilcomments/civilcomments_v1.0/all_data_with_identities_sorted.csv')

