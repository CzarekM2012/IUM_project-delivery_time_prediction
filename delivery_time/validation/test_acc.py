# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from os.path import join
import pandas
import requests
import json

working_dir = ''

@click.command()
@click.argument('endpoint_address', type=click.STRING) # URL of specific endpoint, e.g. http://127.0.0.1:8000/naive
@click.argument('test_data_filepath', type=click.Path(exists=True)) # a file
def main(endpoint_address, test_data_filepath):
    """ Sends on data from (../../data/processed/) to an endpoint
    and calculates accuracy based on labels in the data and received predictions
    """
    global working_dir

    logger = logging.getLogger(__name__)
    logger.info(f'testing model at {endpoint_address} for accuracy')

    data_dir = join(working_dir, test_data_filepath)
    #print(data_dir)

    df = ''

    with open(file=data_dir, mode='r', encoding='UTF-8') as file:
        df = pandas.read_csv(file)
    
    logger.info(f"loaded data, example:\n {df.head(1)}")

    acc_sum = 0
    window_size_sum = 0.0
    for index, row in df.iterrows():
        goal = row[-1]
        
        preds = get_pred(endpoint_address, row)

        if preds[0] > preds[1]:
            raise Exception("Predicted time window has negative size")

        if goal >= preds[0] and goal <= preds[1]: 
            acc_sum += 1

        window_size_sum += preds[1] - preds[0]

    row_count = df.shape[0]
    accuracy = acc_sum / row_count
    window_size_avg = window_size_sum / row_count

    logger.info(f"model at {endpoint_address} achieved accuracy of {accuracy} with average window size {window_size_avg}h")

def get_pred(endpoint, data):
    data_list = data.to_list()[:-1] # Last data point is the expected result

    for i in range(len(data_list)):
        data_list[i] = int(data_list[i])

    data_str = str(data_list)
    data_str = data_str[1:-1]
    data_str = data_str.replace(" ", "")
    response = requests.post(endpoint, json={"sample": data_str})
    results = json.loads(response.json())
    return results["pred_time_from"], results["pred_time_to"]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    working_dir = Path(__file__).resolve().parents[2]
    main()