# -*- coding: utf-8 -*-
import click
import logging
from pandas import DataFrame
from pathlib import Path
from os.path import join
import pandas
import torch
import requests
import json

NUM_CITIES = 7
NUM_COMPANIES = 3
SERVER_ADDRESS = "127.0.0.1:8000"
working_dir = ''

@click.command()
@click.argument('model_type', type=click.STRING) # location of the endpoint, either "naive" or "regressor"
@click.argument('test_data_filepath', type=click.Path(exists=True)) # a file
def main(model_type, test_data_filepath):
    """ Runs the script in (../models/).
    Model parameters are saved in (../../models/naive/).
    """
    global working_dir

    logger = logging.getLogger(__name__)
    logger.info(f'testing {model_type} model accuracy')

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
        #pred = system("python ./delivery_time/models/predict_model_naive.py models/naive/naive_model.csv '{row}'")
        preds = get_pred(model_type, row)

        if preds[0] > preds[1]:
            raise Exception("Predicted time window has negative size")

        if goal >= preds[0] and goal <= preds[1]: 
            acc_sum += 1

        window_size_sum += preds[1] - preds[0]

    row_count = df.shape[0]
    accuracy = acc_sum / row_count
    window_size_avg = window_size_sum / row_count

    logger.info(f"{model_type} model accuracy: {accuracy}, average window size: {window_size_avg}")

def get_pred(endpoint, data):
    data_list = data.to_list()[:-1] # Last data point is the expected result

    for i in range(len(data_list)):
        data_list[i] = int(data_list[i])

    data_str = str(data_list)
    data_str = data_str[1:-1]
    data_str = data_str.replace(" ", "")
    response = requests.post("http://" + SERVER_ADDRESS + "/" + endpoint, json={"sample": data_str})
    results = json.loads(response.json())
    return results["pred_time_from"], results["pred_time_to"]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    working_dir = Path(__file__).resolve().parents[2]
    main()