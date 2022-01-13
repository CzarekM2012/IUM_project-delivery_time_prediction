# -*- coding: utf-8 -*-
import click
#import logging
from pandas import DataFrame
from pathlib import Path
from os.path import join

import pandas
import torch

from delivery_time.data.process_data import get_num_deliv, get_num_cities

NUM_CITIES = get_num_cities()
NUM_COMPANIES = get_num_deliv()

WINDOW_SIZE = 32
working_dir = ''

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True)) # a file
@click.argument('input', type=click.STRING)                     # a string containing a single data sample without delivery time, e.g. "4,0,0,1,0,0,0,0,1,0,0"
def main(model_filepath, input):
    result = predict_naive(model_filepath, input)
    print(result)

def predict_naive(model_filepath, input):
    """ Prints an answer predicted by naive model with parameters from (../../models/naive/).
    """
    global working_dir

    #logger = logging.getLogger(__name__)

    model_dir = join(working_dir, model_filepath)
    #print(data_dir)

    df = ''

    with open(file=model_dir, mode='r', encoding='UTF-8') as file:
        df = pandas.read_csv(file, header=None)
    
    row = input.split(',')
    for i in range(len(row)):
        row[i] = int(row[i])
        
    index = [0, 0, 0]
    index[0] = int(row[0])
    index[1] = row.index(1, 1) - 1
    index[2] = row.index(1, index[1]+2) - NUM_CITIES - 1
    #print(index)
    anwser = df.to_numpy()[index[0]][index[1]*3 + index[2]]

    return max(0, anwser - WINDOW_SIZE/2), anwser + WINDOW_SIZE/2 # Window size is +-24h


if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    working_dir = Path(__file__).resolve().parents[2]
    main()