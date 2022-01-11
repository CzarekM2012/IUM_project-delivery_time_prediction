# -*- coding: utf-8 -*-
import click
import logging
from pandas import DataFrame
from pathlib import Path
from os.path import join

import pandas
import torch

NUM_CITIES = 7
NUM_COMPANIES = 3
working_dir = ''

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) # a file
@click.argument('output_filepath', type=click.Path())           # a file
def main(input_filepath, output_filepath):
    """ Trains a naive model on data from  (../../data/processed/).
    Model parameters are saved in (../../data/processed).
    """
    global working_dir

    logger = logging.getLogger(__name__)
    logger.info('training naive model')

    data_dir = join(working_dir, input_filepath)
    #print(data_dir)

    df = ''

    with open(file=data_dir, mode='r', encoding='UTF-8') as file:
        df = pandas.read_csv(file)
    
    #print(df)

    logger.info(f"loaded data, example:\n {df.head(1)}")

    # The naive model is very simple and "learns" by calculating average delivery time for each combination
    count = torch.zeros(7, NUM_CITIES, NUM_COMPANIES)
    sum = torch.zeros(7, NUM_CITIES, NUM_COMPANIES)

    for index, row in df.iterrows():
        row = row.to_list()
        index = [0, 0, 0]
        index[0] = int(row[0])
        index[1] = row.index(1, 1) - 1
        index[2] = row.index(1, index[1]+2) - NUM_CITIES - 1
        
        count[index[0]][index[1]][index[2]] += 1
        sum[index[0]][index[1]][index[2]] += row[-1]
    
    model = sum / count
    #print(model)

    logger.info(f"trained model")

    output_dir = join(working_dir, output_filepath)

    with open(file=output_dir, mode='w', encoding='UTF-8', newline='') as file:
        file.write(DataFrame(model.flatten(1).numpy()).to_csv(index=False, header=False))
    
    logger.info(f"model saved successfully, dir: {output_dir}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    working_dir = Path(__file__).resolve().parents[2]
    main()