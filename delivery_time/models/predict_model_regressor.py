# -*- coding: utf-8 -*-
import click
#import logging
from pandas import DataFrame
from pathlib import Path
from os.path import join

import pandas
import torch

from delivery_time.models.train_model_regressor import DeliveryTimeframeRegressor
from torch.nn.functional import one_hot
from torch import Tensor

from delivery_time.data.process_data import get_num_deliv, get_num_cities

NUM_CITIES = get_num_cities()
NUM_COMPANIES = get_num_deliv()
working_dir = ''

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True)) # a file
@click.argument('input', type=click.STRING)                     # a string containing a single data sample without delivery time, e.g. "4,0,0,1,0,0,0,0,1,0,0"
def main(model_filepath, input):
    result = predict_regressor(model_filepath, input)
    print(result)

def predict_regressor(model_filepath, input):
    """ Prints an answer predicted by regression model with parameters from (../../models/regressor/).
    """
    global working_dir

    #logger = logging.getLogger(__name__)

    model_dir = join(working_dir, model_filepath)
    #print(data_dir)

    state_dict = torch.load(model_dir)

    model = DeliveryTimeframeRegressor(NUM_CITIES, NUM_COMPANIES)
    model.load_state_dict(state_dict)
    
    input = [int(value) for value in input.split(',')]
    
    data_tensor = Tensor(input)

    weekdays = data_tensor[0].to(dtype=torch.long)
    destinations = data_tensor[1 : 1+NUM_CITIES].view((1, -1))
    transporters = data_tensor[1+NUM_CITIES : 1+NUM_CITIES+NUM_COMPANIES].view((1, -1))
    weekdays = one_hot(weekdays, num_classes=7).view((1, -1)).to(dtype=torch.float)

    prediction = model(weekdays, destinations, transporters) 
    prediction = prediction.detach()[0].tolist()

    return prediction[0], prediction[1]


if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    working_dir = Path(__file__).resolve().parents[2]
    main()