# -*- coding: utf-8 -*-
import click
import logging
from pandas import DataFrame
from pathlib import Path
from os.path import join

import pandas
import torch
import torch.nn as nn
import torch.utils.data as data

from delivery_time.data.process_data import get_num_deliv, get_num_cities

NUM_CITIES = get_num_cities()
NUM_COMPANIES = get_num_deliv()
working_dir = ''
train_filepath = 'train_data.csv'
test_filepath = 'test_data.csv'
BATCH_SIZE = 32
EPOCHS_NUM = 60

SEED = 42
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class DeliveryTimeframeRegressor(nn.Module):
    WEEKDAYS_COUNT = 7

    def __init__(self, cities_count, transporters_count):
        super().__init__()
        self.purchase_weekday_embed = nn.Linear(self.WEEKDAYS_COUNT, self.WEEKDAYS_COUNT)
        self.destination_city_embed = nn.Linear(cities_count, cities_count)
        self.transporter_embed = nn.Linear(transporters_count, transporters_count)
        self.Linear1 = nn.Linear(self.WEEKDAYS_COUNT + cities_count + transporters_count, int((self.WEEKDAYS_COUNT + cities_count + transporters_count)/2))
        self.Linear2 = nn.Linear(int((self.WEEKDAYS_COUNT + cities_count + transporters_count)/2), 2)
        self.sigm = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.0001)

    def forward(self, encoded_purchase_weekday, encoded_destination_city, encoded_transporter):
        embedded_weekday = self.purchase_weekday_embed(encoded_purchase_weekday)
        embedded_destination = self.destination_city_embed(encoded_destination_city)
        embedded_transporter = self.transporter_embed(encoded_transporter)
        x = torch.cat((embedded_weekday, embedded_destination, embedded_transporter), dim=1)
        x = self.Linear1(x)
        x = self.sigm(x)
        x = self.Linear2(x)
        x = self.relu(x)
        return x

def loss_function(predicted_timeframes, delivery_times, timeframe_beginning_shift=16, timeframe_end_shift=16):
    low = predicted_timeframes[:, 0]
    high = predicted_timeframes[:, 1]
    low_target = delivery_times - timeframe_beginning_shift
    low_target_limit = torch.Tensor([0] * low_target.shape[0]).to(device)
    low_target = torch.maximum(low_target, low_target_limit)
    high_target = delivery_times + timeframe_end_shift
    return torch.mean(torch.pow(low - low_target, 2) + torch.pow(high - high_target, 2))

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) # a file
@click.argument('output_filepath', type=click.Path())           # a file
def main(input_filepath, output_filepath):
    """ Trains a regressor model on data from  (../../data/processed/).
    Model parameters are saved in (../../data/processed).
    """
    global working_dir

    logger = logging.getLogger(__name__)
    logger.info('training regressor model')

    data_dir = join(working_dir, input_filepath)
    #print(data_dir)

    with open(file=data_dir, mode='r', encoding='UTF-8') as file:
        df = pandas.read_csv(file)

    logger.info(f"loaded data, example:\n {df.head(1)}")

    data_tensor = torch.Tensor(df.values)
    data_tensor = torch.unique(input=data_tensor, dim=0)

    dataset = data.TensorDataset(data_tensor[:, :-1], data_tensor[:, -1])

    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DeliveryTimeframeRegressor(NUM_CITIES, NUM_COMPANIES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    loss_fun = loss_function

    model.train()
    for _ in range(EPOCHS_NUM):
        for samples, values in dataloader:
            samples, values = samples.to(device), values.to(device)
            weekdays = samples[:, 0].to(device=device, dtype=torch.long)
            destinations = samples[:, 1:1+NUM_CITIES]
            transporters = samples[:, 1+NUM_CITIES: 1+NUM_CITIES+NUM_COMPANIES]
            weekdays = nn.functional.one_hot(weekdays, num_classes=7).to(dtype=torch.float)

            optimizer.zero_grad()

            output = model(weekdays, destinations, transporters)
      
            loss = loss_fun(output, values)
            loss.backward()
            optimizer.step()
    

    logger.info(f"trained model")

    output_dir = join(working_dir, output_filepath)

    torch.save(model.state_dict(), output_dir)
    
    logger.info(f"model saved successfully, dir: {output_dir}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    working_dir = Path(__file__).resolve().parents[2]
    main()