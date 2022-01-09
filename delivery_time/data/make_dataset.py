# -*- coding: utf-8 -*-
import click
import logging
import csv
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from os import pardir
from os.path import dirname, join
from json import loads
from datetime import datetime

project_dir = ''

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    global project_dir

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_dir = join(project_dir, input_filepath)
    #print(data_dir)

    data = join_data_files(join(data_dir, 'sessions.jsonl'),
                           join(data_dir, 'users.jsonl'),
                           join(data_dir, 'products.jsonl'),
                           join(data_dir, 'deliveries.jsonl'))

    logger.info(f"loaded data, example: \t{data[0]}")

    processed_data = []
    # = [[row[1], row[8], row[9], row[10]]
    for row in data:

        date_time_str = row[8]
        date_time_str = date_time_str[:date_time_str.find('.')] # Strip seconds' fragments
        date_time_purchase = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')

        date_time_str = row[9]
        date_time_str = date_time_str[:date_time_str.find('.')]
        date_time_delivery = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')

        time_diff = date_time_delivery - date_time_purchase
        diff_h = time_diff.total_seconds() / 3600

        # Take city, weekday, delivery company and delivery time in hours
        processed_data.append([row[1], date_time_purchase.weekday(), row[10], diff_h])

    df = DataFrame.from_records(processed_data)
    print(df)


    #with open(file=join(project_dir, 'data.csv'), mode='w', encoding='UTF-8', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerows(data)


def dummy_sum(a, b):
    """Used exclusively to showcase relative imports in tests. See
       tests/test_make_dataset.py in the repo.
    """
    return a + b


def find_matches(base: dict, search_list: list[dict], key: str) -> list[dict]:
    '''
    Returns subset of `search_list` with values corresponding to `key` equal
    value corresponding to `key` in `base`
    '''
    match_value = base[key]
    return [record for record in search_list if record[key] == match_value]


def filter_out_atr_val(input: list[dict], attribute: str, value) -> list[dict]:
    '''
    Returns subset of `input` where values corresponding to key `attribute`
    do not equal `value`.\n
    Assumes that all values of attribute are saved with the same data type or
    data types that can be compared,
    e.g. all of them are strings or all of them are numbers
    '''
    return [record for record in input if record[attribute] != value]


def filter_out_atr_vals_except(input: list[dict], attribute: str, value)\
                               -> list[dict]:
    '''
    Returns subset of `input` where values corresponding to key `attribute`
    equal `value`.\n
    Assumes that all values of attribute are saved with the same data type or
    data types that can be compared,
    e.g. all of them are strings or all of them are numbers
    '''
    return [record for record in input if record[attribute] == value]


def read_jsonl(file_path: str) -> list[dict]:
    dataset = []
    with open(file=file_path, mode='r', encoding='UTF-8') as file:
        for line in file.readlines():
            dataset.append(loads(line))
    return dataset


def join_data_files(sessions_file: str, users_file: str,
                    products_file: str, deliveries_file: str) -> list:
    joined_data = []

    sessions = read_jsonl(sessions_file)
    sessions = filter_out_atr_vals_except(sessions, 'event_type',
                                          'BUY_PRODUCT')
    users = read_jsonl(users_file)
    products = read_jsonl(products_file)
    deliveries = read_jsonl(deliveries_file)

    for delivery in deliveries:
        session = find_matches(delivery, sessions, 'purchase_id')
        if len(session) > 1:
            print(f'ERROR: Session matches:{len(session)}')
            break
        session = session[0]

        user = find_matches(session, users, 'user_id')
        if len(user) > 1:
            print(f'ERROR: User matches:{len(user)}')
            break
        user = user[0]

        product = find_matches(session, products, 'product_id')
        if len(product) > 1:
            print(f'ERROR: Session matches:{len(product)}')
            break
        product = product[0]

        concatenated_record = [user['name'], user['city'], user['street'],
                               product['product_name'], product['category_path'], product['price'],  # noqa: E501
                               session['timestamp'], session['offered_discount'],  # noqa: E501
                               delivery['purchase_timestamp'], delivery['delivery_timestamp'], delivery['delivery_company']]  # noqa: E501
        joined_data.append(concatenated_record)
    return joined_data




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()