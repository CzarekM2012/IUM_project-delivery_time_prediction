# -*- coding: utf-8 -*-
import click
import logging
from pandas import DataFrame
from pathlib import Path
from os.path import join
from json import loads
from datetime import datetime

DATA_SPLIT_TRAIN = 0.8
working_dir = ''

cities = ["Gdynia", "Kraków", "Poznań", "Radom", "Szczecin", "Warszawa", "Wrocław"]
deliv_comp = [360, 516, 620]

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) # a directory
@click.argument('output_filepath', type=click.Path())           # a directory
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../../data/raw) into
        cleaned data ready to be analyzed (saved in ../../data/processed).
    """
    global working_dir

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_dir = join(working_dir, input_filepath)
    #print(data_dir)

    data = join_data_files(join(data_dir, 'sessions.jsonl'),
                           join(data_dir, 'users.jsonl'),
                           join(data_dir, 'products.jsonl'),
                           join(data_dir, 'deliveries.jsonl'))

    logger.info(f"loaded data, example: {data[0]}")

    data.sort(key = lambda x: x[8]) # Sort by purchase date to split into sets later

    processed_data = []
    # = [[row[1], row[8], row[9], row[10]]
    for row in data:

        sample = process_raw(row[1], row[10], row[8])

        # Numerical data:
        date_time_str = row[8]
        date_time_str = date_time_str[:date_time_str.find('.')] # Strip seconds' fragments
        date_time_purchase = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')

        date_time_str = row[9]
        date_time_str = date_time_str[:date_time_str.find('.')]
        date_time_delivery = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')

        time_diff = date_time_delivery - date_time_purchase
        diff_h = time_diff.total_seconds() / 3600

        sample.append(diff_h)

        # Take weekday, city, delivery company and delivery time in hours
        processed_data.append(sample)
        
        # Old data processing:
        #processed_data.append([date_time_purchase.weekday(), row[1], row[10], diff_h])

    df = DataFrame.from_records(processed_data)

    #df = df.join(df[1].str.get_dummies()).drop(1, axis=1)
    #df = df.join(df[2].astype(str).str.get_dummies()).drop(2, axis=1)
    
    column_names = ["weekday"]
    column_names.extend(cities)
    column_names.extend(deliv_comp)
    column_names.append("delivery_time")

    df.columns=column_names
    #df = df[[c for c in df if c != 'delivery_time'] + ['delivery_time']] # Swap column order

    all_samples = df.shape[0]
    train_samples = int(all_samples * DATA_SPLIT_TRAIN)
    test_samples = all_samples - train_samples

    logger.info(f"splitting data into sets; all: {all_samples}, train: {train_samples}, test: {test_samples}")

    output_dir = join(working_dir, output_filepath, 'train_data.csv')
    with open(file=output_dir, mode='w', encoding='UTF-8', newline='') as file:
        file.write(df.head(train_samples).to_csv(index=False))
    
    logger.info(f"train data saved successfully, dir: {output_dir}")

    output_dir = join(working_dir, output_filepath, 'test_data.csv')
    with open(file=output_dir, mode='w', encoding='UTF-8', newline='') as file:
        file.write(df.tail(test_samples).to_csv(index=False))

    logger.info(f"test data saved successfully, dir: {output_dir}")


def get_num_cities():
    return len(cities)

def get_num_deliv():
    return len(deliv_comp)

def process_raw(city, delivery_company, purchase_timestamp):
    sample = []

    date_time = datetime.strptime(purchase_timestamp, '%Y-%m-%dT%H:%M:%S')
    sample.append(date_time.weekday())

    city_index = cities.index(city)
    deliv_comp_index = deliv_comp.index(delivery_company)

    for i in range(get_num_cities()):
        if i == city_index:
            sample.append(1)
        else:
            sample.append(0)

    for i in range(get_num_deliv()):
        if i == deliv_comp_index:
            sample.append(1)
        else:
            sample.append(0)    

    return sample

def get_data_string_from_raw(city, delivery_company, purchase_timestamp):
    if type(delivery_company) != int:
        delivery_company = int(delivery_company)
    print(type(delivery_company))
    
    return str(process_raw(city, delivery_company, purchase_timestamp))[1:-1]

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
    working_dir = Path(__file__).resolve().parents[2]
    main()