from json import loads
from os import pardir
from os.path import dirname, join
import csv


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


data_dir = join(dirname(__file__), pardir, 'Data')

data = join_data_files(join(data_dir, 'sessions.jsonl'),
                       join(data_dir, 'users.jsonl'),
                       join(data_dir, 'products.jsonl'),
                       join(data_dir, 'deliveries.jsonl'))

with open(file='data.csv', mode='w', encoding='UTF-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
