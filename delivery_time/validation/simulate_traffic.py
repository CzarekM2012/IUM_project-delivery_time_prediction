# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from os.path import join
import requests
import json
import time
import random
import datetime

from delivery_time.data.process_data import cities, deliv_comp

SERVER_ADDRESS = "127.0.0.1:8000"
working_dir = ''

@click.command()
@click.argument('endpoint_address', type=click.STRING, default="http://127.0.0.1:8000/") # URL of an endpoint, e.g. http://127.0.0.1:8000/ for default main endpoint
@click.argument('duration_sec', type=click.INT, default=60)
@click.argument('base_delay_sec', type=click.FLOAT, default=0.1)
@click.argument('user_count', type=click.INT, default=10)
def main(endpoint_address, duration_sec, base_delay_sec, user_count):
    """ Generates and sends many requests distribued over time to given endpoint.
    Requests are generated based on parameters set in delivery_time/data/process_data.py
    """
    global working_dir

    logger = logging.getLogger(__name__)
    logger.info(f'simulating traffic at {endpoint_address}')

    start_time = time.time()
    while time.time() - start_time <= duration_sec:
        city = random.choice(cities)
        deliv_company = random.choice(deliv_comp)
        user_id = random.randrange(1000000000, 1000000000 + user_count) # 10^9 to aviod confusing results with normal users

        now = datetime.datetime.now()
        purchase_time = now.strftime('%Y-%m-%dT%H:%M:%S') # Formatted like: 2021-05-31T19:39:23
    

        requests.post(endpoint_address, json={"city": city, "delivery_company": deliv_company, "purchase_timestamp": purchase_time, "user_id": user_id})

        time.sleep(base_delay_sec * random.randrange(1, 10))
    
    logger.info(f"traffic simulation finished")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    working_dir = Path(__file__).resolve().parents[2]
    main()