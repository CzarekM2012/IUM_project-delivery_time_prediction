
Aleksander Garlikowski, Cezary Moczulski

delivery_time
==============================

A project for IUM (ML Engineering) for the following task:

"Wygląda na to, że nasze firmy kurierskie czasami nie radzą sobie z dostawami. Gdybyśmy
wiedzieli, ile taka dostawa dla danego zamówienia potrwa – moglibyśmy przekazywać tę
informację klientom."

The project consists of:
- a naive model
- an ML regressor
- dedicated training and prediction making scripts
- data processing scripts
- a microservice-based application that servers predictions and enables A/B testing

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── IUM-Projekt-Etap_1-Iter_2.pdf <- Data analysis report copy
    │   └── IUM-Projekt-Etap_2-Iter_1.pdf <- Final report copy
    │
    ├── delivery_time      <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── data           <- Scripts to extract and process data
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   └── validation     <- Scripts to assess the effectiveness of models: accuracy tests
    │                         and request traffic simulator
    |
    ├── poetry.lock        <- Lockfile which allows complete environment reproduction
    │
    ├── pyproject.toml     <- file with settings and dependencies for the environment
    │
    ├── controllers.py     <- script containing endpoint server code and configuration
    │
    └── server.log         <- file storing server app logs, can be used to compute A/B test results

--------


Setting up the environment
------------

1. Install `poetry`: https://python-poetry.org/docs/#installation
2. Create an environment with `poetry install`
3. Run `poetry shell`
4. To add a new package run `poetry add <package>`. Don't forget to commit the lockfile.
5. Run scripts within `poetry shell`. Possible commands below.

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Application setup
------------

The process re-generates process data and models from raw data included in project files

1. Run `poetry shell` in project directory. All subsequent commands will be run from there.
2. Run `python ./delivery_time/data/process_data.py data/raw data/processed` to process raw data
3. Run `python ./delivery_time/models/train_model_naive.py data/processed/train_data.csv models/naive/naive_model.csv` to train naive model
4. Run `python ./delivery_time/models/train_model_regressor.py data/processed/train_data.csv models/regressor/regressor_model.pt` to train the regressor
5. Run `endpoints --prefix=controllers --host=localhost:8000`. This will start the server app. To run on specific address, replace `localhost:8000` with another address.
6. Now the app is running. Test it by sending a request with data at server address (here: `localhost:8000/`). Example request: `curl 127.0.0.1:8000/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-05-31T19:39:23"`

Data in the request can be changed, but the app only returns predictions for valid cities and delivery companies. User id can also be proviced, e.g. `-d "user_id=105"` with `curl`, to identify the user for A/B tests. If A/B tests are toggled on (`AB_TESTS = True` at the start of `controllers.py`), requests are split between both models (setting `TEST_SPLIT_A` changes amount of data going to the first, naive, model). By default, split is set to `0.5`. Requests are split based on user id hash, so that a single user always gets predictions from a certain model. Results can be gathered later from `server.log`, where server app stores its predictions.

Predictions can be also obtained from a specific model by sending a request at its own endpoint, e.g. `localhost:8000/naive` for naive model.

Accuracy of both models can be tested with `test_acc.py` script, located in `delivery_time/validation`. Example command:
`python ./delivery_time/validation/test_acc.py http://127.0.0.1:8000/naive data/processed/test_data.csv`

Finally, a traffic simulation can be made using `simulate_traffic.py` from `delivery_time/validation`. Example command:
`python delivery_time/validation/simulate_traffic.py 10 0.001 100`, with parameters: simulation duration, base delay (multiplied randomly by up to 10) and user count.

Commands:
-------------

Run endpoint server:
`endpoints --prefix=controllers --host=localhost:8000`

Send a request for prediction:
`curl 127.0.0.1:8000/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-05-31T19:39:23"`
`curl 127.0.0.1:8000/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-05-31T19:39:23" -d "user_id=105"`

Direct requests to a specific model use processed data:
`curl 127.0.0.1:8000/naive/ -d "sample=4,0,0,0,0,0,1,0,1,0,0"`
`curl 127.0.0.1:8000/naive/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-06-04T19:39:23"`
`curl 127.0.0.1:8000/regressor/ -d "sample=4,0,0,0,0,0,1,0,1,0,0"`
`curl 127.0.0.1:8000/regressor/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-06-04T19:39:23"`

Examples to run specific scripts:
`python ./delivery_time/data/process_data.py data/raw data/processed`

`python ./delivery_time/models/train_model_naive.py data/processed/train_data.csv models/naive/naive_model.csv`
`python ./delivery_time/models/train_model_regressor.py data/processed/train_data.csv models/regressor/regressor_model.pt`

`python ./delivery_time/models/predict_model_naive.py models/naive/naive_model.csv "4,0,0,1,0,0,0,0,1,0,0"`
`python ./delivery_time/models/predict_model_regressor.py models/regressor/regressor_model.pt "4,0,0,1,0,0,0,0,1,0,0"`

`python ./delivery_time/validation/test_acc.py http://127.0.0.1:8000/naive data/processed/test_data.csv`
`python ./delivery_time/validation/test_acc.py http://127.0.0.1:8000/regressor data/processed/test_data.csv`

`python delivery_time/validation/simulate_traffic.py`
`python delivery_time/validation/simulate_traffic.py 10 0.001 100`

