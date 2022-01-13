delivery_time
==============================

A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── delivery_time      <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to extract and process data
    │   │   
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │   
    │   ├── validation     <- Scripts to assess the effectiveness of models: accuracy tests
    │   │                     and A/B tests
    │   │   
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │      
    ├── tests              <- Unit tests
    |
    ├── poetry.lock        <- Lockfile which allows complete environment reproduction
    │
    ├── pyproject.toml     <- file with settings and dependencies for the environment
    │
    └── controllers.py     <- script that starts the endpoint server


--------


Setting up the environment
------------

1. Install `poetry`: https://python-poetry.org/docs/#installation
2. Create an environment with `poetry install`
3. Run `poetry shell`
4. To add a new package run `poetry add <package>`. Don't forget to commit the lockfile.
5. To run unit tests for your service use `poetry run pytest` or simply `pytest` within `poetry shell`.

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Commands:
-------------

Run endpoint server:
`endpoints --prefix=controllers --host=localhost:8000`

Send a request for prediction:
`curl 127.0.0.1:8000/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-05-31T19:39:23"`
`curl 127.0.0.1:8000/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-05-31T19:39:23" -d "user_id=105"`

Direct requests to a specific model use processed data:
`curl 127.0.0.1:8000/naive/ -d "sample=4,0,0,1,0,0,0,0,1,0,0"`
`curl 127.0.0.1:8000/regressor/ -d "sample=4,0,0,1,0,0,0,0,1,0,0"`

Examples to run specific scripts:
`python ./delivery_time/data/process_data.py data/raw data/processed`

`python ./delivery_time/models/train_model_naive.py data/processed/train_data.csv models/naive/naive_model.csv`
`python ./delivery_time/models/train_model_regressor.py data/processed/train_data.csv models/regressor/regressor_model.pt`

`python ./delivery_time/models/predict_model_naive.py models/naive/naive_model.csv "4,0,0,1,0,0,0,0,1,0,0"`
`python ./delivery_time/models/predict_model_regressor.py models/regressor/regressor_model.pt "4,0,0,1,0,0,0,0,1,0,0"`

`python ./delivery_time/validation/test_acc.py http://127.0.0.1:8000/naive data/processed/test_data.csv`
`python ./delivery_time/validation/test_acc.py http://127.0.0.1:8000/regressor data/processed/test_data.csv`
