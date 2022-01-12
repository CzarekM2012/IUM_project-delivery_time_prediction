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
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── validation     <- Scripts to assess the effectiveness of models: accuracy tests
    │   │   │                 and A/B tests
    │   │   └── visualize.py
    │   │  
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
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
endpoints --prefix=controllers --host=localhost:8000

Examples to run specific scripts:
python ./delivery_time/data/make_dataset.py data/raw data/processed
python ./delivery_time/models/train_model_naive.py data/processed/train_data.csv models/naive/naive_model.csv
python ./delivery_time/models/predict_model_naive.py models/naive/naive_model.csv "4,0,0,1,0,0,0,0,1,0,0"
python ./delivery_time/validation/test_acc.py naive data/processed/test_data.csv
