from endpoints import Controller
from endpoints.decorators import param
from delivery_time.models.predict_model_naive import predict_naive
from delivery_time.models.predict_model_regressor import predict_regressor
from delivery_time.data.process_data import get_data_string_from_raw
import json
import random

MODEL_PATH_NAIVE = "models/naive/naive_model.csv"
MODEL_PATH_REGRESSOR = "models/regressor/regressor_model.pt"

AB_TESTS = False
TEST_SPLIT_A = 0.5
MODEL_A = predict_naive
MODEL_B = predict_regressor
MODEL_PATH_A = MODEL_PATH_NAIVE
MODEL_PATH_B = MODEL_PATH_REGRESSOR

class Default(Controller):
    def GET(self):
        return """
        This main endpoint, send a POST request here with a data sample to obtain predictions for said input data
        Or send a request directly to a specified model at /naive or /regressor endpoints
        Example of a valid request using 'curl' command: `curl 127.0.0.1:8000/ -d "city=Warszawa" -d "delivery_company=360" -d "purchase_timestamp=2021-05-31T19:39:23" -d "user_id=105"`
        user_id and delivery_company can be both integer and strings
        Predictions can only be made for specific cities and delivery company IDs
        """

    @param('user_id', default=None, help="Just for A/B tests")
    def POST(self, **kwargs):
        sample = get_data_string_from_raw(kwargs['city'], kwargs['delivery_company'], kwargs['purchase_timestamp'])
        user_id = kwargs['user_id']

        model = MODEL_A
        model_path = MODEL_PATH_A

        if AB_TESTS:
            if user_id == None: # Unknown user, randomizing according to split, even though little useful data can be gained without identifying the user. User ID is not set to avoid collision with existing users
                if random.randint(0, 999999) / 1000000 > TEST_SPLIT_A:
                    model = MODEL_B
                    model_path = MODEL_PATH_B

            elif (hash(str(user_id)) % 1000000) / 1000000 > TEST_SPLIT_A: # First parsed to string, because hash(int x) = x
                model = MODEL_B
                model_path = MODEL_PATH_B

        results = model(model_path, sample)
        response = {
            "pred_time_from" : results[0],
            "pred_time_to" : results[1]
        }

        return json.dumps(response)

    #def POST(self, **kwargs):
    #    return 'hello {}'.format(kwargs['name'])

class Naive(Controller):
    def GET(self):
        return """
        This is naive model's endpoint
        send a POST request with a data 'sample', e.g.:
        curl 127.0.0.1:8000/naive/ -d "sample=4,0,0,1,0,0,0,0,1,0,0"
        to get a prediction for given data sample
        """
    def POST(self, **kwargs):
        sample = kwargs['sample']
        results = predict_naive(MODEL_PATH_NAIVE, sample)
        response = {
            "pred_time_from" : results[0],
            "pred_time_to" : results[1]
        }
        return json.dumps(response)

class Regressor(Controller):
    def GET(self):
        return """
        This is regression model's endpoint
        send a POST request with a data 'sample', e.g.:
        curl 127.0.0.1:8000/regressor/ -d "sample=4,0,0,1,0,0,0,0,1,0,0"
        to get a prediction for given data sample
        """
    def POST(self, **kwargs):
        sample = kwargs['sample']
        results = predict_regressor(MODEL_PATH_REGRESSOR, sample)
        response = {
            "pred_time_from" : results[0],
            "pred_time_to" : results[1]
        }
        return json.dumps(response)