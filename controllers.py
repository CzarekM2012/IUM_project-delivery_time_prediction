from endpoints import Controller
from delivery_time.models.predict_model_naive import predict_naive
from delivery_time.models.predict_model_regressor import predict_regressor
import json

class Default(Controller):
    def GET(self):
        return """
        This endpoint is unused, send a POST request with a data sample at /naive or /regressor endpoints to obtain predictions for said input data
        Example of a valid request using 'curl' command: curl 127.0.0.1:8000/naive/ -d "sample=4,0,0,1,0,0,0,0,1,0,0"
        """

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
        model_path = "models/naive/naive_model.csv"
        sample = kwargs['sample']
        results = predict_naive(model_path, sample)
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
        curl 127.0.0.1:8000/naive/ -d "sample=4,0,0,1,0,0,0,0,1,0,0"
        to get a prediction for given data sample
        """
    def POST(self, **kwargs):
        model_path = "models/regressor/regressor_model.pt"
        sample = kwargs['sample']
        results = predict_regressor(model_path, sample)
        response = {
            "pred_time_from" : results[0],
            "pred_time_to" : results[1]
        }
        return json.dumps(response)