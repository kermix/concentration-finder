from scipy.optimize import curve_fit
import numpy as np

import json
from json import JSONEncoder,   JSONDecoder

class FourParametricLogistic():
    def __func(x, A, B, C, D):
        if C == 0 or B == 0:
            return np.inf
        return ((A-D)/(1.0+((x/C)**B))) + D

    def fit(self, x, y):
        self.residuals, _ = curve_fit(FourParametricLogistic.__func, x, y)
        return self.residuals

    def predict(self, x):
        return FourParametricLogistic.__func(x, *self.residuals)

    def r2(self, x, y):
        x = np.array(x)
        y_pred = self.predict(x)

        ss_res = sum((y - y_pred)**2)
        ss_tot = sum((y - np.mean(y))**2)

        return 1 - (ss_res/ss_tot)

    def from_json(json_data):
        data = json.loads(json_data)
        if 'residuals' in data:
            model = FourParametricLogistic()
            model.residuals = np.array(data['residuals'])
            return model

    def solve(self, y):
        A, B, C, D = self.residuals
        return C*((((A-D)/(y-D))-1)**(1/B))


class FourParametricLogisticEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o.__dict__



