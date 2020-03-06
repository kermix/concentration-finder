from scipy.optimize import curve_fit
import numpy as np


class FourParametricLogistic:
    def __func(self, x, A, B, C ,D):
        """4PL lgoistic equation."""
        return ((A-D)/(1.0+((x/C)**B))) + D

    def fit(self, x, y):
        self.residuals, _ = curve_fit(self.__func, x, y)
        return self.residuals

    def predict(self, x):
        return self.__func(x, *self.residuals)

    def r2(self, x, y):
        x = np.array(x)
        y_pred = self.predict(x)

        ss_res = sum((y - y_pred)**2)
        ss_tot = sum((y - np.mean(y))**2)

        return 1 - (ss_res/ss_tot)
