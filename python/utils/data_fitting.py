import numpy as np
from lmfit import Parameters, minimize


def fit_data(x_data, y_data, method='leastsq', scale=71 / 355):
    def cosine_lmfit(params, x, y):
        A_1 = params['A_1']
        A_2 = params['A_2']
        B = params['B']

        x = np.array(x) * scale
        y_fit = A_1 * np.cos(np.pi * (x * 4 / 72 + 2 / 72)) + A_2 * np.cos(np.pi * (x * 12 / 72 + 6 / 72)) + B

        return y - y_fit

    params = Parameters()

    for param in ['A_1', 'A_2']:
        params.add(param, min=0.02, max=0.6)
    params.add('B', min=0, max=8, value=np.mean(y_data))

    fitted_params = minimize(cosine_lmfit, params, args=(x_data, y_data), method=method)
    coeffs = [fitted_params.params[coeff].value for coeff in ('B', 'A_1', 'A_2')]
    return coeffs
