import pickle
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from python.utils.data_fitting import fit_data
from python.utils.misc import cosine_func

import time


def split(lst: list, num_of_splits: int) -> List[list]:
    """
    Splits list into equally sized 'num_of_splits' lists
    :param lst: a list to split
    :param num_of_splits: number of splits
    :return: list of lists
    """
    k, m = divmod(len(lst), num_of_splits)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_of_splits)]


def find_dirs(given_dirs: list, params: dict) -> Tuple[tuple, float]:
    """
    Finds combination of directions that better approximates reference DCT function from a given list of combinations
    by checking mean 'metric' on all passed 'surfs'

    :param given_dirs: list of direction combinations
    :param kwargs: packed parameters. includes: 'surfs', 'data_by_dir', 'dct_by_surf', 'metric', 'x_ax'
    :return: (tuple of directions, mean error)
    """

    data = params['raw_data']
    gen = params['rng']
    metric = params['metric']
    dct_by_surf = params['dct_by_surf']
    x_ax = params['x_ax']

    best_dirs = None
    mean_err = float('inf')
    for dirs in given_dirs:
        errors_temp = []
        for amount in (np.array([0.5, 1.5, 3, 6]) * 33).astype(int):
            # calc errors for each surface
            for surf in params['surfs']:
                amount = min(amount, min([len(df) for df in data]))
                mean = [np.mean(gen.choice(data[surf].query('movedir == @dir'), amount, replace=False)) for dir in dirs]
                params_surf = fit_data(dirs, mean, method='leastsq')
                surf_error = metric(dct_by_surf[surf](x_ax, scaled=True),
                                    cosine_func(x_ax, *params_surf))
                errors_temp.append(surf_error)

        # check if better mean error
        m_err_temp = np.mean(errors_temp)
        if m_err_temp < mean_err:
            mean_err = m_err_temp
            best_dirs = dirs

    return best_dirs, mean_err


def find_initial_dirs(params: dict, combinations: Iterable, processes=4) -> Tuple[tuple, float]:
    """
    Brute force all combinations to find the directions combination that better approximates reference DCT function

    :param params: {'surfs':, 'data_by_dir':, 'dct_by_surf':, 'metric':, 'x_ax':}
    :param combinations: list of movement direction combinations
    :param processes: num of paralleled processes used in calculations
    :return:  (the best directions combination, mean error)
    """

    data = split(list(combinations), processes)
    data = [(d, params) for d in data]
    with Pool(processes) as p:
        res = p.starmap(find_dirs, data)

    best_dirs, mean_err = min(res, key=lambda x: x[1])
    return best_dirs, mean_err


def calc_next_dirs(directions: List, start_dirs: tuple, start_err: float, params) -> pd.DataFrame:
    """

    :param directions:
    :param start_dirs:
    :param start_err:
    :param path_to_save:
    :param params: {'surfs':, 'data_by_dir':, 'dct_by_surf':, 'metric':, 'x_ax':, 'name':}
    :return:
    """
    directions_left = directions.copy()
    best_dirs_improved = list(start_dirs)



    dirs_trace = [start_dirs]
    errors_trace = [start_err]
    # pop all best dirs from search
    for d in start_dirs:
        directions_left.remove(d)

    # perform search for next better coefficient
    while len(best_dirs_improved) < 10:
        combs = (best_dirs_improved + [d] for d in directions_left)

        best_dirs, mean_err = find_dirs(combs, params)
        best_dir = best_dirs[-1]
        best_dirs_improved.append(best_dir)
        directions_left.remove(best_dir)

        dirs_trace.append(tuple(best_dirs_improved))
        errors_trace.append(mean_err)

    df_err = pd.DataFrame({'amount': list(range(3, 11)), 'directions': dirs_trace, 'error': errors_trace})

    return df_err



if __name__ == '__main__':
    recalculate_dirs = False
    with open('data/dct_models.pkl', 'rb') as file:
        model, model_std = pickle.load(file)

    surfs = ['gray', 'green', 'table', ]

    x = np.arange(0, 355, 1)
    x_scaled = x * (71 / 355)

    feature = 'Ke1_kalman'
    input_path = Path(f'./data/prepared')
    output_path = Path(f'./data/id_directions')
    output_path.mkdir(exist_ok=True, parents=True)
    data = {surf: pd.read_csv(input_path / f'{surf}.csv') for surf in surfs}
    rng = np.random.default_rng(69)

    metrics = {
        'mse':  mean_squared_error,
        'mape': mean_absolute_percentage_error,
    }

    for metric_name, metric in metrics.items():

        params = {
            'surfs':       surfs,
            'raw_data':    data,
            'dct_by_surf': model,
            'metric':      metric,
            'x_ax':        x,
            'name':        metric_name,
            'rng':         rng,
        }

        # Find 3 initial dirs
        # Warning: It is super time consuming, so we store them after calculations
        lookup_directions = list(range(0, 181, 5))
        file_name = output_path / f'initial_dirs_{metric_name}.pkl'
        if not file_name.exists() or recalculate_dirs:
            print('Calculating initial directions')
            start_time = time.time()
            best_dirs, mean_err = find_initial_dirs(params, combinations(lookup_directions, 3), processes=8)
            end_time = time.time()
            print(f'It took {end_time - start_time:.3f}s')
            with open(file_name, 'wb') as file:
                pickle.dump((best_dirs, mean_err), file)

        else:
            print('Loading initial directions. '
                  'If you want to recalculate them, delete the file or set a variable recalculate_dirs to True')
            with open(file_name, 'rb') as file:
                best_dirs, mean_err = pickle.load(file)

        print('Initial results')
        print(f'Best dirs are: {best_dirs}')
        print(f'Best error is: {mean_err}\n')

        # Find next directions with greedy algorithm
        df_dirs = calc_next_dirs(lookup_directions, best_dirs, mean_err, params=params)
        df_dirs.to_csv(output_path / f'directions_{metric_name}.csv', index=False)
        print(f'All results\n{df_dirs}\n')
