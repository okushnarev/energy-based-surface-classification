import pickle
from copy import copy
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import joblib
import pandas as pd
from python.utils.surface_classifier import Classifier, MedianModel, MetricNormal
from time import time
# noinspection PyUnresolvedReferences
from python.utils.dct import DCT


def task(df, cols, classifier, feature):
    df[cols] = df.apply(
        lambda x: classifier.classify_type_and_prob(x['movedir'], x[feature]), axis=1, result_type='expand')
    return df



if __name__ == '__main__':
    prepared_data = Path('data/prepared')
    feature = 'Ke1_kalman'

    df = pd.read_csv(prepared_data / 'linear.csv')

    classifier = Classifier()
    classifier.info['primary_feature'] = feature
    classifier.alpha = 0.9

    surfs = ['gray', 'green', 'table']
    num_of_coeffs = {s: 3 for s in surfs}

    with open('data/dct_models.pkl', 'rb') as file:
        model, model_std = pickle.load(file)

    for surf in surfs:
        surf_metric = MetricNormal(surf, feature, model_std[surf])

        surf_model = MedianModel(surf, feature)
        surf_model.func = partial(model[surf].numpy_func, scaled=True)
        classifier.add_surf(surf_model, surf_metric)

    # save classsifier
    joblib.dump(classifier, f'data/surface_classifier.joblib')

    # test classifier
    cols = []
    for cls_type in ('raw', 'memory'):
        cols += [f'cls_{cls_type}'] + [f'{s}_{cls_type}' for s in surfs]

    args = ((_df, cols, copy(classifier), feature) for (surf, movedir), _df in
            df.groupby(['surf', 'movedir']))


    print(f'Starting test for alpha: {classifier.alpha}')
    start_time = time()
    with Pool(processes=6) as p:
        res = p.starmap(task, args)
    end_time = time()
    print(f'Time taken: {end_time - start_time:.4f} s.')
    print('Exporting results to file')

    export_path = Path('data/evaluation')
    export_path.mkdir(parents=True, exist_ok=True)
    pd.concat(res, axis=0).to_csv(export_path / 'cls_results.csv', index=False)