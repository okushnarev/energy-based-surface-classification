import pickle
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from python.utils.models_wrappers import SklearnModel, CatBoostModel, NNModel


def calc_diff(val, movedir, model, surfs):
    result = []
    for surf in surfs:
        model_val = model[surf].numpy_func(movedir, scaled=True)
        result.append(abs(model_val - val))
    return result


def calc_single_ds(df, window_size, model, model_std, movedir, ds_path):
    result_frames = []
    for idx, start in enumerate(range(len(df) - window_size + 1)):
        df_window = df.iloc[start:start + window_size]
        prepare_dataset(df_window, model, model_std, result_frames, idx)
    df_prep = pd.DataFrame(dict(zip(['run_id', 'alpha', 'dKe', 'std_surf'], result_frames)))
    df_prep['n_std'] = df_prep['dKe'] / df_prep['std_surf']
    df_prep.to_csv(ds_path / f'{movedir}_prep.csv', index=False)


def prepare_dataset(df_window: pd.DataFrame, model: dict, model_std: dict, result: list, run_id: int,
                    surfs=('gray', 'green', 'table')):
    """

    :param df_window: window piece of a dataset
    :param model: dct model for each surf
    :param model_std: std values for each surf
    :param result: where to put results
    :param run_id:
    :param surfs:
    """
    # Compute diffs for each row in the window
    diffs = df_window.apply(
        func=lambda x: calc_diff(
            val=x['Ke1_corr'],
            movedir=x['movedir'],
            model=model,
            surfs=surfs),
        axis=1,
        result_type='expand'
    )

    run_id = [run_id] * len(surfs)
    mean_movedir = [df_window['movedir'].mean()] * len(surfs)
    mean_dKe = diffs.mean().tolist()
    model_std = [model_std[surf] for surf in surfs]
    out = [run_id, mean_movedir, mean_dKe, model_std]

    if not result:
        result.extend(out)
    else:
        for idx, item in enumerate(out):
            result[idx].extend(item)


if __name__ == '__main__':
    models_path = Path('ml_models')
    raw_data_path = Path('data/prepared')
    prep_data_path = Path('data/detection/test')
    eval_path = Path('data/evaluation')
    prep_data_path.mkdir(parents=True, exist_ok=True)

    with open('data/dct_models.pkl', 'rb') as file:
        model, model_std = pickle.load(file)

    ml_models = {
        'decision_tree':       SklearnModel(models_path / 'decision_tree.joblib'),
        'random_forest':       SklearnModel(models_path / 'random_forest.joblib'),
        'logistic_regression': SklearnModel(models_path / 'logreg.joblib'),
        'catboost':            CatBoostModel(models_path / 'catboost.cbm'),
        'neural_network':      NNModel(path_to_model=models_path / 'nn.pth',
                                       path_to_scaler=models_path / 'nn_scaler.pkl'),
    }

    dfs = {
        'linear': pd.read_csv(raw_data_path / 'linear.csv'),
        'square': pd.read_csv(raw_data_path / 'square.csv'),
        'circle': pd.read_csv(raw_data_path / 'circle.csv'),
    }
    window_size = 66

    prep_datasets = False
    if prep_datasets:
        print('Calculating datasets...')
        # Linear
        params = ((df, window_size, model, model_std, movedir, prep_data_path) for movedir, df in
                  dfs['linear'].groupby('movedir'))
        with Pool(processes=6) as p:
            p.starmap(calc_single_ds, params)

        # Square, Circle
        for name in ('square', 'cricle'):
            calc_single_ds(dfs[name], window_size, model, model_std, name, prep_data_path)

        print('\nCalculating datasets is done\n')

    # Evaluation
    # Linear
    surfs = ['gray', 'green', 'table', 'red']
    big_df = pd.DataFrame(columns=['movedir', 'model', 'accuracy', 'surf'])
    big_df_overall = pd.DataFrame()
    for movedir, df in dfs['linear'].groupby('movedir'):
        try:
            df_prep = pd.read_csv(prep_data_path / f'{movedir}_prep.csv')
        except FileNotFoundError:
            print(f'No such file or directory: {movedir}_prep.csv\n')
            print(f'Check if dataset is calculated for this direction: {movedir}')
            continue

        # Evaluation part
        eval_results = {surf: {m: {} for m in ('accuracy', 'f_0', 'f_1')} for surf in surfs}
        df_short = df.iloc[window_size - 1:, :].copy().reset_index(drop=True)
        for ml_name, ml_model in ml_models.items():
            df_prep['res'] = ml_model.predict(df_prep[['alpha', 'dKe', 'std_surf', 'n_std']])
            result = df_prep[['run_id', 'res']].groupby('run_id').prod()
            df_short[f'is_new_{ml_name}'] = result.astype(bool)

            df_short['is_new'] = result.astype(bool)

            for surf, _df_short in df_short.groupby('surf'):
                eval_results[surf]['accuracy'][ml_name] = accuracy_score(_df_short['is_new_true'],
                                                                         _df_short['is_new'])

                eval_results[surf]['f_0'][ml_name] = f1_score(_df_short['is_new_true'], _df_short['is_new'],
                                                              pos_label=0, zero_division=0)
                eval_results[surf]['f_1'][ml_name] = f1_score(_df_short['is_new_true'], _df_short['is_new'],
                                                              pos_label=1, zero_division=0)
        big_df_overall = pd.concat((big_df_overall, df_short), ignore_index=True)

        for surf in surfs:
            temp_df = pd.DataFrame({
                'movedir':  [movedir] * len(eval_results[surf]['accuracy']),
                'model':    list(eval_results[surf]['accuracy'].keys()),
                'accuracy': list(eval_results[surf]['accuracy'].values()),
                'f_0':      list(eval_results[surf]['f_0'].values()),
                'f_1':      list(eval_results[surf]['f_1'].values()),
                'surf':     [surf] * len(eval_results[surf]['accuracy']),
            })
            big_df = pd.concat(
                (big_df, temp_df),
                axis=0,
                ignore_index=True
            )
    big_df.to_csv(eval_path / f'detector_linear.csv', index=False)

    eval_results_f1 = {
        'model':      [],
        'f1_overall': []
    }
    for ml_name, ml_model in ml_models.items():
        eval_results_f1['model'].append(ml_name)
        eval_results_f1['f1_overall'].append(
            f1_score(big_df_overall['is_new_true'], big_df_overall[f'is_new_{ml_name}']))

    eval_results_f1 = pd.DataFrame(eval_results_f1)
    eval_results_f1.to_csv(eval_path / 'f1_overall_linear.csv', index=False)

    # Square and Circle
    big_df = pd.DataFrame(columns=['dataset', 'model', 'accuracy', 'f_0', 'f_1'])
    df_types = ('circle', 'square')
    for df_type in df_types:
        df_prep = pd.read_csv(prep_data_path / f'{df_type}_prep.csv')
        eval_results = {m: {} for m in ('accuracy', 'f_0', 'f_1')}
        for ml_name, ml_model in ml_models.items():
            df_prep['res'] = ml_model.predict(df_prep[['alpha', 'dKe', 'std_surf', 'n_std']])
            result = df_prep[['run_id', 'res']].groupby('run_id').prod()
            df_short = dfs[df_type].iloc[window_size - 1:, :].copy().reset_index(drop=True)
            df_short['is_new'] = result.astype(bool)

            eval_results['accuracy'][ml_name] = accuracy_score(df_short['is_new_true'], df_short['is_new'])
            eval_results['f_0'][ml_name] = f1_score(df_short['is_new_true'], df_short['is_new'], pos_label=0)
            eval_results['f_1'][ml_name] = f1_score(df_short['is_new_true'], df_short['is_new'], pos_label=1)

        print(f'Models evaluation for {df_type.upper()} dataset')
        print(
            pd.DataFrame(
                data=list(eval_results['accuracy'].values()),
                index=list(eval_results['accuracy'].keys()),
                columns=['accuracy']
            ).sort_values('accuracy', ascending=False)
        )
        print()

        big_df = pd.concat(
            (big_df,
             pd.DataFrame({
                 'dataset':  [df_type] * len(eval_results['accuracy']),
                 'model':    list(eval_results['accuracy'].keys()),
                 'accuracy': list(eval_results['accuracy'].values()),
                 'f_0':      list(eval_results['f_0'].values()),
                 'f_1':      list(eval_results['f_1'].values()),
             })),
            axis=0,
            ignore_index=True,
        )

    big_df.to_csv(eval_path / 'detector_square_circle.csv', index=False)
