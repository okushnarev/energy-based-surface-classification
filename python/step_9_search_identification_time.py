import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from python.utils.data_fitting import fit_data
from python.utils.misc import cosine_func

import plotly.express as px

if __name__ == '__main__':
    with open('data/dct_models.pkl', 'rb') as file:
        model, model_std = pickle.load(file)

    input_path = Path('data/prepared')
    fig_path = Path('figures/identification_time')
    fig_path.mkdir(parents=True, exist_ok=True)

    x = np.arange(0, 356, 1)
    indices = range(20, 61, 10)
    gains = np.arange(0.5, 4.1, 0.5)
    metrics = {
        'MAPE': mean_absolute_percentage_error,
        'RMSE': lambda *args: np.sqrt(mean_squared_error(*args)),
    }
    feature = 'Ke1_kalman'
    surfs = ['gray', 'green', 'table', ]
    dfs = {surf: pd.read_csv(input_path / f'{surf}.csv') for surf in surfs}

    rows_per_second = 33
    for coeffs_src in ('mape', 'mse'):
        df_dirs = pd.read_csv(f'data/id_directions/directions_{coeffs_src}.csv')
        for metric_name, metric in metrics.items():

            df_res = pd.DataFrame()
            for amount in range(3, 10):
                best_dirs = df_dirs.query('amount == @amount')['directions'].iloc[0]
                best_dirs = sorted([int(x) for x in best_dirs[1:-1].split(', ')])

                err_window = {}
                for idx, gain in enumerate(gains):
                    err_window[gain] = {}
                    window_width = int(rows_per_second * gain)
                    for surf in surfs:
                        _errors = []
                        for start_idx in indices:
                            df_temp = dfs[surf].query('movedir in @best_dirs').groupby(by=['movedir']).nth[
                                      start_idx:start_idx + window_width]
                            params_surf = fit_data(df_temp['movedir'], df_temp[feature],
                                                   method='leastsq')
                            _errors.append(
                                metric(
                                    model[surf].numpy_func(x, True),
                                    cosine_func(x, *params_surf)
                                )
                            )
                        err_window[gain][surf] = np.mean(_errors)

                _df = pd.DataFrame(err_window).reset_index(names=['surf'])
                _df['amount'] = amount
                df_res = pd.concat((df_res, _df), axis=0, ignore_index=True)

                plot_df = pd.melt(
                    df_res.drop('surf', axis=1).groupby('amount').mean().reset_index(),
                    id_vars=['amount'],
                    var_name='window_width',
                    value_name=metric_name
                )
                plot_df['time_to_register'] = plot_df['window_width'] * plot_df['amount']

                # Plot figure
                _plot_df = plot_df.query('time_to_register <= 25')
                fig = px.line(
                    _plot_df,
                    x='time_to_register',
                    y=metric_name,
                    color='amount',
                    markers=True,
                    hover_data=['window_width', 'amount'],
                    template='plotly_white',

                    color_discrete_sequence=px.colors.sequential.Agsunset,
                )

                fig.update_layout(
                    yaxis_title=f'{metric_name}',
                    xaxis_title='Identification time, s',
                    legend_title='Number of<br>directions',
                    font=dict(size=18, color='#000000', ),
                    height=500,
                    width=800,

                    legend=dict(
                        x=0.8,
                        y=1,
                        bgcolor='rgba(255,255,255, 0.6)',
                        font=dict(size=15),
                    ),
                    xaxis=dict(
                        mirror=True,
                        ticks='outside',
                        showline=False,
                        linecolor='#000000',
                        range=(0, 25.),
                        gridcolor='rgba(0,0,0,0.2)',
                        griddash='dot',

                    ),
                    yaxis=dict(
                        mirror=True,
                        ticks='outside',
                        showline=False,
                        linecolor='#000000',
                        # range=(1, 9),
                        nticks=8,
                        gridcolor='rgba(0,0,0,0.2)',
                        griddash='dot',

                    ),
                )

                fig.update_traces(
                    line=dict(width=3),
                    marker=dict(size=12, symbol='circle'),

                )

                fig.write_html(fig_path / f'{coeffs_src.upper()}src_{metric_name.upper()}metric.html', include_plotlyjs='cdn')
