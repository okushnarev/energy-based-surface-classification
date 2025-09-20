import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from python.step_1_visualize_data import linear_plot
from python.utils.dct import DCT

if __name__ == '__main__':
    prepared_data = Path('data/prepared')
    figures = Path('figures/DCT')

    df = pd.read_csv(prepared_data / 'linear.csv')

    features = ['Ke1_kalman', ]

    dct_models = {}
    dct_models_std = {}

    for ftr in features:
        res = df.groupby(['surf', 'movedir']).agg(
            mean=(ftr, 'mean'),
            p25=(ftr, lambda x: x.quantile(0.25)),
            p75=(ftr, lambda x: x.quantile(0.75)),
            std=(ftr, 'std'),
        ).reset_index()

        fig = linear_plot(res, ftr)

        x = np.arange(0, 356, 1)
        for idx, (surf, _df) in enumerate(res.groupby('surf')):
            dct = DCT(
                data=_df['mean'],
                cutoff_amount=3,
                range_width=355
            )
            dct_models[surf] = dct
            dct_models_std[surf] = _df['std'].mean()

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=dct.numpy_func(x, scaled=True),
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='DCT',
                    showlegend=True if idx == 0 else False,
                    legendgroup='DCT',
                )
            )

        export_path = figures / ftr
        export_path.mkdir(parents=True, exist_ok=True)

        fig.write_image(export_path / f'{ftr}_dct.svg')
        fig.write_html(export_path / f'{ftr}_dct.html', include_plotlyjs='cdn')

        # write ml_models
        with open('data/dct_models.pkl', 'wb') as file:
            pickle.dump((dct_models, dct_models_std), file)
