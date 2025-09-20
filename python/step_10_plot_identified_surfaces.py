import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from python.utils.data_fitting import fit_data
from python.utils.misc import cosine_func, init_matplotlib, my_pal
init_matplotlib()

if __name__ == '__main__':
    with open('data/dct_models.pkl', 'rb') as file:
        model, model_std = pickle.load(file)

    surfs = ['gray', 'green', 'table', 'red']

    feature = 'Ke1_kalman'
    input_path = Path(f'data/prepared')
    fig_path = Path(f'figures/identification_results')
    fig_path.mkdir(parents=True, exist_ok=True)
    data = {surf: pd.read_csv(input_path / f'{surf}.csv') for surf in surfs}

    dfs = {surf: pd.read_csv(input_path / f'{surf}.csv') for surf in surfs}
    x = np.arange(0, 356, 1)

    gain, amount = 0.5, 9
    window_width = int(33 * gain)

    coeffs_src = 'mape'
    df_dirs = pd.read_csv(f'data/id_directions/directions_{coeffs_src}.csv')

    colors = ['tab:blue', 'tab:purple', 'tab:red', 'tab:green', ]

    fig = plt.figure(figsize=(10, 5.5))

    plt.grid(axis='x', linestyle='--', alpha=0.3, color='k')

    plt.xticks(range(0, 355 + 1, 45), rotation=0)
    plt.xlim((0, 355))
    plt.ylabel(r'$K_e$')
    plt.xlabel('Direction, deg')

    for surf, df in dfs.items():

        stats = df.groupby('movedir')[feature].agg(
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
        ).reset_index()

        movedir = stats['movedir']
        p25 = stats['p25']
        p75 = stats['p75']

        plt.fill_between(
            x=movedir,
            y1=p25,
            y2=p75,
            alpha=0.2,
            color=my_pal[surf],
        )

        plt.plot(model[surf].numpy_func(x, scaled=True), color=my_pal[surf], linestyle=(1, (5, 3)), linewidth=1.5,
                 label='Reference model' if surf == surfs[0] else None)

        best_dirs = df_dirs.query('amount == @amount')['directions'].iloc[0]
        best_dirs = sorted([int(x) for x in best_dirs[1:-1].split(', ')])

        for idx, start_idx in enumerate(range(30, 31, 10)):
            df_temp = dfs[surf].query('movedir in @best_dirs').groupby('movedir').nth[
                      start_idx:start_idx + window_width]
            params_surf = fit_data(df_temp['movedir'], df_temp[feature], method='leastsq')
            mean_by_dir = df_temp.groupby('movedir')[feature].mean()

            plt.plot(x, cosine_func(x, *params_surf), color=colors[idx], linestyle='-', linewidth=2.5,
                     label='Identified model' if surf == surfs[0] else None)
            if idx == 0:
                box = True
                if box:
                    boxplot_color = '#6b6b6b'
                    boxplot_alpha = 1
                    boxplot_linewidth = 1.3

                    plt.boxplot(
                        [x[feature] for _, x in df_temp[['movedir', feature]].groupby('movedir')],
                        positions=df_temp['movedir'].unique(),
                        manage_ticks=False,
                        widths=5,

                        showfliers=False,
                        boxprops=dict(color=boxplot_color, alpha=boxplot_alpha, linewidth=boxplot_linewidth),
                        whiskerprops=dict(color=boxplot_color, alpha=boxplot_alpha, linewidth=boxplot_linewidth),
                        capprops=dict(color=boxplot_color, alpha=boxplot_alpha, linewidth=boxplot_linewidth),
                        medianprops=dict(color=boxplot_color, alpha=boxplot_alpha, linewidth=boxplot_linewidth * 2),
                    )
                else:
                    plt.scatter(df_temp['movedir'], df_temp[feature], facecolors=my_pal['brown'], s=5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path / 'fig_identfied_vs_ref.svg')
