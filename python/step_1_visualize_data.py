import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from python.utils.misc import my_pal, hex_to_rgba


def linear_plot(df, ftr):
    fig = go.Figure(
        layout=go.Layout(
            template='plotly_white',
            width=900,
            height=500,
            xaxis=dict(title='Direction, deg'),
            yaxis=dict(title=ftr),
        )
    )
    for surf, _df in df.groupby('surf'):
        fig.add_trace(
            go.Scatter(
                x=_df['movedir'],
                y=_df['mean'],
                mode='lines',
                line=dict(color=my_pal[surf], width=2, dash='dash'),
                name=surf,
                legendgroup=surf,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=_df['movedir'],
                y=_df['p25'],
                mode='lines',
                line=dict(color=my_pal[surf], width=1),
                fillcolor=hex_to_rgba(my_pal[surf], alpha=0.2),
                name=surf,
                legendgroup=surf,
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=_df['movedir'],
                y=_df['p75'],
                mode='lines',
                fill='tonexty',
                line=dict(color=my_pal[surf], width=1),
                fillcolor=hex_to_rgba(my_pal[surf], alpha=0.2),
                name=surf,
                legendgroup=surf,
                showlegend=False,
            )
        )
    return fig


def polar_plot(df, ftr):
    fig = go.Figure(
        layout=go.Layout(
            template='plotly_white',
            width=500,
            height=500,
            polar=dict(
                angularaxis=dict(
                    direction='counterclockwise',
                    rotation=90,
                ),
                radialaxis=dict(
                    title=ftr
                )
            )
        )
    )
    for surf, _df in df.groupby('surf'):
        fig.add_trace(
            go.Scatterpolar(
                r=_df['mean'],
                theta=_df['movedir'],
                mode='lines',
                line=dict(color=my_pal[surf], width=2, dash='dash'),
                name=surf,
                legendgroup=surf,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=_df['p25'],
                theta=_df['movedir'],
                mode='lines',
                line=dict(color=my_pal[surf], width=1),
                fillcolor=hex_to_rgba(my_pal[surf], alpha=0.2),
                name=surf,
                legendgroup=surf,
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=_df['p75'],
                theta=_df['movedir'],
                mode='lines',
                fill='tonext',
                line=dict(color=my_pal[surf], width=1),
                fillcolor=hex_to_rgba(my_pal[surf], alpha=0.2),
                name=surf,
                legendgroup=surf,
                showlegend=False,
            )
        )
    return fig


if __name__ == '__main__':
    prepared_data = Path('data/prepared')
    figures = Path('figures/base_figs')

    df = pd.read_csv(prepared_data / 'linear.csv')

    features = ['Ke1', 'Ke1_kalman']

    for ftr in features:
        res = df.groupby(['surf', 'movedir']).agg(
            mean=(ftr, 'mean'),
            p25=(ftr, lambda x: x.quantile(0.25)),
            p75=(ftr, lambda x: x.quantile(0.75))
        ).reset_index()

        # linear plot
        fig = linear_plot(res, ftr)

        export_path = figures / ftr
        export_path.mkdir(parents=True, exist_ok=True)

        fig.write_image(export_path / f'{ftr}_linear.svg')
        fig.write_html(export_path / f'{ftr}_linear.html', include_plotlyjs='cdn')

        # radial plot
        fig = polar_plot(res, ftr)

        export_path = figures / ftr
        export_path.mkdir(parents=True, exist_ok=True)

        fig.write_image(export_path / f'{ftr}_radial.svg')
        fig.write_html(export_path / f'{ftr}_radial.html', include_plotlyjs='cdn')


