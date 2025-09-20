from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from python.utils.kalman_filter import KalmanFilter


def calc_motor_voltages(df, suffix='') -> None:
    # motor consts
    C_e = 0.0501
    R_a = 5.3351

    for idx in range(1, 4):
        df[f'm{idx}vol{suffix}'] = (
                df[f'm{idx}cur{suffix}'] * R_a * np.sign(df[f'm{idx}vel{suffix}']) + df[f'm{idx}vel{suffix}'] * C_e)


def calc_rpower(df, name='rpower', suffix='') -> None:
    df[name] = 0
    for idx in range(1, 4):
        df[name] += df[f'm{idx}vol{suffix}'].abs() * df[f'm{idx}cur{suffix}'].abs()


def calc_ke1(df, name='Ke1', suffix='') -> None:
    nominal_current = 0.17
    df[name] = df[f'rpower{suffix}'] / (
            df[[f'm{idx}vol{suffix}' for idx in range(1, 4)]].abs().sum(axis=1) * nominal_current)


if __name__ == '__main__':
    raw_data = Path('data/raw')
    prepared_data = Path('data/prepared')

    motor_params = loadmat('data/motor_models.mat')['motor_model'][0]

    #   Raw data. Filter data, find energy coefficient
    for file in raw_data.glob('*.csv'):
        df = pd.read_csv(file)

        for m_idx in range(1, 4):
            KF = KalmanFilter(*motor_params[m_idx - 1])
            df[[f'm{m_idx}cur_kalman', f'm{m_idx}vel_kalman']] = df.apply(
                lambda x: KF.update([x[f'm{m_idx}setvel']], np.array(
                    [x[f'm{m_idx}cur'] * np.sign(x[f'm{m_idx}vel']), x[f'm{m_idx}vel']]))[:2], axis=1,
                result_type='expand')
            df[f'm{m_idx}cur_kalman'] = df[f'm{m_idx}cur_kalman'].abs()

        calc_motor_voltages(df)  # calc usual voltages
        calc_motor_voltages(df, suffix='_kalman')

        calc_rpower(df)  # calc usual rpower
        calc_rpower(df, name='rpower_kalman', suffix='_kalman')

        calc_ke1(df)  # calc usual ke1
        calc_ke1(df, name='Ke1_kalman', suffix='_kalman')

        if 'w1surf' in df.columns:
            df['surf'] = df.apply(lambda row: row['w1surf'] if row['w1surf'] == row['w2surf'] == row[
                'w3surf'] else 'undefined', axis=1)

        df.to_csv(prepared_data / file.name, index=False)

    #   Create all-in-one file for linear motion
    surfs = ['gray', 'green', 'table', 'red']
    big_df = pd.DataFrame()
    for surf in surfs:
        df = pd.read_csv(prepared_data / f'{surf}.csv')
        df['surf'] = surf
        big_df = pd.concat([big_df, df])
    big_df.to_csv(prepared_data / 'linear.csv', index=False)
