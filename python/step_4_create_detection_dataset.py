import pickle
from multiprocessing import Manager, Pool
from pathlib import Path

import numpy as np
import pandas as pd

from time import time

# noinspection PyUnresolvedReferences
from python.utils.dct import DCT


def random_partition(win_size, alpha_amount, generator):
    """
    Generates a random partition of `win_size` into `alpha_amount` parts where each part is at least 1.

    Parameters:
        win_size (int): The total sum to partition.
        alpha_amount (int): The number of partitions, each of which is at least 1.

    Returns:
        list: A list of `alpha_amount` integers that sum to `win_size`, with each integer >= 1.
    """
    partitions = []
    remaining = win_size

    for i in range(1, alpha_amount):
        # Ensure there is enough left for each of the remaining categories
        max_val = remaining - (alpha_amount - i)
        next_part = generator.integers(1, max_val / 2 + 1)
        partitions.append(next_part)
        remaining -= next_part

    # The last partition takes the remaining value
    partitions.append(remaining)

    generator.shuffle(partitions)
    return partitions


def compose_string(*args):
    return ','.join((str(x) for x in args)) + '\n'


# Define worker function
def worker_task(args):
    (win_size, dir_amount, repeat_dirs, repeat_rows, directions, surfs, surfs_except, data, model_values, model_std,
     gen_seed, lock, file_path) = args
    gen = np.random.default_rng(gen_seed)
    results = []

    for _ in range(repeat_dirs):
        rows_by_dir = random_partition(win_size, dir_amount, gen)
        dirs = gen.choice(directions, size=dir_amount, replace=False)
        for _ in range(repeat_rows):
            for true_surf in surfs:
                for surf in [true_surf, gen.choice(surfs_except[true_surf])]:
                    dKe = []
                    for dir, amount in zip(dirs, rows_by_dir):
                        rows = gen.choice(data[true_surf][dir], amount, replace=False)
                        delta = abs(rows - model_values[dir][surf])
                        dKe.append(np.mean(delta))
                    alpha = np.average(dirs, weights=rows_by_dir)
                    dKe_avg = np.average(dKe, weights=rows_by_dir)
                    std_surf = model_std[surf]
                    is_new = int(surf != true_surf)
                    results.append(compose_string(alpha, dir_amount, dKe_avg, std_surf, is_new))

    with lock:
        with open(file_path, 'a') as file:
            file.writelines(results)


def main():
    input_path = Path('data/prepared')
    export_path = Path('data/detection/train')
    export_path.mkdir(parents=True, exist_ok=True)

    surfs = ['gray', 'green', 'table']

    surfs_except = {}
    for surf in surfs:
        s_copy = surfs.copy()
        s_copy.remove(surf)

    feature = 'Ke1_kalman'
    directions = range(0, 356, 5)

    data = {surf: pd.read_csv(input_path / f'{surf}_noavg_kalman.csv')[['movedir', feature]] for surf in surfs}
    for surf, df in data.items():
        data[surf] = df.groupby('movedir').nth[10:]
        data[surf] = {_dir: _val[feature].to_numpy() for (_dir, _val) in list(df.groupby('movedir'))}

    with open('data/dct_models.pkl', 'rb') as file:
        dct_by_surf, model_std = pickle.load(file)

    model_values = {dir: {surf: dct_by_surf[surf].numpy_func(dir, scaled=True) for surf in surfs} for dir in directions}

    # Parameters
    repeat_partition = 50
    repeat_dirs = 30
    repeat_rows = 30

    file_path = export_path / 'dataset_v1.csv'

    # Initialize file
    with open(file_path, 'w') as file:
        file.write(compose_string('alpha', 'n_alpha', 'dKe', 'std_surf', 'is_new'))

    # Prepare multiprocessing
    manager = Manager()
    lock = manager.Lock()
    gen_seed = np.random.SeedSequence().generate_state(repeat_partition)

    tasks = []
    for win_size in range(66, 67):
        for dir_amount in range(1, 11):
            for i in range(repeat_partition):
                tasks.append((
                    win_size, dir_amount, repeat_dirs, repeat_rows,
                    directions, surfs, surfs_except, data, model_values, model_std,
                    gen_seed[i], lock, file_path
                ))

    # Execute tasks in parallel
    with Pool(processes=5) as pool:
        pool.map(worker_task, tasks)


if __name__ == "__main__":
    start = time()
    main()
    end = time()

    print(f'Time: {end - start:.4f} s')
