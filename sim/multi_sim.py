from model.agent import Agent
from sim.simulator import Simulator
from multiprocessing import Pool
import json
import numpy as np
from datetime import datetime
import sys


DATES = [
    '20161124',
    '20161125',
    '20161126',
    '20161127',
    '20161128',
    '20161129',
    '20161130',
]


DATES_SMALL = [
    '20161124_small',
    '20161125_small',
    '20161126_small',
    '20161127_small',
    '20161128_small',
    '20161129_small',
    '20161130_small',
]


def get_metrics(ds: str):
    s = Simulator(Agent(), ds, disable_progress_bar=False)
    return s.run()


def main():
    use_small = len(sys.argv) > 1 and sys.argv[1].startswith('s')
    pool = Pool(processes=7)
    if use_small:
        print('Using small versions of orders and drivers')
        metrics = pool.map(get_metrics, DATES_SMALL)
    else:
        print('Using normal versions of orders and drivers')
        metrics = pool.map(get_metrics, DATES)
    avg_dict = {}
    for k in metrics[0].keys():
        avg_dict[k] = np.mean([m[k] for m in metrics])
    full_output = {
        'averages': avg_dict,
        'individual_days': metrics
    }
    if use_small:
        output_file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_small.json'
    else:
        output_file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_output.json'
    print(f'Saving to {output_file_name}')
    with open(output_file_name, 'wt') as f:
        json.dump(full_output, f, indent=4)


if __name__ == '__main__':
    main()
