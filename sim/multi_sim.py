from model.agent import Agent
from typing import List
from sim.simulator import Simulator
from multiprocessing import Pool
import json
import sys
from itertools import product, chain
from pathlib import Path


SAVE_DIR = Path(__file__).parent.parent.resolve() / 'sim_results'


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

GAMMAS = [0.80, 0.85, 0.90, 0.95]
UNASSIGNED_PENALTIES = [0.80, 0.85, 0.90, 0.95]
ALPHAS = [0.15, 0.20, 0.25, 0.30, 0.35]
PARAMS = product(GAMMAS, UNASSIGNED_PENALTIES, ALPHAS)


def get_metrics(tup):
    i, kwarg_dict = tup
    ds = kwarg_dict['ds']
    gamma = kwarg_dict['gamma']
    unassigned_penalty = kwarg_dict['unassigned_penalty']
    alpha = kwarg_dict['alpha']
    s = Simulator(Agent(gamma=gamma, unassigned_penalty=unassigned_penalty, alpha=alpha), ds)
    metrics = s.run()
    metrics.update(kwarg_dict)
    output_file = SAVE_DIR / f'{i:03d}.json'
    with open(output_file, 'wt') as f:
        json.dump(metrics, f, indent=4)


def make_iter_list(date_list: List[str], gamma: float, unassigned_penalty: float, alpha: float):
    kwarg_dict = [{'ds': ds, 'gamma': gamma, 'unassigned_penalty': unassigned_penalty, 'alpha': alpha}
                  for ds in date_list]
    return kwarg_dict


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    use_small = len(sys.argv) > 1 and sys.argv[1].startswith('s')
    if use_small:
        print('Using small versions of orders and drivers')
        date_list = DATES_SMALL
    else:
        print('Using normal versions of orders and drivers')
        date_list = DATES

    nested = [make_iter_list(date_list, gamma, unassigned_penalty, alpha)
              for (gamma, unassigned_penalty, alpha) in PARAMS]
    flat = list(chain.from_iterable(nested))
    pool = Pool(processes=len(flat))
    pool.map(get_metrics, [t for t in enumerate(flat)])


if __name__ == '__main__':
    main()
