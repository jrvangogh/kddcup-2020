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

GAMMAS = [0.90]
UNASSIGNED_PENALTIES = [0.90]
MIN_X = [25, 50, 100, 200]
EXP_DECAY = [0.97]
MAX_DEPTH = [3, 5, 7]
NUM_TREES = [100]
PARAMS = product(GAMMAS, UNASSIGNED_PENALTIES, MIN_X, EXP_DECAY, MAX_DEPTH, NUM_TREES)


def get_metrics(tup):
    i, kwarg_dict = tup
    ds = kwarg_dict['ds']
    gamma = kwarg_dict['gamma']
    unassigned_penalty = kwarg_dict['unassigned_penalty']
    min_x = kwarg_dict['min_x']
    num_trees = kwarg_dict['num_trees']
    max_depth = kwarg_dict['max_depth']
    agent = Agent(
        gamma=gamma,
        unassigned_penalty=unassigned_penalty,
        min_x_len=min_x,
        num_trees=num_trees,
        max_depth=max_depth,
    )
    s = Simulator(agent, ds)
    metrics = s.run()
    metrics.update(kwarg_dict)
    output_file = SAVE_DIR / f'{i:03d}.json'
    with open(output_file, 'wt') as f:
        json.dump(metrics, f, indent=4)


def make_iter_list(date_list: List[str], gamma: float, unassigned_penalty: float,
                   min_x: int, exp_decay: float, max_depth: int, num_trees: int):
    kwarg_dict = [{'ds': ds, 'gamma': gamma, 'unassigned_penalty': unassigned_penalty,
                   'min_x': min_x, 'exp_decay': exp_decay, 'num_trees': num_trees, 'max_depth': max_depth}
                  for ds in date_list]
    return kwarg_dict


def not_done(i):
    output_file = SAVE_DIR / f'{i:03d}.json'
    return not output_file.exists()


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    use_small = len(sys.argv) > 1 and sys.argv[1].startswith('s')
    if use_small:
        print('Using small versions of orders and drivers')
        date_list = DATES_SMALL
    else:
        print('Using normal versions of orders and drivers')
        date_list = DATES

    nested = [make_iter_list(date_list, gamma, unassigned_penalty, min_x, exp_decay, max_depth, num_trees)
              for (gamma, unassigned_penalty, min_x, exp_decay, max_depth, num_trees) in PARAMS]
    flat = sorted(chain.from_iterable(nested), key=lambda d: d['ds'])
    pool = Pool(processes=30)
    enumerated = [t for t in enumerate(flat) if not_done(t[0])]
    pool.map(get_metrics, enumerated)


if __name__ == '__main__':
    main()
