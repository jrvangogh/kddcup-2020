from model.agent import Agent
from sim.simulator import Simulator
import sys
from pathlib import Path
from time import time

from datetime import datetime


SAVE_DIR = Path(__file__).parent.parent / 'hot_starts'


DATES = [
    '20161101',
    '20161102',
    '20161103',
    '20161104',
    '20161105',
    '20161106',
    '20161107',
    '20161108',
    '20161109',
    '20161110',
    '20161111',
    '20161112',
    '20161113',
    '20161114',
    '20161115',
    '20161116',
    '20161117',
    '20161118',
    '20161119',
    '20161120',
    '20161121',
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


def main():
    start_time = time()

    SAVE_DIR.mkdir(exist_ok=True)
    use_small = len(sys.argv) > 1 and sys.argv[1].startswith('s')
    if use_small:
        output_prefix = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_small'
    else:
        output_prefix = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print(f'Using prefix {output_prefix}')

    agent = Agent(load_state_model=False)
    ds_list = DATES_SMALL if use_small else DATES
    print('Running the following:')
    for ds in ds_list:
        print(f'    {ds}')
    for ds in ds_list:
        print(f'Running {ds}')
        s = Simulator(agent, ds)
        s.run()
        save_path = SAVE_DIR / f'{output_prefix}_{ds}.pickle'
        agent.save_state_model(save_path)

    end_time = time()
    print(f'Duration: {(end_time - start_time) / 3600} hours')


if __name__ == '__main__':
    main()
