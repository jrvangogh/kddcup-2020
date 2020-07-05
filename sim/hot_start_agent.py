from model.agent import Agent
from sim.simulator import Simulator
import sys

from datetime import datetime


DATES = [
    '20161101',
    '20161102',
    '20161103',
    '20161104',
    '20161105',
    '20161106',
    '20161107',
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
    agent = Agent(load_state_model=False)
    use_small = len(sys.argv) > 1 and sys.argv[1].startswith('s')
    ds_list = DATES_SMALL if use_small else DATES
    for ds in ds_list:
        print(f'Running {ds}')
        s = Simulator(agent, ds)
        s.run()
    if use_small:
        output_file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_small_agent.pickle'
    else:
        output_file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_full_agent.pickle'
    print(f'Saving to {output_file_name}')
    agent.save_state_model(output_file_name)


if __name__ == '__main__':
    main()
