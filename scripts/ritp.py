from config import read_config
from optimization.ritp_process import RITPOptimization
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ritp')
    parser.add_argument('--map_name', type=str, default='map414diffuser')
    parser.add_argument('--merged_dataset', type=str, default='2025-03-08_14-54-06_merged_3599')
    parser.add_argument('--planner_name', type=str, default='ritp')
    parser.add_argument('--is_parallel', type=int, default=0)
    args = parser.parse_args()

    config = read_config.read_config(config_name='config')

    # ---------------------------------------------------------------- #
    # --------------------------- Instance --------------------------- #
    # ---------------------------------------------------------------- #
    optimizer = RITPOptimization(config, args)
    optimizer.process()

    print("\033[32mRITP dataset completed,\033[0m The program exited successfully ...")
    sys.exit()
