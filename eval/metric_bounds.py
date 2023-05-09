import os
import ujson

import argparse
import pandas as pd

from eval.run import METRIC_COLS


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Summary Metric Statistics for Corruptions')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--datasets', default='clinical,chemistry,pubmed')
    parser.add_argument('--experiment', default='primera_ft_{}')

    args = parser.parse_args()

    out_df = []
    for dataset in args.datasets.split(','):
        results_fn = os.path.join(
            args.data_dir, 'weights', args.experiment.format(dataset), 'results', 'predictions_with_metrics.csv'
        )
        stats = {}
        results_df = pd.read_csv(results_fn)
        for col in METRIC_COLS:
            stats[col] = {'mean': results_df[col].mean(), 'std': results_df[col].std()}

        out_fn = os.path.join(args.data_dir, f'{dataset}_metric_bounds.json')
        print(f'Saving statistics to {out_fn}')
        with open(out_fn, 'w') as fd:
            ujson.dump(stats, fd)

        for k, v in stats.items():
            print(str(k) + ' ' + str(v['mean']) + ' ' + str(v['std']))
