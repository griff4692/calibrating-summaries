import os
from glob import glob
import pandas as pd
import ujson
import regex as re

import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate Select and Save best Validation Checkpoint')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--experiment', default='primera_fft_clinical_relevance_lc_low_scale')
    parser.add_argument('--metric', default='relevance')

    args = parser.parse_args()
    pattern = os.path.join(args.data_dir, 'weights', args.experiment, '*', 'validation_predictions_with_metrics.csv')

    print(f'Looking for outputs matching {pattern}')
    fns = list(glob(pattern))

    metric_norm_fn = os.path.join(args.data_dir, f'{args.dataset}_metric_bounds.json')
    with open(metric_norm_fn, 'r') as fd:
        stats = ujson.load(fd)

    agg_scores = []
    for fn in fns:
        results = pd.read_csv(fn)
        if args.metric == 'faithful':
            contrast_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
        elif args.metric == 'relevance':
            contrast_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']
        else:
            raise Exception('Nope!')

        def score_candidate_fn(row):
            norm_vals = []
            for metric in contrast_metrics:
                stat = stats[metric]
                norm_vals.append((row[metric] - stat['mean']) / stat['std'])
            return sum(norm_vals) / len(norm_vals)

        score_metric = np.mean(
            list(map(score_candidate_fn, results.to_dict('records')))
        )
        agg_scores.append(score_metric)

    best_fn = fns[int(np.argmax(agg_scores))]
    out_fn = os.path.join(args.data_dir, 'weights', args.experiment, 'the_chosen_one.txt')
    print(f'Selected {best_fn}')
    best_step = str(int(re.search(r'ckpt_(\d+)_steps', best_fn).group(1)))
    with open(out_fn, 'w') as fd:
        fd.write(best_step)
