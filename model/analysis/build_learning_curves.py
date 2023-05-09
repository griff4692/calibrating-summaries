import os

import argparse
import pandas as pd


from eval.run import METRIC_COLS


DATA_DIR = os.path.expanduser('~/data_tmp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collect results for learning curves')

    parser.add_argument('--experiment', default='primera_fft_clinical_relevance_lc_low_scale')  # WandB name
    parser.add_argument('--dataset', default='clinical')

    args = parser.parse_args()

    weight_dir = os.path.join(DATA_DIR, 'weights', args.experiment)
    ft_fn = os.path.join(DATA_DIR, 'weights', f'primera_ft_{args.dataset}', 'results', 'predictions_with_metrics.csv')
    ft_df = pd.read_csv(ft_fn)

    stats = {}
    for col in METRIC_COLS:
        col_mean = ft_df[col].mean()
        col_std = ft_df[col].std()
        stats[col] = (col_mean, col_std)

    steps = [
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
    ]

    rel_scores = []
    faith_scores = []
    for step in steps:
        print(step)
        fn = os.path.join(weight_dir, f'ckpt_{step}_steps', 'predictions_with_metrics.csv')
        df = pd.read_csv(fn)

        # Faithful metrics
        faith_metrics = ['bs_src_f1', 'fact_score', 'bart_score']
        rel_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']

        faith_score_arr = []
        for metric in faith_metrics:
            norm_score = (df[metric].mean() - stats[metric][0]) / stats[metric][1]
            faith_score_arr.append(norm_score)
        faith_scores.append(sum(faith_score_arr) / len(faith_score_arr))
        rel_score_arr = []
        for metric in rel_metrics:
            print(metric, df[metric].mean())
            norm_score = (df[metric].mean() - stats[metric][0]) / stats[metric][1]
            rel_score_arr.append(norm_score)
        rel_scores.append(sum(rel_score_arr) / len(rel_score_arr))

    print(args.experiment)
    print('Faith')
    print(faith_scores)
    print('Relevance')
    print(rel_scores)
