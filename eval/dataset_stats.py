import os

import argparse
from tqdm import tqdm
from p_tqdm import p_uimap
import numpy as np
import pandas as pd

from preprocess.preprocess import data_loader
from eval.run import tokenize


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments compute basic dataset statistics')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical')
    parser.add_argument('--metric', default='faithful')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--split', default='train')

    args = parser.parse_args()

    ref_fn = os.path.join(args.data_dir, args.dataset, 'references_with_metrics.csv')
    print(f'Loading references with metrics from {ref_fn}')
    references = pd.read_csv(ref_fn)

    print(f'Density: {references.density.mean()}')
    print(f'Coverage: {references.coverage.mean()}')
    print(f'Num Reference Tokens: {references.num_prediction_tokens.mean()}')

    full_data = data_loader(args.dataset, contrast_subsample=False)[args.split]
    source_tokens = list(p_uimap(tokenize, full_data['input']))
    source_tok_lens = list(map(len, source_tokens))
    print(f'Num Source Tokens: {np.mean(source_tok_lens)}')
