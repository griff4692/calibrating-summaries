import os

import argparse
import numpy as np


from preprocess.preprocess import data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Generate Paraphrases of Abstracts')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--mode', default='annotations')
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--num_to_annotate', default=10, type=int)
    parser.add_argument('--max_tokens', default=256, type=int)

    args = parser.parse_args()

    summaries = data_loader(args.dataset, contrast_subsample=False)['validation']['target']
    summaries = [x for x in summaries if len(x.split(' ')) < args.max_tokens]

    para_dir = os.path.join(args.data_dir, args.dataset, 'paraphrase')
    os.makedirs(para_dir, exist_ok=True)
    out_fn = os.path.join(para_dir, 'paraphrase_annotations.txt')

    n = len(summaries)
    idxs = np.arange(n)
    np.random.seed(1992)
    np.random.shuffle(idxs)
    abstracts = [summaries[idx] for idx in idxs[:args.num_to_annotate]]

    out_str = ''
    for abstract in abstracts:
        out_str += 'Abstract:\n'
        out_str += abstract
        out_str += '\n\n'
        out_str += 'Paraphrase:\n'
        out_str += '\n\n'

    print(f'Saving paraphrase annotation data to {out_fn}')
    with open(out_fn, 'w') as fd:
        fd.write(out_str)
