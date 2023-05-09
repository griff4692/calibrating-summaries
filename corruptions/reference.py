import argparse
import os
import pandas as pd

from preprocess.preprocess import data_loader


DATA_DIR = os.path.expanduser('~/data_tmp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to extract the reference into its own file.')
    parser.add_argument('--dataset', default='chemistry', choices=['pubmed', 'clinical', 'chemistry'])

    args = parser.parse_args()

    datasets = data_loader(args.dataset, contrast_subsample=True)
    out_fn = os.path.join(DATA_DIR, args.dataset, 'references.csv')

    references = []
    for split, data in datasets.items():
        targets = data['target']
        uuids = data['uuid']
        for target, uuid, in zip(data['target'], data['uuid']):
            references.append({
                'uuid': uuid,
                'prediction': target,
                'split': split,
            })

    references = pd.DataFrame(references)
    print(f'Saving {len(references)} references to {out_fn}')
    references.to_csv(out_fn, index=False)
