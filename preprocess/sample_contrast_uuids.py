import argparse
import os
import numpy as np

import pandas as pd

from preprocess.preprocess import data_loader
from corruptions.entity.bern_entities import clean_uuid


DATA_DIR = os.path.expanduser('~/data_tmp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Record the UUIDs used for the contrastive learning training subset.')

    parser.add_argument('--dataset', default='chemistry', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--max_train_examples', default=50000, type=int)

    args = parser.parse_args()

    dataset = data_loader(name=args.dataset)
    out_fn = os.path.join(DATA_DIR, args.dataset, 'contrast_uuids.csv')

    np.random.seed(1992)
    train_uuids = dataset['train']['uuid']
    val_uuids = dataset['validation']['uuid']
    test_uuids = dataset['test']['uuid']

    if args.dataset == 'chemistry':
        # Only the ones for which we have returned BERN entities
        names = open(os.path.expanduser('~/tmp.txt'), 'r').readlines()
        clean_uuids = []
        for line in names:
            line = line.strip()
            if len(line) == 0:
                continue
            clean_uuids.append(line.split(' ')[1].replace('.csv;', ''))
        clean_uuids = set(clean_uuids)
        prev_n = len(train_uuids)
        train_uuids = [x for x in train_uuids if clean_uuid(x) in clean_uuids]
        n = len(train_uuids)
        print(f'{n}/{prev_n} processed by BERN2')

    keep_train_uuids = train_uuids
    if args.max_train_examples < len(train_uuids):
        sampled_train_uuid = np.arange(len(train_uuids))
        np.random.shuffle(sampled_train_uuid)
        keep_train_uuid_idxs = set(sampled_train_uuid[:args.max_train_examples])
        keep_train_uuids = [train_uuids[i] for i in keep_train_uuid_idxs]

    outputs = []
    for uuids, split in [(keep_train_uuids, 'train'), (val_uuids, 'validation'), (test_uuids, 'test')]:
        for uuid in uuids:
            outputs.append({'uuid': uuid, 'split': split})

    outputs = pd.DataFrame(outputs)
    print(f'Saving {len(outputs)} UUIDs to {out_fn}')
    outputs.to_csv(out_fn, index=False)
