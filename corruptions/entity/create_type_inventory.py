import argparse
import string
from collections import Counter, defaultdict
import itertools
import os
import ujson
from glob import glob

import pandas as pd
from p_tqdm import p_uimap


def read_fn(fn):
    try:
        return pd.read_csv(fn).to_dict('records')
    except pd.errors.EmptyDataError:
        print(f'Empty Dataframe: {fn}')
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process extract entities')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    pattern = os.path.join(args.data_dir, args.dataset, 'entity', '*.csv')
    fns = list(glob(pattern))

    records = list(itertools.chain(*list(filter(None, list(p_uimap(read_fn, fns))))))
    raw_inventory = defaultdict(list)
    for record in records:
        try:
            raw_inventory[record['category']].append(record['text'].strip(string.punctuation).strip())
        except:
            t = record['text']
            print(f'Invalid text -> {t}')

    inventory = {}
    for k, v in raw_inventory.items():
        counts = Counter(v)
        total = len(v)
        k_v = counts.most_common()
        p = [x[1] / total for x in k_v]
        v_sorted = [x[0] for x in k_v]
        inventory[k] = {
            'probability': p,
            'text': v_sorted
        }

    out_fn = os.path.join(args.data_dir, args.dataset, 'entity_inventory.json')
    print(f'Dumping inventory to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(inventory, fd)
