from glob import glob
import os
import ujson

import argparse
from tqdm import tqdm


def move_invalid(args):
    contrast_dir = os.path.join(args.data_dir, args.dataset, 'corruptions')
    # Filter for available uuids
    train_pattern = os.path.join(contrast_dir, 'train', '*.json')
    val_pattern = os.path.join(contrast_dir, 'validation', '*.json')
    train_fns = list(glob(train_pattern))
    val_fns = list(glob(val_pattern))

    train_bad = os.path.join(contrast_dir, 'train_invalid')
    val_bad = os.path.join(contrast_dir, 'validation_invalid')
    os.makedirs(train_bad, exist_ok=True)
    os.makedirs(val_bad, exist_ok=True)

    no_para = 0
    few_neg = 0
    for fn in tqdm(train_fns + val_fns):
        with open(fn, 'r') as fd:
            obj = ujson.load(fd)
            num_pos = len([x for x in obj if x['sign'] == 'positive' and x['method'] != 'reference'])
            num_ref = len([x for x in obj if x['method'] == 'reference'])
            num_neg = len([x for x in obj if x['sign'] == 'negative'])

            move = False
            if num_pos == 0:
                move = True
                no_para += 1
            elif num_ref == 0:
                raise Exception('No reference.')
            elif num_neg < 2:
                move = True
                few_neg += 1
            if move:
                last = fn.split('/')[-1]
                new_fn = '/' + os.path.join(os.path.join(*fn.split('/')[:-1]) + '_invalid', last)
                os.rename(fn, new_fn)

    print('Insufficient negatives: ', few_neg)
    print('No paraphrases: ', no_para)
    print(f'Num moved: ', no_para + few_neg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Arguments to validate each contrast candidate set to ensure > 1 positive and negative'
    )
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])

    args = parser.parse_args()

    move_invalid(args)
