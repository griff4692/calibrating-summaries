import os
import regex as re

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import ujson

from preprocess.preprocess import data_loader
from eval.run import METRIC_COLS
from corruptions.validate import move_invalid


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


np.random.seed(1992)  # For reproducibility
REMOVE_COLS = ['abstract', 'masked_input', 'input', 'source', 'target']
ENSURE_COLS = ['uuid', 'prediction'] + METRIC_COLS
REMOVE_UUIDS = {
    'pubmed': {'validation_1105', 'validation_2180'},
    'clinical': {},
    'chemistry': {}
}


def load_corruption(fn, method, sign='positive', strict=False):
    print(f'Loading {method} corruptions from {fn}')
    df = pd.read_csv(fn)
    for col in ENSURE_COLS:
        try:
            assert col in df.columns
        except:
            print(f'\nWARNING: {col} not in {fn}\n')
            if strict:
                exit(1)

    # keep_cols = [col for col in df.columns.tolist() if col not in REMOVE_COLS]
    removed_cols = [col for col in df.columns.tolist() if col in REMOVE_COLS]
    removed_str = ', '.join(removed_cols)
    print(f'Removing {len(removed_cols)} columns: {removed_str}')
    df['method'] = method
    df['sign'] = sign
    print(f'Loaded {len(df)} {method} corruptions from {fn}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Combine All Corruptions into a Single DataFrame')
    
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--diverse_experiments', default='primera_ft_{},long_t5_ft_{}')
    parser.add_argument('-diverse_decoding', default=False, action='store_true')
    parser.add_argument('-paraphrase', default=False, action='store_true')
    parser.add_argument('-mask_and_fill', default=False, action='store_true')
    parser.add_argument('-reference', default=False, action='store_true')
    parser.add_argument('-entity_swap', default=False, action='store_true')
    parser.add_argument('-all', default=False, action='store_true')
    parser.add_argument('-strict', default=False, action='store_true')

    args = parser.parse_args()

    dataset = data_loader(args.dataset, contrast_subsample=True)

    train_uuids = dataset['train']['uuid']
    train_references = dataset['train']['target']
    validation_uuids = dataset['validation']['uuid']
    validation_references = dataset['validation']['target']
    num_train = len(train_uuids)
    num_val = len(validation_uuids)
    uuid2split = {k: 'train' for k in train_uuids}
    for k in validation_uuids:
        assert k not in uuid2split
        uuid2split[k] = 'validation'

    corrupt_dfs = []

    mask_and_fill_fn = os.path.join(args.data_dir, args.dataset, 'mask_and_fill', 'span_fills_with_metrics.csv')
    intrinsic_swap_fn = os.path.join(args.data_dir, args.dataset, 'intrinsic_swaps_with_metrics.csv')
    extrinsic_swap_fn = os.path.join(args.data_dir, args.dataset, 'extrinsic_swaps_with_metrics.csv')
    paraphrase_fn = os.path.join(args.data_dir, args.dataset, 'paraphrase', 'predictions_with_metrics.csv')
    reference_fn = os.path.join(args.data_dir, args.dataset, 'references_with_metrics.csv')

    num_methods = 0

    if args.reference or args.all:
        corrupt_dfs.append(load_corruption(reference_fn, 'reference', sign='positive', strict=args.strict))
        num_methods += 1

    if args.mask_and_fill or args.all:
        corrupt_dfs.append(load_corruption(mask_and_fill_fn, 'mask_and_fill', sign='negative', strict=args.strict))
        num_methods += 1

    if args.diverse_decoding or args.all:
        for de in args.diverse_experiments.split(','):
            ded = de.format(args.dataset)
            diverse_decoding_fn = os.path.join(
                args.data_dir, 'weights', ded, 'results', 'diverse_decoding_train',
                'train_predictions_with_metrics.csv'
            )
            train_corruption = load_corruption(
                diverse_decoding_fn, 'diverse_decoding_' + ded, sign='mixed', strict=args.strict
            )
            diverse_decoding_fn = os.path.join(
                args.data_dir, 'weights', ded, 'results', 'diverse_decoding_validation',
                'validation_predictions_with_metrics.csv'
            )
            validation_corruption = load_corruption(
                diverse_decoding_fn, 'diverse_decoding_' + ded, sign='mixed', strict=args.strict
            )
            full_corruption = pd.concat([train_corruption, validation_corruption])

            # Backwards compatibility with diverse_decode_debug script (Remove these lines if/when we re-run)
            assert 'uuid_fixed' not in full_corruption.columns
            corrupt_dfs.append(full_corruption)
            num_methods += 1

    if args.entity_swap or args.all:
        corrupt_dfs.append(load_corruption(extrinsic_swap_fn, 'extrinsic_swap', sign='negative', strict=args.strict))
        corrupt_dfs.append(load_corruption(intrinsic_swap_fn, 'intrinsic_swap', sign='negative', strict=args.strict))
        num_methods += 2

    if args.paraphrase or args.all:
        corrupt_dfs.append(load_corruption(paraphrase_fn, 'paraphrase', sign='positive', strict=args.strict))
        num_methods += 1

    shared_uuids = set(corrupt_dfs[0]['uuid'])
    print(
        'Searching for common examples among methods '
        '(i.e., some might not have returned answers for each example or be done.'
    )
    for idx in range(1, len(corrupt_dfs)):
        shared_uuids = shared_uuids.intersection(set(corrupt_dfs[idx]['uuid']))

    print(f'Common examples among methods -> {len(shared_uuids)}')
    corrupt_df = pd.concat(corrupt_dfs)
    print(f'Combined {len(corrupt_df)} corruptions for {len(corrupt_df.uuid.unique())} examples')
    print('Filtering for shared UUIDs among methods...')
    corrupt_df = corrupt_df[corrupt_df['uuid'].isin(shared_uuids)]
    corrupt_df = corrupt_df[~corrupt_df['uuid'].isin(REMOVE_UUIDS[args.dataset])]
    print(f'Left with {len(corrupt_df)}')
    corrupt_df = corrupt_df.drop_duplicates(
        subset=['prediction'], keep='first'
    ).sort_values(by='uuid').reset_index(drop=True)
    print(
        f'After removing duplicates, combined {len(corrupt_df)} corruptions for '
        f'{len(corrupt_df.uuid.unique())} examples ({len(corrupt_df)/len(corrupt_df.uuid.unique())})'
    )

    out_fn = os.path.join(args.data_dir, args.dataset, 'corruptions.csv')
    corrupt_df = corrupt_df.assign(split=corrupt_df['uuid'].apply(lambda uuid: uuid2split[uuid]))

    # Group By UUID
    out_dir = os.path.join(args.data_dir, args.dataset, 'corruptions')
    print(f'Grouping by UUID and saving to individual json files in {out_dir}')
    uuid2corruptions = corrupt_df.groupby('uuid')
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, 'train')
    validation_dir = os.path.join(out_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    for uuid, corruptions in tqdm(uuid2corruptions, total=len(uuid2corruptions)):
        corruptions = corruptions.sort_values(by='method').reset_index(drop=True)
        records = corruptions.to_dict('records')
        split = records[0]['split']
        assert all([x['split'] == split for x in records])
        if args.dataset == 'chemistry':
            out_fn = os.path.join(out_dir, split, f'{clean_uuid(uuid)}.json')
        else:
            out_fn = os.path.join(out_dir, split, f'{uuid}.json')
        with open(out_fn, 'w') as fd:
            ujson.dump(records, fd)

    move_invalid(args)
