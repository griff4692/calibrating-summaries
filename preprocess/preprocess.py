from collections import defaultdict
import ujson
import os
import regex as re

import argparse
from datasets import load_dataset
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import swifter
from transformers import AutoTokenizer, T5Tokenizer
from tqdm import tqdm


T5_MODEL = 'google/long-t5-tglobal-base'
PRIMERA_MODEL = 'allenai/PRIMERA'
DATA_DIR = os.path.expanduser('~/data_tmp')
PUBMED_PATH = 'ccdv/pubmed-summarization'


def linearize_mimic(text):
    # HTML_REGEX_NO_SPACE = r'(<[a-z][^>]+>|<\/?[a-z]>)'
    "https://github.com/amazon-research/summary-reference-revision/src/comp_med_dsum_eval/ref_reviser/dataset.py"
    def get_attr(tag, attr):
        return re.search(attr + r'=([^ ]+)', tag).group(1).strip('<>: ')
    inputs = text.split('<SEP>')
    output_str = ''
    for i, input in enumerate(inputs):
        if input.startswith('<h'):
            output_str += '\n' + get_attr(input, 'raw').replace('_', ' ').strip() + ':\n'
        elif input.startswith('<s'):
            output_str += inputs[i + 1].strip() + ' '
    return output_str.strip()


def load_chemistry(contrast_subsample=False):
    data_fn = os.path.join(DATA_DIR, 'chemistry', 'processed_docs.json')
    print(f'Opening processed data from {data_fn}')
    with open(data_fn, 'r') as fd:
        dataset = ujson.load(fd)

        splits = defaultdict(list)
        for example in tqdm(dataset):
            splits[example['split']].append(process(example))

        train_df = pd.DataFrame(splits['train'])
        if contrast_subsample:
            uuid_fn = os.path.join(DATA_DIR, 'chemistry', 'contrast_uuids.csv')
            uuids = pd.read_csv(uuid_fn)
            uuids_to_keep = set(uuids[uuids['split'] == 'train']['uuid'])
            train_df = train_df[train_df['uuid'].isin(uuids_to_keep)]

        validation_df = pd.DataFrame(splits['validation'])
        test_df = pd.DataFrame(splits['test'])
        train_dataset = Dataset.from_dict(train_df, split='train')
        validation_dataset = Dataset.from_dict(validation_df, split='validation')
        test_dataset = Dataset.from_dict(test_df, split='test')

        print(f'{len(train_dataset)} / {len(validation_dataset)} / {len(test_dataset)} Train / Val / Test splits')

        dataset = DatasetDict(
            {'train': train_dataset, 'test': test_dataset, 'validation': validation_dataset}
        )
        return dataset


def load_clinical(contrast_subsample=False):
    data_fn = os.path.join(DATA_DIR, 'clinical', 'mimic_clinsum.csv')
    if os.path.exists(data_fn):
        print(f'Reading in dataset file from {data_fn}')
        data = pd.read_csv(data_fn).dropna()
        assert 'uuid' in data.columns
    else:
        raw_data_fn = os.path.join(DATA_DIR, 'clinical', 'summary_dataset_rouge_annotated.csv')
        print(f'Reading in dataset file from {raw_data_fn}')
        data = pd.read_csv(raw_data_fn).dropna()
        split = ['validation'] * 1000 + ['test'] * 2000 + ['train'] * (len(data) - 3000)
        np.random.seed(1992)
        np.random.shuffle(split)
        data['split'] = split
        data = data.rename(columns={
            'example_id': 'uuid',
            'source': 'input',
        })

        data['input'] = data['input'].swifter.apply(linearize_mimic)
        data['target'] = data['target'].swifter.apply(linearize_mimic)
        data.dropna().to_csv(data_fn, index=False)
    train_df = data[data['split'] == 'train']
    validation_df = data[data['split'] == 'validation']
    test_df = data[data['split'] == 'test']

    if contrast_subsample:
        print('Skipping. There is no contrast sub-sample because dataset is small.')

    train_dataset = Dataset.from_dict(train_df, split='train')
    validation_dataset = Dataset.from_dict(validation_df, split='validation')
    test_dataset = Dataset.from_dict(test_df, split='test')

    print(f'{len(train_dataset)} / {len(validation_dataset)} / {len(test_dataset)} Train / Val / Test splits')

    dataset = DatasetDict(
        {'train': train_dataset, 'test': test_dataset, 'validation': validation_dataset}
    )
    return dataset


def load_pubmed(contrast_subsample=False):
    dataset = load_dataset(PUBMED_PATH)
    dataset = dataset.rename_columns({
        'article': 'input',
        'abstract': 'target'
    })

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].add_column('uuid', [f'{split}_{i}' for i in range(len(dataset[split]))])
        if contrast_subsample and split == 'train':
            uuid_fn = os.path.join(DATA_DIR, 'pubmed', 'contrast_uuids.csv')
            uuids = pd.read_csv(uuid_fn)
            uuids_to_keep = set(uuids[uuids['split'] == 'train']['uuid'])
            dataset_idxs_to_keep = [i for i, uuid in enumerate(dataset[split]['uuid']) if uuid in uuids_to_keep]
            sub_n = len(dataset_idxs_to_keep)
            n = len(dataset[split])
            print(f'Returning {sub_n}/{n} of the train set reserved for CL')
            dataset[split] = dataset[split].select(dataset_idxs_to_keep)
    return dataset


def data_loader(name, contrast_subsample=False):
    if name == 'pubmed':
        return load_pubmed(contrast_subsample=contrast_subsample)
    elif name == 'chemistry':
        return load_chemistry(contrast_subsample=contrast_subsample)
    elif name == 'clinical':
        return load_clinical(contrast_subsample=contrast_subsample)
    else:
        raise Exception('Not implemented Yet!')


def linearize_sections(sections):
    out_str = ''
    for section in sections:
        if section['header'] is not None:
            out_str += section['header'] + '\n\n'
        paragraphs = [x.strip() for x in re.split('</?p>', section['body']) if len(x.strip()) > 0]
        out_str += '\n\n'.join(paragraphs)
        out_str += '\n\n'
    return out_str.strip()


def process(example):
    return {
        'uuid': example['uuid'],
        'fp': example['fp'],
        'fn': example['fn'],
        'input': linearize_sections(example['sections']),
        'target': example['abstract'],
    }


def process_mimic(example):
    return {
        'uuid': example['example_id'],
        'input': linearize_mimic(example['source']),
        'target': linearize_mimic(example['target']),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenize with T5 / PRIMERA and split into train-validation-test')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--max_input_length', default=4096, type=int)
    parser.add_argument('--max_target_length', default=1024, type=int)
    parser.add_argument('--model', default='primera', choices=['t5', 'primera'])
    parser.add_argument('-debug', action='store_true', default=False)

    args = parser.parse_args()

    if args.model == 't5':
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
    elif args.model == 'primera':
        args.max_input_length = min(4096, args.max_input_length)
        tokenizer = AutoTokenizer.from_pretrained(PRIMERA_MODEL)
    else:
        raise Exception(f'Unrecognized model -> {args.model}')
    
    print(f'Input length max is {args.max_input_length}')
    out_dir = os.path.join(DATA_DIR, args.dataset, f'{args.model}_splits')
    print(f'Output directory is {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == 'chemistry':
        dataset = load_chemistry()
    elif args.dataset == 'clinical':
        dataset = load_clinical()
    elif args.dataset == 'pubmed':
        dataset = load_pubmed()
    else:
        raise Exception('Not Done Yet!')

    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['target']
        model_inputs = tokenizer(
            inputs, add_special_tokens=True, max_length=args.max_input_length, padding=False, truncation=True
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, add_special_tokens=True, max_length=args.max_target_length, padding=False, truncation=True
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=['input', 'target'],
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )
    if not args.debug:
        print(f'Saving to {out_dir}')
        dataset.save_to_disk(out_dir)
