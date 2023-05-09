from collections import Counter
import os

import stanza
import argparse
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize

from preprocess.preprocess import data_loader
from corruptions.entity.bern_entities import clean_uuid, is_incomplete


def extract_entities_for_paper(record, stanza_nlp, max_input_sents=100):
    all_ents = []

    target_ents = extract_stanza_entities(record['target'], stanza_nlp)
    for ent in target_ents:
        ent['source'] = 'target'
        all_ents.append(ent)

    input_sents = sent_tokenize(record['input'])
    num_sent = len(input_sents)
    input_sents_trunc = input_sents[:min(num_sent, max_input_sents)]
    input_trunc = ' '.join(input_sents_trunc)
    input_ents = extract_stanza_entities(input_trunc, stanza_nlp)
    for ent in input_ents:
        ent['source'] = 'input'
        all_ents.append(ent)

    return pd.DataFrame(all_ents)


def process(record, out_dir, stanza_nlp):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    entity_df = extract_entities_for_paper(record, stanza_nlp)
    if entity_df is None:
        print(f'Issue parsing entities for {uuid} to {out_fn}')
        return 'error'
    else:
        print(f'Saving {len(entity_df)} entities for {uuid} to {out_fn}')
        entity_df.drop_duplicates().reset_index(drop=True).to_csv(out_fn, index=False)
    return 'good'


def is_incomplete(record, out_dir):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    return not os.path.exists(out_fn)


def extract_stanza_entities(text, stanza_nlp):
    doc = stanza_nlp(text)
    # print out all entities
    out = []
    for ent in doc.entities:
        out.append({
            'text': ent.text,
            'category': ent.category if hasattr(ent, 'category') else ent.type,
        })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process extract entities')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical'])
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.dataset, 'entity')
    os.makedirs(out_dir, exist_ok=True)

    data = data_loader(args.dataset, contrast_subsample=True)

    if args.dataset == 'pubmed':
        stanza_nlp = stanza.Pipeline(
            'en', package='craft', processors={'ner': 'BioNLP13CG'}, use_gpu=True
        )
    else:
        stanza_nlp = stanza.Pipeline(
            'en', package='mimic', processors={'ner': 'i2b2'}, use_gpu=True
        )

    for split, split_data in data.items():
        prev_n = len(split_data)
        if not args.overwrite:
            print('Filtering out already done examples...')
            split_data = list(filter(lambda x: is_incomplete(x, out_dir), split_data))
        n = len(split_data)
        print(f'Processing {n}/{prev_n} incomplete records')
        statuses = list(tqdm(map(
            lambda record: process(record, out_dir, stanza_nlp), split_data
        ), total=n))
        print(Counter(statuses).most_common())
