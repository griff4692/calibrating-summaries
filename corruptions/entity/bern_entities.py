import requests
import regex as re
import argparse
import os
from tqdm import tqdm
from p_tqdm import p_uimap
import pandas as pd
from collections import Counter

import ujson


LOCAL_HOST_SERVER = 'http://localhost:8888/plain'
API_HOST = 'http://bern2.korea.ac.kr/plain'


def get_paragraphs(sections):
    paragraphs = []
    for section in sections:
        paragraphs.extend([x.strip() for x in re.split('</?p>', section['body']) if len(x.strip()) > 0])
    return paragraphs


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


def query_plain(text, url=LOCAL_HOST_SERVER):
    response = requests.post(url, json={'text': text}).json()
    if 'annotations' not in response:
        return None
    out = []
    for obj in response['annotations']:
        out.append({
            'id': '|'.join(obj['id']),
            'text': obj['mention'],
            'start': obj['span']['begin'],
            'end': obj['span']['end'],
            'category': obj['obj']
        })
    return out


def extract_entities_for_paper(record):
    abstract = record['abstract']

    all_ents = []
    abstract_ents = query_plain(abstract)
    if abstract_ents is None:
        return None
    for ent in abstract_ents:
        ent['source'] = 'abstract'
        all_ents.append(ent)
    paragraphs = get_paragraphs(record['sections'])
    raw_text = '\n'.join(paragraphs)
    body_ents = query_plain(raw_text)
    if body_ents is None:
        return pd.DataFrame(all_ents)
    for ent in body_ents:
        ent['source'] = 'paragraph'
        all_ents.append(ent)
    return pd.DataFrame(all_ents)


def process(record, out_dir):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    entity_df = extract_entities_for_paper(record)
    if entity_df is None:
        print(f'Issue parsing entities for {uuid} to {out_fn}')
        return 'error'
    else:
        print(f'Saving {len(entity_df)} entities for {uuid} to {out_fn}')
        entity_df.to_csv(out_fn, index=False)
    return 'good'


def is_incomplete(record, out_dir):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    return not os.path.exists(out_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process extract entities')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--num_cpus', default=1, type=int)

    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, 'entity')
    os.makedirs(out_dir, exist_ok=True)

    data_fn = os.path.join(args.data_dir, 'processed_docs.json')
    print(f'Loading dataset from {data_fn}')

    with open(data_fn, 'r') as fd:
        data = ujson.load(fd)
        prev_n = len(data)
        if not args.overwrite:
            print('Filtering out already done examples...')
            data = list(filter(lambda x: is_incomplete(x, out_dir), data))
        n = len(data)
        print(f'Processing {n}/{prev_n} incomplete records')
        if args.num_cpus > 1:
            statuses = list(p_uimap(lambda record: process(record, out_dir), data, num_cpus=args.num_cpus))
        else:
            statuses = list(tqdm(map(lambda record: process(record, out_dir), data), total=n))
        print(Counter(statuses).most_common())
