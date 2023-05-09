from p_tqdm import p_uimap
import itertools
from glob import glob
import math

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import regex as re
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from preprocess.preprocess import data_loader


# from corruptions.entity.bern_entities import clean_uuid
def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


HF_MODEL = 'razent/SciFive-base-Pubmed_PMC'
# https://www.dampfkraft.com/penn-treebank-tags.html
# KEEP_TAGS = ['NP']
DATA_DIR = os.path.expanduser('~/data_tmp')


class MaskFiller:
    def __init__(self, device='cuda:0', num_beams=4) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL).to(device).eval()
        self.device = device
        self.mask_pattern = r'<extra_id_\d+>'
        self.max_length = 1024
        self.num_beams = num_beams
    
    def fill(self, texts, target_length):
        encoding = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids, attention_mask = encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
        kwargs = {
            'num_beams': self.num_beams,
            'min_length': min(self.max_length, target_length + 3),
            'max_length': self.max_length
        }
        with torch.no_grad():
            preds = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, **kwargs
            )
        decoded = self.tokenizer.batch_decode(preds)
        batch_filled = []
        for text, pred in zip(texts, decoded):
            snippets = re.split(self.mask_pattern, pred.replace('<pad>', ''))
            assert len(snippets[0]) == 0
            snippets = snippets[1:]
            unmasked_text = text
            for snippet in snippets:
                mask_location = re.search(r'<extra_id_\d+>', unmasked_text)
                if mask_location is not None:
                    unmasked_text = unmasked_text[:mask_location.start()] + snippet + unmasked_text[mask_location.end():]
            unmasked_text = re.sub(r'\s+', ' ', unmasked_text).strip()
            if re.search(self.mask_pattern, unmasked_text) is not None:
                print('Invalid generation. Could not fully unmask.')
                batch_filled.append(None)
            else:
                batch_filled.append(unmasked_text)
        return batch_filled

    def cleanup(self):
        self.model.cpu()


def implement_mask(abstract, noun_chunks_to_mask):
    masked_abstract = abstract
    offset = 0
    num_to_mask = len(noun_chunks_to_mask)
    num_removed_tokens = 0
    for mask_idx in range(num_to_mask):
        placeholder = f'<extra_id_{mask_idx}>'
        span_size = noun_chunks_to_mask[mask_idx].end_char - noun_chunks_to_mask[mask_idx].start_char
        actual_start = noun_chunks_to_mask[mask_idx].start_char + offset
        actual_end = noun_chunks_to_mask[mask_idx].end_char + offset
        masked_abstract = masked_abstract[:actual_start] + placeholder + masked_abstract[actual_end:]
        offset += len(placeholder) - span_size
        num_removed_tokens += len(str(noun_chunks_to_mask[mask_idx]).split(' '))

    return masked_abstract, num_removed_tokens, num_to_mask


def build_masks(record, mask_rates, nlp, max_masks=20):
    outputs = []

    noun_chunks = [x for x in list(nlp(record['target']).noun_chunks) if len(str(x).strip()) > 1]
    n = len(noun_chunks)
    for mask_rate in mask_rates:
        num_to_mask = min(int(math.ceil(mask_rate * n)), max_masks)
        seen_masked = set()
        samples_collected = 0
        for sample_idx in range(args.samples_per_bucket * 10):
            idxs_to_mask = np.random.choice(np.arange(n), size=(num_to_mask), replace=False).tolist()
            noun_chunks_to_mask = [noun_chunks[i] for i in sorted(idxs_to_mask)]
            masked, removed_tokens, num_masks = implement_mask(record['target'], noun_chunks_to_mask)
            if masked in seen_masked:
                # print('Duplicate Mask. Skipping')
                continue
            seen_masked.add(masked)
            outputs.append({
                'uuid': record['uuid'],
                'target_mask_rate': mask_rate,
                'sample_idx': sample_idx,
                'abstract': record['target'],
                'masked_input': masked,
                'removed_tokens': removed_tokens,
                'num_masks': num_masks,
            })
            samples_collected += 1
            if samples_collected == args.samples_per_bucket:
                break

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Extract, Mask, and Fill Syntactic Spans from References')
    parser.add_argument('--dataset', default='chemistry', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument(
        '--mode', default='mask_spans',
        choices=['mask_spans', 'fill_spans', 'merge_chunks']
    )
    # Extract Span Arguments
    parser.add_argument('-overwrite', default=False, action='store_true')
    # Mask Span Arguments
    parser.add_argument('--mask_rates', default='0.25,0.75')
    parser.add_argument('--samples_per_bucket', default=10, type=int)
    # Fill Span Arguments
    parser.add_argument('--batch_size', default=16, type=int)  # Will use cuda:0 by default
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)
    parser.add_argument('--num_chunks', default=10, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=0, type=int)

    args = parser.parse_args()

    datasets = data_loader(args.dataset, contrast_subsample=True)
    mask_and_fill_dir = os.path.join(DATA_DIR, args.dataset, 'mask_and_fill')
    os.makedirs(mask_and_fill_dir, exist_ok=True)

    if args.mode in {'mask_spans', 'all'}:
        print('Loading SciSpacy...')
        nlp = spacy.load('en_core_sci_sm')
        mask_rates = list(map(float, args.mask_rates.split(',')))
        outputs = []
        for split, dataset in datasets.items():
            if args.debug:
                outputs += list(map(lambda record: build_masks(record, mask_rates, nlp), dataset))
            else:
                outputs += list(p_uimap(lambda record: build_masks(record, mask_rates, nlp), dataset))
        outputs = [x for x in outputs if x is not None]
        outputs = list(itertools.chain(*outputs))
        outputs = pd.DataFrame(outputs)
        out_fn = os.path.join(mask_and_fill_dir, 'span_masks.csv')
        print(f'Saving {len(outputs)} masked inputs to {out_fn}')
        outputs.to_csv(out_fn, index=False)

        for mask_rate in mask_rates:
            mr = outputs[outputs['target_mask_rate'] == mask_rate]
            removed_tokens = mr['removed_tokens'].dropna().mean()
            num_masks = mr['num_masks'].dropna().mean()
            print(f'Mask Rate {mask_rate}: Remove Tokens={removed_tokens}, Number of Masks={num_masks}')
    if args.mode in {'fill_spans', 'all'}:
        mask_filler = MaskFiller(device=args.gpu_device, num_beams=args.num_beams)
        in_fn = os.path.join(mask_and_fill_dir, 'span_masks.csv')
        print(f'Reading in masked abstracts from {in_fn}')
        df = pd.read_csv(in_fn)
        df['target_length'] = df['removed_tokens'] + df['num_masks']
        df.sort_values(by='target_length', inplace=True)
        prev_n = len(df)
        df = df[df['removed_tokens'] >= 1]
        n = len(df)
        empty_ct = prev_n - n
        print(f'{empty_ct} abstracts have no masks. Filtering them out.')
        if args.chunk_idx is None:
            chunk_df = df
            chunk_suffix = ''
        else:
            chunk_df = np.array_split(df, args.num_chunks)[args.chunk_idx]
            chunk_suffix = '_' + str(args.chunk_idx)

        records = chunk_df.to_dict('records')
        n = len(records)

        augmented_records = []
        batch_starts = list(range(0, n, args.batch_size))
        print('Starting to Fill...')
        for s in tqdm(batch_starts, desc=f'Filling in {n} masked abstracts'):
            e = min(n, s + args.batch_size)
            batch = records[s:e]
            batch_inputs = [x['masked_input'] for x in batch]
            target_length = batch[0]['target_length']
            batch_preds = mask_filler.fill(batch_inputs, target_length)

            for pred, row in zip(batch_preds, batch):
                row['prediction'] = pred
                augmented_records.append(row)
            print(f'Saved {len(augmented_records)} records so far.')
        augmented_df = pd.DataFrame(augmented_records)
        augmented_df.dropna(subset=['prediction'], inplace=True)
        augmented_df['num_abstract_tokens'] = augmented_df['abstract'].apply(lambda x: len(x.split(' ')))
        augmented_df['num_prediction_tokens'] = augmented_df['prediction'].apply(lambda x: len(x.split(' ')))

        print('Mean abstract tokens: ', augmented_df['num_abstract_tokens'].mean())
        print('Mean prediction tokens: ', augmented_df['num_prediction_tokens'].mean())
        print('Mean removed tokens: ', augmented_df['removed_tokens'].mean())
        print('Mean masks: ', augmented_df['num_masks'].mean())

        if 'AMLT_OUTPUT_DIR' in os.environ and os.environ['AMLT_OUTPUT_DIR'] is not None:
            out_dir = os.environ['AMLT_OUTPUT_DIR']
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = mask_and_fill_dir
        out_fn = os.path.join(out_dir, f'span_fills{chunk_suffix}.csv')
        print(f'Saving {len(augmented_df)} filled in examples to {out_fn}')
        augmented_df.to_csv(out_fn, index=False)
        mask_filler.cleanup()
    if args.mode in {'merge_chunks', 'all'}:
        chunk_fns = list(glob(os.path.join(mask_and_fill_dir, f'span_fills_*.csv')))
        chunk_fns = [x for x in chunk_fns if any(chr.isdigit() for chr in x)]

        output_df = []
        for fn in tqdm(chunk_fns, desc='Loading disjoint dataset chunks before merging into single dataframe...'):
            chunk_df = pd.read_csv(fn)
            print(f'Adding {len(chunk_df)} examples from {fn}')
            output_df.append(chunk_df)

        output_df = pd.concat(output_df)
        out_fn = os.path.join(mask_and_fill_dir, 'span_fills.csv')
        print(f'Saving {output_df} outputs to {out_fn}')
        # Ensure no duplicates
        uuid = output_df['uuid'] + output_df['target_mask_rate'].astype(str) + output_df['sample_idx'].astype(str)
        print(len(uuid), len(set(uuid)))
        output_df.to_csv(out_fn, index=False)
