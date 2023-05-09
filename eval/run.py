import os
from glob import glob

import argparse
from datasets import load_metric
import pandas as pd
import nltk
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from p_tqdm import p_uimap

from eval.bertscore import BertScoreWrapper
from eval.bartscore import LikelihoodWrapper
from eval.extractive_fragments import parse_extractive_fragments
from corruptions.diverse_decoding import compute_rouge
from eval.fact_checker import FactChecker
from preprocess.preprocess import data_loader
from corruptions.entity.bern_entities import clean_uuid


METRICS = ['rouge', 'extractive_fragments', 'bert_score', 'bart_score', 'fact_score']
METRIC_COLS = list(sorted([
    'num_prediction_tokens', 'coverage', 'density', 'compression', 'rouge1', 'rouge2',  # 'rougeL',
    'bs_src_recall', 'bs_src_precision', 'bs_src_f1', 'bs_ref_recall', 'bs_ref_precision', 'bs_ref_f1',
    'bart_score', 'fact_score'
]))


def df_to_table(args, df):
    print('Paste into Excel and ensure columns line up')
    print(','.join(METRIC_COLS))
    output_str = []
    for col in METRIC_COLS:
        if col not in df:
            val = 'N/A'
        else:
            val = str(round(df[col].dropna().mean(), 4))
        output_str.append(val)
    print(','.join(output_str))

    import ujson
    metric_norm_fn = os.path.join(args.data_dir, f'{args.dataset}_metric_bounds.json')
    with open(metric_norm_fn, 'r') as fd:
        stats = ujson.load(fd)

    faith_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
    relevance_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']

    agg_rel_scores = []
    agg_faith_scores = []
    for record in df.to_dict('records'):
        rel_row = float(np.mean([(record[c] - stats[c]['mean']) / stats[c]['std'] for c in relevance_metrics]))
        faith_row = float(np.mean([(record[c] - stats[c]['mean']) / stats[c]['std'] for c in faith_metrics]))
        agg_faith_scores.append(faith_row)
        agg_rel_scores.append(rel_row)

    agg_rel = float(np.mean(agg_rel_scores))
    agg_faith = float(np.mean(agg_faith_scores))
    print(f'Relevance Agg: {round(agg_rel, 3)}')
    print(f'Faith Agg: {round(agg_faith, 3)}')


def source_sent_alignment(candidates, comparison):
    compare_set = set(list(map(lambda x: x.lower(), comparison.split(' '))))
    priority = []
    for cand in candidates:
        cand_toks = set(list(map(lambda x: x.lower(), cand.split(' '))))
        overlap = len(cand_toks.intersection(compare_set)) / max(1, len(cand_toks))
        priority.append(-overlap)
    order = np.argsort(priority)
    return list(order)


def remove_eos_bos_from_str(text):
    return text.replace('<s>', ' ').replace('</s>', ' ')


def tokenize(text, lower=True):
    tokens = nltk.word_tokenize(text)
    if lower:
        tokens = [tok.lower() for tok in tokens]
    return tokens


def prepare(record, orig_data, uuid_cache=None, include_tokens=True, include_source=True):
    uuid = record['uuid']

    prediction = remove_eos_bos_from_str(record['prediction'])
    prediction_sents = nltk.sent_tokenize(prediction)
    prediction_tokens = None
    if include_tokens:
        try:
            prediction_tokens = tokenize(prediction)
        except:
            print(prediction)
            print(type(prediction), prediction is None)
            print('Error tokenizing prediction.')
            prediction_tokens = prediction.split(' ')

    if uuid_cache is not None and uuid in uuid_cache:
        outputs = uuid_cache.get(uuid).copy()
    else:
        reference = remove_eos_bos_from_str(orig_data['target'])
        reference_sents = nltk.sent_tokenize(reference)
        outputs = {
            'reference': reference,
            'reference_sents': reference_sents,
        }
        if include_source:
            source = remove_eos_bos_from_str(orig_data['input'])
            source_tokens = tokenize(source) if include_tokens else None
            source_sents = nltk.sent_tokenize(source)
            source_pre = {
                'source': source,
                'source_tokens': source_tokens,
                'source_sents': source_sents,
                'source_sent_alignment': source_sent_alignment(source_sents, prediction),
            }
            outputs.update(source_pre)
    processed = {
        'uuid': uuid,
        'temp_id': record['temp_id'],
        'prediction': prediction,
        'prediction_sents': prediction_sents,
        'prediction_tokens': prediction_tokens,
        'num_prediction_tokens': None if prediction_tokens is None else len(prediction_tokens)
    }

    for k, v in processed.items():
        outputs[k] = v
    return outputs


def single_frac(record):
    frag_obj = parse_extractive_fragments(record['source_tokens'], record['prediction_tokens'], remove_stop=False)
    frag_obj.pop('fragments')
    row = {'temp_id': record['temp_id'], 'num_prediction_tokens': len(record['prediction_tokens'])}
    row.update(frag_obj)
    return row


def _compute_extractive_frags(records, queue=None):
    outputs = list(p_uimap(lambda record: single_frac(record), records, num_cpus=5))
    if queue is None:
        return outputs
    queue.put(outputs)
    print('Exiting extractive fragments...')
    exit(0)


def single_rouge(record, rouge_metric):
    row = {'temp_id': record['temp_id']}
    row.update(compute_rouge(rouge_metric, record['reference'], record['prediction'], rouge_types=['rouge1', 'rouge2']))
    return row


def _compute_rouge(records, queue=None):
    rouge_metric = load_metric('rouge')
    outputs = list(p_uimap(lambda record: single_rouge(record, rouge_metric), records, num_cpus=0.5))
    if queue is None:
        return outputs
    queue.put(outputs)
    print('Exiting ROUGE...')
    exit(0)


def run_single_metric(records, bartscore_hf_model, bartscore_path, uuid2data, metric='rouge'):
    print('Preprocessing inputs and outputs...')
    uuid_cache = {}
    eval_inputs = []
    for record in tqdm(records, total=len(records)):
        if args.dataset == 'chemistry':
            uuid = clean_uuid(record['uuid'])
        else:
            uuid = record['uuid']

        row = prepare(
            record, uuid2data[uuid], uuid_cache, include_tokens=metric=='extractive_fragments',
            include_source=metric != 'rouge'
        )

        if metric == 'extractive_fragments':
            # We just need the source and prediction tokens
            # Memory grows if we keep everything
            row = {
                'uuid': row['uuid'],
                'temp_id': row['temp_id'],
                'source_tokens': row['source_tokens'],
                'prediction_tokens': row['prediction_tokens'],
                'num_prediction_tokens': row['num_prediction_tokens']
            }

        if row['uuid'] not in uuid_cache:
            uuid_cache[row['uuid']] = row
        eval_inputs.append(row)
    del uuid_cache  # Clear up the memory
    print('Done preprocessing...')

    if metric == 'bert_score':
        print('Initializing BERTScore')
        bert_scorer = BertScoreWrapper()
        metric_outputs = bert_scorer.compute_batch(eval_inputs)
    elif metric == 'bart_score':
        print('Initializing BartScore')
        bart_scorer = LikelihoodWrapper(hf_config=bartscore_hf_model, model_path=bartscore_path)
        metric_outputs = bart_scorer.compute_batch(eval_inputs)
        bart_scorer.cleanup()
    elif metric == 'extractive_fragments':
        metric_outputs = _compute_extractive_frags(eval_inputs)
    elif metric == 'rouge':
        metric_outputs = _compute_rouge(eval_inputs)
    elif metric == 'fact_score':
        print('Initializing FactChecker')
        fact_checker = FactChecker()
        metric_outputs = fact_checker.compute_batch(eval_inputs)
        fact_checker.cleanup()
    else:
        raise Exception(f'Unrecognized metric: {metric}')

    print('Merging metrics')
    metric_outputs_by_id = defaultdict(dict)
    for metric_output in metric_outputs:
        metric_outputs_by_id[metric_output.pop('temp_id')].update(metric_output)

    for record in records:
        temp_id = record.pop('temp_id')
        for k, v in metric_outputs_by_id[temp_id].items():
            record[k] = v

    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate Abstracts (real and synthetic corruptions)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset')
    parser.add_argument('--fp', default='weights/primera_final/results/predictions.csv')
    parser.add_argument('--mode', default='evaluate', choices=['evaluate', 'merge_chunks', 'merge_metrics', 'to_table'])
    parser.add_argument('-erase_after_merge', default=False, action='store_true')
    parser.add_argument('--metric', default=None, choices=METRICS)
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    if args.dataset is None:
        if 'pubmed' in args.fp:
            args.dataset = 'pubmed'
        elif 'chem' in args.fp:
            args.dataset = 'chemistry'
        elif 'clin' in args.fp:
            args.dataset = 'clinical'
        else:
            raise Exception(f'Could not infer dataset from {args.fn}. Please set explicitly with --dataset flag.')

    if args.dataset in {'chemistry', 'pubmed'}:
        bartscore_path = None
        bartscore_hf_model = 'google/pegasus-pubmed'
    else:
        bartscore_path = os.path.join(args.data_dir, args.dataset, 'clinical_bart_score.ckpt')
        bartscore_hf_model = 'allenai/led-base-16384'
    prediction_fn = os.path.join(args.data_dir, args.fp)
    metric_suffix = 'metrics' if args.metric is None else args.metric
    out_fn = prediction_fn.replace('.csv', '') + f'_with_{metric_suffix}.csv'

    if args.mode == 'to_table':
        if 'with_metrics' not in prediction_fn:
            prediction_fn = prediction_fn.replace('.csv', '') + f'_with_metrics.csv'
        df = pd.read_csv(prediction_fn)
        df_to_table(args, df)
        exit(0)

    full_fn = prediction_fn.replace('.csv', '') + f'_with_metrics.csv'
    if os.path.exists(out_fn) or os.path.exists(full_fn):
        print(f'Metric outfile already exists -> {out_fn} or {full_fn}')
        if not args.overwrite:
            print('Exiting. Must run with -overwrite to re-run this evaluation.')
            exit(0)

    if args.mode == 'merge_chunks':
        in_pattern = prediction_fn.replace('.csv', '') + '_with_metrics_\d_\d.csv'
        fns = list(glob(in_pattern))

        dfs = pd.concat([
            pd.read_csv(fn) for fn in fns
        ])
        out_fn = prediction_fn.replace('.csv', '') + '_with_metrics.csv'
        print(f'Saving merged to {out_fn}')
        dfs.to_csv(out_fn, index=False)
        if args.erase_after_merge:
            for fn in fns:
                os.remove(fn)
        exit(0)
    if args.mode == 'merge_metrics':
        in_pattern = prediction_fn.replace('.csv', '') + '_with_{}.csv'
        out_fn = prediction_fn.replace('.csv', '') + '_with_metrics.csv'

        if os.path.exists(out_fn):
            if args.overwrite:
                print(f'Overwriting existing metrics file: {out_fn}')
            else:
                print(f'{out_fn} exists. Run with -overwrite if you want to replace.')
                exit(0)

        dfs = []
        fns = []
        for metric in METRICS:
            fn = in_pattern.format(metric)
            if os.path.exists(fn):
                print(f'Loading {fn}')
                fns.append(fn)
            else:
                print(f'{fn} does not exist.')
        
        dfs = [pd.read_csv(fn) for fn in fns]
        lens = [len(x) for x in dfs]
        if len(set(lens)) != 1:
            for fn, l, in zip(fns, lens):
                print(fn + ' -> ' + str(l))
            raise Exception('Merge files not the same length')

        merged = dfs[0]
        cols = merged.columns
        merged_predictions = merged['prediction'].tolist()
        for idx in range(1, len(dfs)):
            new_df = dfs[idx]
            new_cols = list(new_df.columns)
            new_cols = [col for col in new_cols if col not in list(merged.columns)]
            new_predictions = new_df['prediction'].tolist()
            assert all([a == b for a, b in zip(merged_predictions, new_predictions)])
            new_col_str = ', '.join(new_cols)
            print(f'Adding {new_col_str} from {fns[idx]}')
            for new_col in new_cols:
                merged[new_col] = new_df[new_col]
        print(f'Saving merged to {out_fn}')
        merged.to_csv(out_fn, index=False)
        if args.erase_after_merge:
            for fn in fns:
                print(f'Erasing {fn}')
                os.remove(fn)
        
        df_to_table(args, merged)
        exit(0)

    data = data_loader(args.dataset, contrast_subsample=False)
    uuid2data = {}
    for split, split_data in data.items():
        for record in split_data:
            if args.dataset == 'chemistry':
                uuid = clean_uuid(record['uuid'])
            else:
                uuid = record['uuid']
            uuid2data[uuid] = record

    print(f'Loading in predictions from {prediction_fn}')
    predict_df = pd.read_csv(prediction_fn)

    # TODO eventually remove
    if 'uuid_fixed' in predict_df.columns:
        print('Treating uuid_fixed as uuid...')
        # Backwards compatibility with diverse_decode_debug script (Remove these lines if/when we re-run)
        predict_df = predict_df.dropna(subset='uuid_fixed').reset_index(drop=True)
        predict_df['uuid_prev'] = predict_df['uuid']
        predict_df['uuid'] = predict_df['uuid_fixed']
        predict_df.drop(columns=['uuid_fixed'], inplace=True)

    predict_df = predict_df.sort_values(by='uuid')
    predict_df.dropna(subset=['prediction', 'uuid'], inplace=True)
    predict_df = predict_df.assign(temp_id=list(range(len(predict_df))))
    records = predict_df.to_dict('records')

    augmented_records = run_single_metric(
        records, bartscore_hf_model=bartscore_hf_model, bartscore_path=bartscore_path, uuid2data=uuid2data,
        metric=args.metric
    )
    print('Statistics returned. Storing them in a dataframe with original columns.')
    augmented_df = pd.DataFrame(augmented_records)
    n = len(augmented_df)

    if 'AMLT_OUTPUT_DIR' in os.environ and os.environ['AMLT_OUTPUT_DIR'] is not None:
        out_dir = os.environ['AMLT_OUTPUT_DIR']
        os.makedirs(out_dir, exist_ok=True)
        out_fn = os.path.join(out_dir, out_fn.split('/')[-1])
    print(f'Saving {n} to {out_fn}')
    augmented_df.to_csv(out_fn, index=False)
