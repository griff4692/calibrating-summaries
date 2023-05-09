import os
import pandas as pd
import ujson
from tqdm import tqdm
import regex as re
from glob import glob
import argparse
from collections import defaultdict
import numpy as np
from collections import Counter
from eval.diversity import diversity_score
from p_tqdm import p_uimap
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from transformers import DataCollatorForContrastSeq2Seq


def cross_diversity(arr_a, arr_b):
    cross_divs = []
    for a in range(len(arr_a)):
        max_div = max([diversity_score([arr_a[a], arr_b[b]]) for b in range(len(arr_b))])
        cross_divs.append(max_div)
    return float(np.mean(cross_divs))


def record(args, fn):
    stats_by_method = defaultdict(lambda: defaultdict(list))
    with open(fn, 'r') as fd:
        cset = ujson.load(fd)
        seen = set()
        cset_filt = [x for x in cset if x['method'] == 'reference']
        non_ref = [x for x in cset if x['method'] != 'reference']
        for cs in non_ref:
            cs['prediction'] = cs['prediction'].strip()
            if cs['prediction'] in seen:
                continue
            else:
                seen.add(cs['prediction'])
                cset_filt.append(cs)

        for strategy in strategies:
            collator.contrast_sample_strategy = strategy
            if args.metric == 'relevance':
                try:
                    subset = collator.select_mixed_methods(cset_filt, args.max_num_rank)
                except Exception as e:
                    print(e)
                    continue
                subset_obj = []
                for pred in subset:
                    for x in cset_filt:
                        if x['prediction'] == pred:
                            subset_obj.append(x)
                            break

                likelihoods = []
                for x in subset_obj:
                    if 'primera_bertscore' in x:
                        likelihoods.append(x['primera_bertscore'])
                    elif 'primera_bartscore' in x:
                        likelihoods.append(x['primera_bartscore'])
                if len(likelihoods) > 0:
                    stats_by_method[strategy]['likelihood'].append(float(np.mean(likelihoods)))
                    rank_corel = spearmanr(likelihoods, -np.arange(len(likelihoods)))[0]
                    if not np.isnan(rank_corel):
                        stats_by_method[strategy]['calibration'].append(rank_corel)
                avg_beam = np.mean([x['sample_idx'] + 1 for x in subset_obj])

                toks = [[x.strip() for x in re.split(r'\s+', y) if len(x.strip()) > 0] for y in subset]
                lengths = list(map(len, toks))
                length = np.mean(lengths)
                stats_by_method[strategy]['beam'].append(avg_beam)
                stats_by_method[strategy]['length'].append(length)
                methods = Counter([x['method'] for x in subset_obj])
                for k, v in methods.items():
                    key = 'primera' if 'primera' in k else 'long_t5'
                    stats_by_method[strategy][key].append(v / len(subset_obj))
                avg_relevance = float(np.mean([
                    score_candidate_fn(x, relevance_metrics) for x in subset_obj
                ]))
                rels = [score_candidate_fn(x, relevance_metrics) for x in subset_obj]
                faiths = [score_candidate_fn(x, faith_metrics) for x in subset_obj]

                faith_rel_corel = spearmanr(rels, faiths)[0]
                stats_by_method[strategy]['faith_relevance_pearson_corel'].append(faith_rel_corel)

                rels = np.sort(rels)
                gaps = []
                for i in range(1, len(rels)):
                    gaps.append(abs(rels[i] - rels[i - 1]))
                avg_gap = np.mean(gaps)
                avg_faithful = float(np.mean([
                    score_candidate_fn(x, faith_metrics) for x in subset_obj
                ]))
                avg_density = float(np.mean([x['density'] for x in subset_obj]))
                avg_coverage = float(np.mean([x['coverage'] for x in subset_obj]))
                stats_by_method[strategy]['diversity'].append(diversity_score(subset))
                stats_by_method[strategy]['density'].append(avg_density)
                stats_by_method[strategy]['coverage'].append(avg_coverage)
                stats_by_method[strategy]['relevance'].append(avg_relevance)
                stats_by_method[strategy]['faithful'].append(avg_faithful)
                stats_by_method[strategy]['metric_gap'].append(avg_gap)
            else:
                try:
                    subset = collator.select_hard_set(cset_filt)
                except Exception as e:
                    print(f'Caught exception {e}')
                    print('Skipping...')
                    continue

                pos_obj = [x for x in cset_filt if x['prediction'] in subset[:2]]
                neg_obj = [x for x in cset_filt if x['prediction'] in subset[2:]]

                neg_abs = [x['prediction'] for x in neg_obj]
                pos_abs = [x['prediction'] for x in pos_obj]
                neg_div = diversity_score(neg_abs)
                pos_div = diversity_score(pos_abs)

                pos_toks = [[x.strip() for x in re.split(r'\s+', y) if len(x.strip()) > 0] for y in pos_abs]
                neg_toks = [[x.strip() for x in re.split(r'\s+', y) if len(x.strip()) > 0] for y in neg_abs]
                pos_lens = list(map(len, pos_toks))
                neg_lens = list(map(len, neg_toks))

                pos_rels = [score_candidate_fn(x, relevance_metrics) for x in pos_obj]
                neg_rels = [score_candidate_fn(x, relevance_metrics) for x in neg_obj]

                pos_faiths = [score_candidate_fn(x, faith_metrics) for x in pos_obj]
                neg_faiths = [score_candidate_fn(x, faith_metrics) for x in neg_obj]

                faith_gap = np.mean(pos_faiths) - np.mean(neg_faiths)
                rel_gap = np.mean(pos_rels) - np.mean(neg_rels)
                stats_by_method[strategy]['faithful_gap'].append(faith_gap)
                stats_by_method[strategy]['relevance_gap'].append(rel_gap)

                stats_by_method[strategy]['relevance_pos'].append(float(np.mean(pos_rels)))
                stats_by_method[strategy]['faithful_pos'].append(float(np.mean(pos_faiths)))
                stats_by_method[strategy]['relevance_neg'].append(float(np.mean(neg_rels)))
                stats_by_method[strategy]['faithful_neg'].append(float(np.mean(neg_faiths)))

                avg_pos_len = np.mean(pos_lens)
                avg_neg_len = np.mean(neg_lens)
                stats_by_method[strategy]['lengths_positive'].append(avg_pos_len)
                stats_by_method[strategy]['lengths_negative'].append(avg_neg_len)
                stats_by_method[strategy]['length_gap'].append(avg_pos_len - avg_neg_len)

                stats_by_method[strategy]['diversity_positive'].append(neg_div)
                stats_by_method[strategy]['diversity_negative'].append(pos_div)
                stats_by_method[strategy]['diversity_cross'].append(cross_diversity(pos_abs, neg_abs))
                neg_methods = Counter([x['method'] for x in neg_obj])
                max_method_frac = neg_methods.most_common(1)[0][1] / len(neg_obj)
                stats_by_method[strategy]['max_neg_fraction'].append(max_method_frac)
                uses_reference = 0
                for x in pos_obj:
                    if x['method'] == 'reference':
                        uses_reference = 1
                        break
                stats_by_method[strategy]['includes_reference'].append(uses_reference)

                for k, v in neg_methods.items():
                    stats_by_method[strategy][k].append(v / len(neg_obj))

                pos_likelihoods = []
                for x in pos_obj:
                    if 'primera_bertscore' in x:
                        pos_likelihoods.append(x['primera_bertscore'])
                    elif 'primera_bartscore' in x:
                        pos_likelihoods.append(x['primera_bartscore'])

                pos_turing = float(np.mean([x['turing_score'] for x in pos_obj]))
                neg_turing = float(np.mean([x['turing_score'] for x in neg_obj]))

                stats_by_method[strategy]['turing_positive'].append(pos_turing)
                stats_by_method[strategy]['turing_negative'].append(neg_turing)
                stats_by_method[strategy]['turing_gap'].append(pos_turing - neg_turing)

                neg_likelihoods = []
                for x in neg_obj:
                    if 'primera_bertscore' in x:
                        neg_likelihoods.append(x['primera_bertscore'])
                    elif 'primera_bartscore' in x:
                        neg_likelihoods.append(x['primera_bartscore'])

                if len(neg_likelihoods) > 0:
                    ap = float(np.mean(pos_likelihoods))
                    an = float(np.mean(neg_likelihoods))
                    stats_by_method[strategy]['likelihood_positive'].append(ap)
                    stats_by_method[strategy]['likelihood_positive'].append(an)
                    stats_by_method[strategy]['likelihood_gap'].append(ap - an)

                neg_density = float(np.mean([x['density'] for x in neg_obj]))
                neg_coverage = float(np.mean([x['coverage'] for x in neg_obj]))
                stats_by_method[strategy]['density_negative'].append(neg_density)
                stats_by_method[strategy]['coverage_negative'].append(neg_coverage)

                pos_density = float(np.mean([x['density'] for x in pos_obj]))
                pos_coverage = float(np.mean([x['coverage'] for x in pos_obj]))

                stats_by_method[strategy]['density_positive'].append(pos_density)
                stats_by_method[strategy]['coverage_positive'].append(pos_coverage)

                stats_by_method[strategy]['density_gap'].append(pos_density - neg_density)
                stats_by_method[strategy]['coverage_gap'].append(pos_coverage - neg_coverage)
    return stats_by_method


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to analyze different sampling strategies for calibration')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical')
    parser.add_argument('--metric', default='faithful')
    parser.add_argument('--max_num_rank', default=4, type=int)
    parser.add_argument('--max_num_negative', default=2, type=int)
    parser.add_argument('--max_num_positive', default=2, type=int)
    parser.add_argument('--max_examples', default=1000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--split', default='train')

    args = parser.parse_args()
    dummy = AutoTokenizer.from_pretrained('sshleifer/bart-tiny-random')

    metric_norm_fn = os.path.join(args.data_dir, f'{args.dataset}_metric_bounds.json')
    with open(metric_norm_fn, 'r') as fd:
        stats = ujson.load(fd)

    faith_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
    relevance_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']
    if args.metric == 'faithful':
        strategies = [
            'random', 'max_margin', 'min_margin', 'avg_margin', 'max_diversity', 'min_diversity',
            'easy', 'hard', 'max_extractive_gap'
        ]
        default_metrics = faith_metrics.copy()
    elif args.metric == 'relevance':
        strategies = [
            'random', 'max_margin', 'min_margin', 'min_metric', 'max_metric', 'max_gap', 'min_gap',
            'max_diversity', 'min_diversity', 'top_beam', 'bottom_beam',
            'wide_beam', 'max_length', 'min_length', 'max_faithful'
        ]
        default_metrics = relevance_metrics.copy()
    else:
        raise Exception(f'Unrecognized metric: {args.metric}')

    def score_candidate_fn(row, contrast_metrics=default_metrics):
        norm_vals = []
        for metric in contrast_metrics:
            stat = stats[metric]
            norm_vals.append((row[metric] - stat['mean']) / stat['std'])
        return sum(norm_vals) / len(norm_vals)

    collator = DataCollatorForContrastSeq2Seq(
        tokenizer=dummy,
        max_num_rank=args.max_num_rank,
        max_num_positive=args.max_num_positive,
        max_num_negative=args.max_num_negative,
        score_candidate_fn=score_candidate_fn,
        metric_mode='max',
        positive_methods='all',
        mixed_methods='all',
        negative_methods='none' if args.metric == 'relevance' else 'all',
        reference_status='remove' if args.metric == 'relevance' else 'positive',
        use_mixed_methods=args.metric == 'relevance'
    )

    pattern = os.path.join(args.data_dir, args.dataset, 'corruptions', args.split, '*.json')
    print(f'Looking for files matching {pattern}')
    fns = list(glob(pattern))

    n = len(fns)
    if n > args.max_examples:
        np.random.seed(1992)
        fns = list(np.random.choice(fns, size=(args.max_examples, ), replace=False))
    all_stats_by_method = defaultdict(lambda: defaultdict(list))
    if args.debug:
        single_stats_by_method = list(tqdm(map(lambda fn: record(args, fn), fns), total=len(fns)))
    else:
        single_stats_by_method = list(p_uimap(lambda fn: record(args, fn), fns, num_cpus=16))

    for stats_by_method in single_stats_by_method:
        for strategy, obj in stats_by_method.items():
            for k, v in obj.items():
                all_stats_by_method[strategy][k] += v

    out_df = []
    for strategy, obj in all_stats_by_method.items():
        strat_row = {'strategy': strategy}
        for k, v in obj.items():
            v_valid = [z for z in v if not np.isnan(z)]
            strat_row[k] = float(np.mean(v_valid))
        out_df.append(strat_row)
    out_df = pd.DataFrame(out_df)
    out_fn = os.path.join(args.data_dir, f'{args.dataset}_{args.metric}_strategy_covariates.csv')
    out_df = out_df.reindex(sorted(out_df.columns), axis=1)
    out_df.reset_index(drop=True, inplace=True)
    print(f'Saving to {out_fn}...')
    out_df.to_csv(out_fn, index=False)
