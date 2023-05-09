from evaluate import load
from nltk import sent_tokenize
from tqdm import tqdm
from collections import defaultdict

from eval.utils import get_batch_ranges


class BertScoreWrapper:
    def __init__(self, num_process=1, batch_size=128, device='cuda:0'):
        self.bertscore = load(
            'bertscore',
            batch_size=batch_size,
            num_process=num_process,
            device=device,
            model_type='allenai/scibert_scivocab_uncased',
            use_fast_tokenizer=True
        )
        self.batch_size = batch_size

    def compute(self, predictions, references, prefix='_src'):
        n = len(predictions)
        scores = self.bertscore.compute(predictions=predictions, references=references, lang='en')
        outputs = []
        for i in range(n):
            outputs.append({
                f'bs{prefix}_recall': scores['recall'][i],
                f'bs{prefix}_precision': scores['precision'][i],
                f'bs{prefix}_f1': scores['f1'][i],
            })
        return outputs

    def top_k_sents(self, candidates, source_order, max_tokens=1024):
        output_idxs = []
        tokens_so_far = 0
        for idx in source_order:
            candidate = candidates[idx]
            num_toks = len(candidate.split(' '))
            if tokens_so_far + num_toks > max_tokens:
                break

            tokens_so_far += num_toks
            output_idxs.append(idx)

        return [candidates[idx] for idx in sorted(output_idxs)]

    def compute_full(self, predictions, references, sources, source_is_trunc=False):
        batch_size = len(predictions)
        ref_outputs = self.compute(predictions=predictions, references=references, prefix='_ref')
        # Assumes source has already been split into sentences if each source is a list of strings rather than a string
        if type(sources[0]) == list:
            source_sents = sources
        else:
            source_sents = list(map(sent_tokenize, sources))

        if source_is_trunc:
            trunc_sources = sources
        else:
            trunc_sources = list(map(
                lambda idx: '\n'.join(self.top_k_sents(source_sents[idx], references[idx])), range(batch_size)
            ))
        outputs = self.compute(predictions=predictions, references=trunc_sources, prefix='_src')

        for i in range(len(outputs)):
            outputs[i].update(ref_outputs[i])
        return outputs
    
    def compute_batch(self, batch, queue=None, id_col='temp_id'):
        ids = []
        for x in batch:
            ids.append(x[id_col])

        trunc_sources = list(map(
            lambda idx: '\n'.join(
                self.top_k_sents(batch[idx]['source_sents'], batch[idx]['source_sent_alignment'])),
            range(len(batch))
        ))
        predictions_dup = []
        for x in batch:
            predictions_dup.extend([x['prediction']] * 2)

        references_dup = []
        for src, batch in zip(trunc_sources, batch):
            references_dup.extend([src, batch['reference']])

        n = len(predictions_dup)
        assert len(references_dup) == len(predictions_dup)
        batch_ranges = get_batch_ranges(n, 1024)
        scores = defaultdict(list)
        for batch_start, batch_end in tqdm(batch_ranges, total=len(batch_ranges), desc='BertScore'):
            p = [predictions_dup[i] for i in range(batch_start, batch_end)]
            r = [references_dup[i] for i in range(batch_start, batch_end)]
            batch_scores = self.bertscore.compute(predictions=p, references=r, lang='en', verbose=False)
            for k, arr in batch_scores.items():
                scores[k] += arr

        assert len(scores['recall']) == n
        outputs = []
        for start_idx in range(0, n, 2):
            outputs.append({
                'bs_src_recall': scores['recall'][start_idx],
                'bs_src_precision': scores['precision'][start_idx],
                'bs_src_f1': scores['f1'][start_idx],
                'bs_ref_recall': scores['recall'][start_idx + 1],
                'bs_ref_precision': scores['precision'][start_idx + 1],
                'bs_ref_f1': scores['f1'][start_idx + 1],
            })

        assert len(ids) == len(outputs)
        for id, output in zip(ids, outputs):
            output[id_col] = id

        if queue is None:
            return outputs
        queue.put(outputs)
        print('Exiting BertScore...')
        exit(0)
