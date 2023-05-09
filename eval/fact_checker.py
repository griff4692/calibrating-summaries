from tqdm import tqdm

from eval.multivers.longchecker.model import LongCheckerModel
from eval.multivers.longchecker.data import get_tokenizer
from eval.multivers.longchecker.util import *
from eval.utils import get_batch_ranges

# Heavily borrowed from https://github.com/dwadden/multivers/
PATH_TO_FACT_CHECKER = os.path.expanduser('~/data_tmp/scifact.ckpt')


class FactChecker:
    def __init__(self, device=0, batch_size=32) -> None:
        print(f'Loading fact checker from {PATH_TO_FACT_CHECKER}')
        self.device = device
        self.model = LongCheckerModel.load_from_checkpoint(
            PATH_TO_FACT_CHECKER, strict=False
        ).to(self.device).eval().half()
        self.model.freeze()
        self.tokenizer = get_tokenizer()
        self.labels = ['CONTRADICT', 'NEI', 'SUPPORT']
        self.batch_size = batch_size
    
    def _tokenize(self, claim, sentences):
        cited_text = self.tokenizer.eos_token.join(sentences)
        tokenized = self.tokenizer(claim + self.tokenizer.eos_token + cited_text, max_length=256, truncation=True)
        tokenized['global_attention_mask'] = self._get_global_attention_mask(tokenized)
        return tokenized
    
    def _get_global_attention_mask(self, tokenized):
        "Assign global attention to all special tokens and to the claim."
        input_ids = torch.tensor(tokenized.input_ids)
        # Get all the special tokens.
        is_special = (input_ids == self.tokenizer.bos_token_id) | (
            input_ids == self.tokenizer.eos_token_id
        )
        # Get all the claim tokens (everything before the first </s>).
        first_eos = torch.where(input_ids == self.tokenizer.eos_token_id)[0][0]
        is_claim = torch.arange(len(input_ids)) < first_eos
        # Use global attention if special token, or part of claim.
        global_attention_mask = is_special | is_claim
        # Unsqueeze to put in batch form, and cast like the tokenizer attention mask.
        global_attention_mask = global_attention_mask.to(torch.int64)
        return global_attention_mask.tolist()

    def _pad_tokenized(self, tokenized):
        """
        Pad the tokenizer outputs. Need to do this manually because the
        tokenizer's default padder doesn't expect `global_attention_mask` as an
        input.
        """
        fields = ["input_ids", "attention_mask", "global_attention_mask"]
        pad_values = [self.tokenizer.pad_token_id, 0, 0]
        tokenized_padded = {}
        for field, pad_value in zip(fields, pad_values):
            tokenized_padded[field] = self._pad_field(tokenized, field, pad_value)

        return tokenized_padded

    def _pad_field(self, entries, field_name, pad_value):
        xxs = [entry[field_name] for entry in entries]
        return self._pad(xxs, pad_value)

    @staticmethod
    def _pad(xxs, pad_value):
        """
        Pad a list of lists to the length of the longest entry, using the given
        `pad_value`.
        """
        res = []
        max_length = max(map(len, xxs))
        for entry in xxs:
            to_append = [pad_value] * (max_length - len(entry))
            padded = entry + to_append
            res.append(padded)

        return torch.tensor(res)

    def top_k_sents(self, candidates, comparison, max_tokens=128):
        compare_set = set(list(map(lambda x: x.lower(), comparison.split(' '))))
        priority = []
        for cand in candidates:
            cand_toks = set(list(map(lambda x: x.lower(), cand.split(' '))))
            overlap = len(cand_toks.intersection(compare_set)) / max(1, len(cand_toks))
            priority.append(-overlap)

        order = np.argsort(priority)
        output_idxs = []
        tokens_so_far = 0
        for idx in order:
            candidate = candidates[idx]
            num_toks = len(candidate.split(' '))
            if tokens_so_far + num_toks > max_tokens:
                break

            tokens_so_far += num_toks
            output_idxs.append(idx)

        return [candidates[idx] for idx in sorted(output_idxs)]

    def _compute(self, batch_summary_sents, batch_source_sents):
        n = len(batch_summary_sents)
        inputs = []
        example_idxs = []
        for batch_idx in tqdm(list(range(n)), total=n, desc='Top-K sentence alignment'):
            summary_sents = batch_summary_sents[batch_idx]
            source_sents = batch_source_sents[batch_idx]
            example_idxs.append((len(inputs), len(inputs) + len(summary_sents)))
            for claim in summary_sents:
                top_k_sents = self.top_k_sents(source_sents, claim)
                inputs.append({'claim': claim, 'sentences': top_k_sents})
        flat_n = len(inputs)
        all_logits = []
        batch_ranges = get_batch_ranges(flat_n, self.batch_size)
        with torch.no_grad():
            for batch_start, batch_end in tqdm(batch_ranges, total=len(batch_ranges), desc='Fact Checker'):
                batch_data = [inputs[batch_idx] for batch_idx in range(batch_start, batch_end)]
                batch_inputs = self._pad_tokenized([self._tokenize(x['claim'], x['sentences']) for x in batch_data])
                logits = self.model({k: v.to(self.device) for k, v in batch_inputs.items()})
                all_logits.append(logits.cpu())
        all_logits = torch.cat(all_logits)
        assert flat_n == len(all_logits)
        neg_preds = all_logits[:, 0].numpy().tolist()
        pos_preds = all_logits[:, 2].numpy().tolist()

        outputs = []
        for sent_idx_start, sent_idx_end in example_idxs:
            sent_level_pos_prob = []
            for neg, pos in zip(neg_preds[sent_idx_start:sent_idx_end], pos_preds[sent_idx_start:sent_idx_end]):
                norm_pos_prob = np.exp(pos) / (np.exp(pos) + np.exp(neg))
                sent_level_pos_prob.append(norm_pos_prob)
            avg_pos_prob = np.mean(sent_level_pos_prob)
            outputs.append({
                'fact_score': avg_pos_prob,
                'fact_score_str': ','.join(list(map(str, sent_level_pos_prob)))
            })
        return outputs

    def compute_batch(self, batch, queue=None, id_col='temp_id'):
        batch_source_sents = []
        batch_summary_sents = []
        for x in batch:
            sents_substantial = [y for y in x['prediction_sents'] if len(y.split(' ')) >= 5]
            if len(sents_substantial) == 0:
                sents_substantial = x['prediction_sents']
            sents_substantial = sents_substantial[:min(len(sents_substantial), 10)]
            batch_summary_sents.append(sents_substantial)
            batch_source_sents.append(x['source_sents'])

        outputs = self._compute(batch_summary_sents, batch_source_sents)
        for idx, output in enumerate(outputs):
            output[id_col] = batch[idx][id_col]

        if queue is None:
            return outputs
        queue.put(outputs)
        print('Exiting FactChecker...')
        exit(0)

    def cleanup(self):
        self.model.cpu()
