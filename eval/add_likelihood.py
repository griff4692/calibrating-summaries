import os
from glob import glob

import argparse
import torch
import ujson
from transformers import AutoConfig, LEDForConditionalGeneration, LEDTokenizerFast
from transformers.trainer_pt_utils import LabelSmoother
from tqdm import tqdm
from preprocess.preprocess import data_loader
from eval.utils import get_batch_ranges


def remove_eos_bos_from_str(text):
    return text.replace('<s>', ' ').replace('</s>', ' ')


HF_TRANSFORMER = os.path.expanduser('~/RoBERTa-base-PM-M3-Voc-distill-align-hf')


def compute_likelihood(
        source, predictions, model, tokenizer, max_target_length=512, max_source_length=4096, batch_size=4
):
    model_inputs = tokenizer(
        source, add_special_tokens=True, max_length=max_source_length,
        padding=True, truncation=True, return_tensors='pt',
    )

    global_attention_mask = torch.zeros_like(model_inputs['input_ids'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    model_inputs['global_attention_mask'] = global_attention_mask
    n = len(predictions)

    encoder_outputs = model.led.encoder(
        **{k: v.to(model.device) for k, v in model_inputs.items()}
    ).last_hidden_state  # .repeat(len(predictions), 1, 1)

    batches = get_batch_ranges(n, batch_size=batch_size)

    scores = []
    for s, e in batches:
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                predictions[s:e], add_special_tokens=True, max_length=max_target_length, padding=True, truncation=True,
                return_tensors='pt'
            )['input_ids'].to(model.device)
        labels[labels == tokenizer.pad_token_id] = -100
        decoder_inputs = {
            'encoder_outputs': [encoder_outputs.repeat(len(labels), 1, 1)],
            'decoder_input_ids': model.prepare_decoder_input_ids_from_labels(labels),
        }

        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_logits = model(**decoder_inputs).logits
            for logit, label in zip(batch_logits, labels):
                nll = float(label_smoother({'logits': logit}, label).cpu().item())
                scores.append(-nll)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate Abstracts (real and synthetic corruptions)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--experiment', default='primera_ft_{}')
    parser.add_argument('--ckpt_name', default='best_ckpt')
    parser.add_argument('--device', default=2, type=int)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-overwrite', action='store_true', default=False)
    parser.add_argument('--hf_path', default='allenai/PRIMERA')
    parser.add_argument('--store_col', default='primera_bertscore')  # should be primera_bartscore

    args = parser.parse_args()

    weight_dir = os.path.join(args.data_dir, 'weights')
    experiment_dir = os.path.join(weight_dir, args.experiment.format(args.dataset))

    ckpt_dir = os.path.join(experiment_dir, args.ckpt_name)
    tokenizer_dir = os.path.join(experiment_dir, 'tokenizer')

    label_smoother = LabelSmoother(0.1)

    print(f'Loading config from {args.hf_path}')
    config = AutoConfig.from_pretrained(args.hf_path)
    print(f'Loading tokenizer from {tokenizer_dir}')

    corrupt_dir = os.path.join(args.data_dir, args.dataset, 'corruptions')
    print(f'Reading in data to annotate with labels from {corrupt_dir}')
    pattern = os.path.join(corrupt_dir, '*', '*.json')
    fns = glob(pattern)
    fns = [fn for fn in fns if 'invalid' not in fn]
    print(f'Found {len(fns)} files matching {pattern}')

    tokenizer = LEDTokenizerFast.from_pretrained(tokenizer_dir)

    config.vocab_size = len(tokenizer)
    print(f'Loading model from {ckpt_dir}')

    model = LEDForConditionalGeneration.from_pretrained(ckpt_dir, from_tf=False, config=config).to(args.device)
    model = model.eval().half()
    model.resize_token_embeddings(len(tokenizer))

    data = data_loader(args.dataset, contrast_subsample=True)
    uuid2data = {}
    for record in data['train']:
        uuid2data[record['uuid']] = record
    for record in data['validation']:
        uuid2data[record['uuid']] = record

    for fn in tqdm(fns):
        with open(fn, 'r') as fd:
            records = ujson.load(fd)

        if args.store_col in records[0] and not args.overwrite:
            print(f'Already Done! Skipping {fn}...')
            continue

        predictions = [x['prediction'] for x in records]
        try:
            orig_data = uuid2data[records[0]['uuid']]
        except:
            print('Could not find UUID ', records[0]['uuid'])
            continue
        source = remove_eos_bos_from_str(orig_data['input'])
        bartscores = compute_likelihood(source, predictions, model, tokenizer)
        for bartscore, record in zip(bartscores, records):
            record[args.store_col] = bartscore

        with open(fn, 'w') as fd:
            ujson.dump(records, fd)
