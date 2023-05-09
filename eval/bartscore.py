import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LEDForConditionalGeneration,
    PegasusForConditionalGeneration,
    PegasusTokenizer
)
from transformers.trainer_pt_utils import LabelSmoother
from tqdm import tqdm

from eval.utils import get_batch_ranges


def add_global_attention_mask(batch):
    global_attention_mask = torch.zeros_like(batch['input_ids']).to(batch['input_ids'].device)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    batch['global_attention_mask'] = global_attention_mask


class LikelihoodWrapper:
    def __init__(self, hf_config, model_path=None, device='cuda:0', batch_size=4):
        self.device = device
        self.batch_size = batch_size
        if 'pegasus' in hf_config:
            model_constructor = PegasusForConditionalGeneration
            tokenizer_constructor = PegasusTokenizer
            self.max_source_length = 1024
        elif 'led' in hf_config:
            model_constructor = LEDForConditionalGeneration
            tokenizer_constructor = AutoTokenizer
            self.max_source_length = 4096
        else:
            raise Exception('Unrecognized HF model...')
        self.max_target_length = 512  # Too slow otherwise
        self.hf_config = hf_config
        config = AutoConfig.from_pretrained(hf_config)
        self.tokenizer = tokenizer_constructor.from_pretrained(hf_config)
        config.vocab_size = len(self.tokenizer)
        if 'led' in hf_config and model_path is not None:  # This is a Pytorch Lightning wrapped HuggingFace model
            weights = torch.load(model_path)['state_dict']
            led_weights = {k.replace('model.', ''): v for k, v in weights.items()}
            for k, v in weights.items():
                if k.startswith('model.'):
                    led_weights[k.replace('model.', '')] = v
            self.model = model_constructor(config=config).to(device).eval().half()
            self.model.load_state_dict(led_weights)
        else:
            model_path = model_path if model_path is not None else hf_config
            self.model = model_constructor.from_pretrained(model_path, config=config).to(device).eval()
        self.label_smoother = LabelSmoother(0.1)
        self.metric_names = 'bart_score'

    def compute(self, summary, source):
        """
        p(summary|source) by pre-trained model
        Returns log likelihood
        Model is any pre-trained HuggingFace model loaded in constructor.
        """
        model_inputs = self.tokenizer(
            source, add_special_tokens=True, max_length=self.max_source_length,
            padding=False, truncation=True, return_tensors='pt'
        )
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summary, add_special_tokens=True, max_length=self.max_target_length, padding=False, truncation=True,
                return_tensors='pt'
            )

        model_inputs['labels'] = labels['input_ids']

        if 'led' in self.hf_config:
            global_attention_mask = torch.zeros_like(model_inputs['input_ids'])
            # put global attention on <s> token
            global_attention_mask[:, 0] = 1
            model_inputs['global_attention_mask'] = global_attention_mask
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in model_inputs.items()})
            loss = outputs.loss
            return {'bart_score': -loss.item()}

    def compute_batch(self, batch, queue=None, id_col='temp_id'):
        n = len(batch)
        ids = [x[id_col] for x in batch]
        batch_ranges = get_batch_ranges(n, self.batch_size)

        print('Starting BartScore inference...')
        bartscores = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch_start, batch_end in tqdm(batch_ranges, total=len(batch_ranges), desc='BartScore'):
                batch_data = [batch[i] for i in range(batch_start, batch_end)]
                model_inputs = self.tokenizer(
                    [x['source'] for x in batch_data],
                    add_special_tokens=True, max_length=self.max_source_length,
                    padding=True, truncation=True, return_tensors='pt'
                )
                # Setup the tokenizer for targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        [x['prediction'] for x in batch_data],
                        add_special_tokens=True, max_length=self.max_target_length,
                        padding=True, truncation=True,
                        return_tensors='pt'
                    )['input_ids']

                labels[labels == self.tokenizer.pad_token_id] = -100
                model_inputs['labels'] = labels

                if 'led' in self.hf_config or 'primera' in self.hf_config:
                    # print(f'Adding global attention masks')
                    add_global_attention_mask(model_inputs)

                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
                batch_logits = self.model(**model_inputs).logits
                for logit, label in zip(batch_logits, model_inputs['labels']):
                    nll = float(self.label_smoother({'logits': logit}, label).cpu().item())
                    bartscores.append(-nll)
        assert len(ids) == len(bartscores)
        outputs = [{'bart_score': bartscore, id_col: id} for bartscore, id in zip(bartscores, ids)]
        if queue is None:
            return outputs
        queue.put(outputs)
        print('Exiting BartScore...')
        exit(0)

    def cleanup(self):
        self.model.cpu()
