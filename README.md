# Background

This is the official PyTorch / HuggingFace codebase for the ACL 2023 paper: [What are the Desired Characteristics of Calibration Sets? Identifying Correlates on Long Form Scientific Summarization](https://openreview.net/pdf?id=bIC0BfWzCs).
This repository provides an adaptable toolkit for constructing, selecting, and optimizing contrast sets for relevance and faithfulness calibration across three long-form scientific summariation datasets (spanning Chemistry, Clinical, and Biomedical).

It can be adapted for other summarization datasets in other domains as well.

**We encourage you to submit pull requests and raise issues! The code will be actively maintained by the authors.**  Otherwise, please reach out to `griffin.adams@columbia.edu`.

#  Setup

```angular2html
pip install -e .
cd transformers && pip install -e .
python -m nltk.downloader stopwords
mkdir ~/data_tmp
```

## Download Data

To download the `Fine-Tuned` PRIMERA and Long T5 models, which are used for initialization the weights for calibration (`Further Fine-Tuning`, and download the pre-processed and scored candidate sets (`corruptions`), run 

```
bash data/download.sh {dataset}
```

The **FactScore** metric described in the paper simply uses the [MultiVerS](https://aclanthology.org/2022.findings-naacl.6/) model trained on the [SciFact dataset](https://aclanthology.org/2020.emnlp-main.609/)). To be able to run it, download the model weights from Wadden et al:

```angular2html
wget -O ~/data_tmp/scifact.ckpt https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/scifact.ckpt
wget -O ~/data_tmp/longformer_large_science.ckpt https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/longformer_large_science.ckpt
```

## Preprocessing

To pre-tokenize the datasets, run:

```angular2html
python preprocess/preprocess.py --dataset {pubmed,clinical,chemistry} --model {primera,t5}
```

(T5 stands for Long-T5.)

# Re-Creating Contrast Sets

`cd corruptions/`

To recreate the datasets, separately run the following scripts `reference.py`, `mask_and_fill.py`, `diverse_decoding.py`, and `entity/swap.py`, before running `merge.py`.

Before running `entity/swap.py`, you must run `entity/bern_entities.py` for chemistry / pubmed, and `entity/stanza_entities.py` for clinical, before running `create_type_inventory.py` for both.

# Training Calibration Models

Calibration defines a set of offline methods for aligning model outputs to quality metrics.  In this codebase, we consider two metrics for summarization: **relevance**, and **faithfulness**.

`cd model/`

## Quickstart

To run different relevance calibration strategies for *further fine-tuning* (FFT):

```angular2html
bash rel_fft.sh {device} {dataset} {sample strategy} {experiment name}
```

To run faithfulness calibration

```angular2html
bash faith_fft.sh {device} {dataset} {sample strategy} {experiment name}
```

As of now, dataset must be one of `chemistry, pubmed, clinical`

Example runs are:

```angular2html
bash rel_fft.sh 0 chemistry random random_chemistry_relevance_fft
bash faith_rel_fft.sh 0 chemistry random random_chemistry_faithful_fft
```

Depending on whether you are calibrating for relevance [R] or faithfulness [F], assign {sample strategy} according to the below. The left-hand-side (LHS) represents the strategy as defined in Figure 1 of the paper and the RHS is the strategy name in the codebase.

![Contrast Sampling Strategies](https://github.com/griff4692/calibrating-summaries/blob/master/images/Faithful_Contrast_Experiments.png)

### Relevance [R] Strategies

**Random**
```angular2html
Random -> random
```

**Quality Metric Based**

```angular2html
Extreme -> max_margin
Average -> min_margin
Min -> min_metric
Max -> max_metric
```

**Margin-Based**

```angular2html
Max Margin -> max_gap
Min Margin -> min_gap
```

**Diversity-Based**

```angular2html
Max Diversity -> max_diversity
Min Diversity -> min_diversity
```

**Likelihood-Based**

```angular2html
Top Beams -> top_beam
Bottom Beams -> bottom_beam
Extreme Beams -> wide_beam
```

**Spurious Correlates**

```angular2html
Long -> max_length
Short -> min_length
```

### Faithfulness [F] Strategies

**Random**
```angular2html
Random -> random
```

**Quality Metric Based**

```angular2html
Average -> avg_margin
```

**Margin-Based**

```angular2html
Max Margin -> max_margin
Min Margin -> min_margin
```

**Diversity-Based**

```angular2html
Max Diversity -> max_diversity
Min Diversity -> min_diversity
```

**Likelihood-Based**

```angular2html
Hard -> hard
Easy -> easy
```

**Spurious Correlates**

```angular2html
Max Extractive Gap -> max_extractive_gap
Min Extractive Gap -> min_extractive_gap
```

The code for each sampling method can be found in the newly defined class **DataCollatorForContrastSeq2Seq** under `transformers/src/transformers/data/data_collator.py`.

New sampling strategies can be built using this class as well.

## Customization

Those scripts will run with the default hyper-parameters and objective functions as described in the paper. Yet, there is more flexibility.

To run the contrastive FFT script directly, run `python run.py -contrast` and directly adjust hyper-parameters.

### Metrics

To choose the metrics by which contrast sets will be ordered and selected, set `--contrast_metrics` to either `relevance` or `faithful`.  Relevance will compute the normalized aggregation of relevance metrics (BertScore, Rouge1, Rouge2) defined in the paper as Rel<sub>Agg</sub>. 
 Faithful will compute the normalized aggregation of faithful metrics (BertScore, FactScore, BARTScore) defined in the paper as Faith<sub>Agg</sub>.

To create custom groupings of metrics, simply choose from the list and separate by `,`

```
CONTRAST_METRIC_LIBRARY = {
    'rouge1',
    'rouge2',
    'rougeL',
    'rougeLsum',
    'coverage',
    'density',
    'compression',
    'bs_src_recall',
    'bs_src_precision',
    'bs_src_f1',
    'bs_ref_recall',
    'bs_ref_precision',
    'bs_ref_f1',
    'bart_score',
    'fact_score',
    'num_prediction_tokens'
}
```

An example is `--contrast_metrics coverage,num_prediction_tokens`. This would optimize for longer, more extractive summaries. 

### Objectives

To choose the contrastive objective, set `--contrast_objective` to one of `unlikelihood`, `margin_rank`, `contrast`, `positive_distillation`.

In the paper,

```angular2html
Relevance FFT Objective -> margin_rank
Faithful FFT Objective -> contrast
```

But they can be mixed and matched freely. `unlikelihood` optimizes negative sets or the bottom half of relevance rank sets with [unlikelihood](https://arxiv.org/abs/1908.04319). `positive_distillation` takes the most positive (as defined by quality metric of choice) and trains the model with standard maximum likelihood. It represents a distillation of positive examples and is quicker to train and can be used when references are highly noisy.

###  Methods

To control which corruption methods are eligible to be selected by each sampling strategy, please change the following flags from their defaults:

```angular2html
positive_methods -> all
mixed_methods -> all
negative_methods -> all
reference_status -> positive
```

**Positive Methods**: These control which methods are used for forming positive sets for faithfulness. The options are `paraphrase` and `reference`.

**Mixed Methods**: `diverse_decoding_primera_ft_{dataset}`, `diverse_decoding_long_t5_ft_{dataset}`. Dataset should be one of `chemistry`, `pubmed`, `clinical`.

**Negative Methods**: These are corruption methods: `mask_and_fill`, and `intrinsic_swap`, `extrinsic_swap`.

**Reference Status**: Defines how to deal with the reference summary for faithfulness calibration. As of now, references are note used for relevance calibration but could be added. 

```angular2html
remove -> Never use references as a positive example
ensure -> Ensure reference is included in every positive subset
positive [Default] -> Treat the reference just like the other positive examples (paraphrases).
```

# Using and Evaluating Calibration Models

## Inference

Run `python model/inference.py --experiment {experiment} --dataset {dataset}` with the same values for `{experiment}` and `{dataset}` used for training.

At the end of the script it will save to a csv file and log the name.

## Metrics

Run `bash eval/run_all.sh {dataset} {path-to-csv} {metrics}`

where `{path-to-csv}` is from above and metrics is one of `{all,faithful,relevance}`.
 
# Citation

If you use this codebase for your research, please cite:

```angular2html
@article{adams-et-al-desired-2023,
  title={What are the Desired Characteristics of Calibration Sets? Identifying Correlates on Long Form Scientific Summarization},
  author={Adams, Griffin and Nguyen, Bichlien and Smith, Jake and Xia, Yingce and Xie, Shufang and Ostropolets, Anna and Deb, Budhaditya and Frost, Kali and Chen, Yuan-Jyue and Naumann, Tristan and others}
}
```