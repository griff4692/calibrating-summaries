# Background

This is the official PyTorch codebase for the ACL 2023 paper: [What are the Desired Characteristics of Calibration Sets? Identifying Correlates on Long Form Scientific Summarization](https://openreview.net/pdf?id=bIC0BfWzCs).

This repository provides an adaptable toolkit for constructing, selecting, and optimizing contrast sets for relevance and faithfulness calibration across three long-form scientific summariation datasets (spanning Chemistry, Clinical, and Biomedical). 

It can be adapted for other summarization datasets in other domains as well.

Please feel free to submit pull requests or raise issues on the tracker. The code will be actively maintained by the authors.

# Code Setup

```angular2html
pip install -e .
cd transformers && pip install -e .
python -m nltk.downloader stopwords
mkdir ~/data_tmp
```

# Creating Contrast Sets

xx

# Training Calibration Models

Calibration defines a set of offline methods for aligning model outputs to quality metrics.  In this codebase, we consider two metrics for summarization: **relevance**, and **faithfulness**.

## Setup

`cd model/`

## Quickstart

To run faithfulness calibration

```angular2html
bash faith_fft.sh {dataset} {sample strategy} {experiment name}
```

Depending on whether you are calibrating for relevance [R] or faithfulness [F], assign {sample strategy} according to the below. The left-hand-side (LHS) represents the strategy as defined in Figure 1 of the paper and the RHS is the strategy name in the codebase.

![Contrast Sampling Strategies](/images/Faithful_Contrast_Experiments.png)


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

# Citation

If you use this codebase for your research, please cite:

```angular2html
@article{adamsdesired,
  title={What are the Desired Characteristics of Calibration Sets? Identifying Correlates on Long Form Scientific Summarization},
  author={Adams, Griffin and Nguyen, Bichlien and Smith, Jake and Xia, Yingce and Xie, Shufang and Ostropolets, Anna and Deb, Budhaditya and Frost, Kali and Chen, Yuan-Jyue and Naumann, Tristan and others}
}
```