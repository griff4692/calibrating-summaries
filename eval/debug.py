import pandas as pd

import ujson
from scipy.stats import pearsonr
import os


def add_len(df):
    df['pred_len'] = df['prediction'].apply(lambda x: len(x.split(' ')))


if __name__ == '__main__':
    EXP = 'margin_max_div_rel'
    MAX = 'clinical_fft_margin_max_margin_rel'
    MIN = 'clinical_fft_margin_min_margin_rel'

    MAX_LIKE = 'clinical_unlikelihood_max_margin_relevance'
    MIN_LIKE = 'clinical_unlikelihood_min_margin_relevance'

    EXPS = [
        EXP
    ]

    form = os.path.expanduser('~/data_tmp/weights/{}')

    outputs = []
    x, y = [], []
    for exp in EXPS:
        weight = form.format(exp)
        for step in range(1000, 11000, 1000):
            val_fn = os.path.join(weight, 'results_' + str(step) + '_steps.json')
            row = {'experiment': exp, 'step': step}
            try:
                with open(val_fn, 'r') as fd:
                    metrics = ujson.load(fd)
                row['val_rouge1_a'] = metrics['eval_rouge1']
                row['val_rouge2_a'] = metrics['eval_rouge2']
            except:
                print(val_fn + ' does not exist')

            ckpt_dir = os.path.join(weight, f'ckpt_{step}_steps')

            val_pred_fn = os.path.join(ckpt_dir, 'validation_predictions.csv')
            if os.path.exists(val_pred_fn):
                val_pred = pd.read_csv(val_pred_fn)
                val_pred_metrics = pd.read_csv(os.path.join(ckpt_dir, 'validation_predictions_with_metrics.csv'))
                add_len(val_pred)
                add_len(val_pred_metrics)

                x.append(val_pred_metrics.pred_len.mean())
                y.append(val_pred_metrics.rouge1.mean())

                row['val_rouge1_b'] = val_pred.rouge1.mean()
                row['val_rouge2_b'] = val_pred.rouge2.mean()
                row['val_rouge1'] = val_pred_metrics.rouge1.mean()
                row['val_rouge2'] = val_pred_metrics.rouge2.mean()
                row['val_len'] = val_pred_metrics.pred_len.mean()

            else:
                print(val_pred_fn + ' does not exist')

            test_fn = os.path.join(ckpt_dir, 'test_predictions.csv')
            test_fn_metrics = os.path.join(ckpt_dir, 'test_predictions_with_metrics.csv')
            if os.path.exists(test_fn_metrics):
                test_pred = pd.read_csv(test_fn)
                test_pred_metrics = pd.read_csv(test_fn_metrics)
                add_len(test_pred)
                add_len(test_pred_metrics)

                row['test_rouge1_a'] = test_pred.rouge1.mean()
                row['test_rouge2_a'] = test_pred.rouge2.mean()

                row['test_rouge1'] = test_pred_metrics.rouge1.mean()
                row['test_rouge2'] = test_pred_metrics.rouge2.mean()
                row['test_len'] = test_pred_metrics.pred_len.mean()

            outputs.append(row)

    outputs = pd.DataFrame(outputs)

    print(pearsonr(x, y))

    for exp in EXPS:
        sub = outputs[outputs['experiment'] == exp]
        print(exp)
        for col in sub.columns:
            if col == 'experiment':
                continue
            v = sub[col].dropna()
            if len(v) == 0:
                continue
            print('\t' + col, round(min(v), 2), round(v.mean(), 2), round(v.max(), 2))
        print('\n\n')
