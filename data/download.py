import os
import subprocess

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Fine-Tuned model checkpoints and contrast sets')
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])

    args = parser.parse_args()

    dataset = args.dataset
    name = 'todo'

    url = f'https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/{name}.ckpt'
    out_file = os.path.expanduser(f'~/data_tmp/{name}.ckpt')
    cmd = ['wget', '-O', out_file, url]

    if not os.path.exists(out_file):
        subprocess.run(cmd)
