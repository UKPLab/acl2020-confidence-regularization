# Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance
> **Abstract:** Models for natural language understanding (NLU) tasks often rely on the idiosyncratic biases of the dataset, which make them brittle against test cases outside the training distribution. 
Recently, several proposed debiasing methods are shown to be very effective in improving out-of-distribution performance. However, their improvements come at the expense of performance drop when models are evaluated on the in-distribution data, which contain examples with higher diversity. 
This seemingly inevitable trade-off may not tell us much about the changes in the reasoning and understanding capabilities of the resulting models on broader types of examples beyond the small subset represented in the out-of-distribution data.
In this paper, we address this trade-off by introducing a novel debiasing method, called confidence 
regularization, which discourage models from exploiting biases while enabling them to receive enough incentive to learn from all the training examples. We evaluate our method on three NLU tasks and show that, in contrast to its predecessors, it improves the performance on out-of-distribution datasets (e.g., 7pp gain on HANS dataset) while maintaining the original in-distribution accuracy.


The repository contains the code to reproduce our work in debiasing NLU models without in-distribution degradation.
We provide 2 runs of experiment that are shown in our paper:
1. Debias [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) model from syntactic bias and evaluate on 
[HANS](https://arxiv.org/abs/1902.01007) as the out-of-distribution data.
2. Debias [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) model from hypothesis-only bias and evaluate
 on [MNLI-hard](https://arxiv.org/abs/1902.01007) sets as the out-of-distribution data. 

## Requirements
The code requires python >= 3.6 and pytorch >= 1.1.0.

Additional required dependencies can be found in `requirements.txt`.
Install all requirements by running:
```bash
pip install -r requirements.txt
```


## Data
Our experiments use MNLI dataset version provided by GLUE benchmark.
Download the file from <a href="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce" target="_blank">here</a>, 
and unzip under the directory ``./dataset`` 
Additionally download the following files <a href="https://drive.google.com/drive/folders/1PaWhpJRqPvipsTc1Xp8juISHkYBSOzcL" target="_blank">here</a>
for evaluating on hard/easy splits of both MNLI dev and test sets.
The dataset directory should be structured as the following:
```bash
└── dataset 
    └── MNLI
        ├── train.tsv
        ├── dev_matched.tsv
        ├── dev_mismatched.tsv
        ├── dev_mismatched.tsv
        ├── dev_matched_easy.tsv
        ├── dev_matched_hard.tsv
        ├── dev_mismatched_easy.tsv
        ├── dev_mismatched_hard.tsv
        ├── multinli_hard
        │   ├── multinli_0.9_test_matched_unlabeled_hard.jsonl
        │   └── multinli_0.9_test_mismatched_unlabeled_hard.jsonl
        ├── multinli_test
        │   ├── multinli_0.9_test_matched_unlabeled.jsonl
        │   └── multinli_0.9_test_mismatched_unlabeled.jsonl
        └── original
```


## Running the experiments
For each evaluation setting, use the `--mode` and `which_bias` arguments to
set the appropriate loss function and the type of bias to mitigate (e.g, hans, hypo).

To reproduce our result on MNLI ⮕ HANS, run the following:

```
cd src/
CUDA_VISIBLE_DEVICES=6 python train_distill_bert.py \
    --output_dir ../checkpoints/hans/bert_confreg_lr5_epoch3_seed444 \
    --do_train --do_eval --mode smoothed_distill \
    --seed 444 --which_bias hans
```

For the MNLI ⮕ hard splits, run the following:
```
cd src/
CUDA_VISIBLE_DEVICES=6 python train_distill_bert.py \
    --output_dir ../checkpoints/hypo/bert_confreg_lr5_epoch3_seed444 \
    --do_train --do_eval --mode smoothed_distill \
    --seed 444 --which_bias hypo
```

## Expected results

Results on the MNLI ⮕ HANS setting:

|Mode|Seed|MNLI-m|MNLI-mm|HANS avg.|
|-----|----|---|---|---|
|None|4444|84.57|84.72|62.04|
|conf-reg|444|84.13|84.66|69.53|

Results on the MNLI ⮕ Hard-splits setting:

|Mode|Seed|MNLI-m|MNLI-mm|MNLI-m hard|MNLI-mm hard|
|-----|----|---|---|---|---|
|None|2222|84.62|84.71|76.07|76.75|
|conf-reg|333|85.01|84.87|78.02|78.89|


## Contact
Contact person: [Ajie Utama](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/staff_ukp/detailseite_mitarbeiter_1_71488.en.jsp), utama@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

Please reach out to us for further questions or if you encounter any issue.
You can cite this work by the following:
```
@InProceedings{UtamaDebias2020,
  author    = {Utama, P. Ajie and Moosavi, Nafise Sadat and Gurevych, Iryna},
  title     = {Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance},
  booktitle = {Proceedings of the 58th Conference of the Association for Computational Linguistics},
  month     = jul,
  year      = {2020},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgement
The code in this repository is build on the implementation of debiasing method by Clark et al.
The original version can be found [here](https://github.com/chrisc36/debias) 