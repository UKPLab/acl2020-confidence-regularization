#CUDA_VISIBLE_DEVICES=15 python train_qqp_distill.py --output_dir ../experiments_qqp_paws_shallow/paws0_bert_smoothed_distill_lr5_epoch3_seed111 --do_train --do_eval --mode smoothed_distill --seed 111 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0
#
#CUDA_VISIBLE_DEVICES=15 python train_qqp_distill.py --output_dir ../experiments_qqp_paws_shallow/paws0_bert_smoothed_distill_lr5_epoch3_seed222 --do_train --do_eval --mode smoothed_distill --seed 222 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0
#
#CUDA_VISIBLE_DEVICES=15 python train_qqp_distill.py --output_dir ../experiments_qqp_paws_shallow/paws0_bert_smoothed_distill_lr5_epoch3_seed333 --do_train --do_eval --mode smoothed_distill --seed 333 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0


#CUDA_VISIBLE_DEVICES=8 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_smoothed_distill_lr2_epoch3_seed111 --do_train --do_eval --mode smoothed_distill --seed 111 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=10 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_smoothed_distill_lr2_epoch3_seed222 --do_train --do_eval --mode smoothed_distill --seed 222 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=14 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_smoothed_distill_lr2_epoch3_seed333 --do_train --do_eval --mode smoothed_distill --seed 333 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=14 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_smoothed_distill_lr2_epoch3_seed444 --do_train --do_eval --mode smoothed_distill --seed 444 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=15 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_smoothed_distill_lr2_epoch3_seed555 --do_train --do_eval --mode smoothed_distill --seed 555 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5


#CUDA_VISIBLE_DEVICES=8 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_bias_product_lr2_epoch3_seed111 --do_train --do_eval --mode bias_product_baseline --seed 111 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=9 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_bias_product_lr2_epoch3_seed222 --do_train --do_eval --mode bias_product_baseline --seed 222 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=10 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_bias_product_lr2_epoch3_seed333 --do_train --do_eval --mode bias_product_baseline --seed 333 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=14 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_bias_product_lr2_epoch3_seed444 --do_train --do_eval --mode bias_product_baseline --seed 444 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=14 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_bias_product_lr2_epoch3_seed555 --do_train --do_eval --mode bias_product_baseline --seed 555 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5

CUDA_VISIBLE_DEVICES=6 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_learned_mixin_lr2_epoch3_seed111 --do_train --do_eval --mode learned_mixin_baseline --seed 111 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5 --penalty 0.01

CUDA_VISIBLE_DEVICES=7 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_learned_mixin_lr2_epoch3_seed222 --do_train --do_eval --mode learned_mixin_baseline --seed 222 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5 --penalty 0.01

CUDA_VISIBLE_DEVICES=8 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_learned_mixin_lr2_epoch3_seed333 --do_train --do_eval --mode learned_mixin_baseline --seed 333 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5 --penalty 0.01

CUDA_VISIBLE_DEVICES=10 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_learned_mixin_lr2_epoch3_seed444 --do_train --do_eval --mode learned_mixin_baseline --seed 444 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5 --penalty 0.01

CUDA_VISIBLE_DEVICES=15 screen -d -m python train_qqp_distill.py --output_dir ../experiments_qqp_paws_reproduce/paws0_bert_learned_mixin_lr2_epoch3_seed555 --do_train --do_eval --mode learned_mixin_baseline --seed 555 --qqp_dataset qqp_paws --which_bias qqp_hans_json --paws_num 0 --learning_rate 2e-5 --penalty 0.01