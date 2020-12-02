CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode learned_mixin_baseline --seed 111 --output_dir ../experiments_fever_bow/bert_learned_mixin_lr2_epoch3_seed111_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode learned_mixin_baseline --seed 222 --output_dir ../experiments_fever_bow/bert_learned_mixin_lr2_epoch3_seed222_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode learned_mixin_baseline --seed 333 --output_dir ../experiments_fever_bow/bert_learned_mixin_lr2_epoch3_seed333_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode learned_mixin_baseline --seed 444 --output_dir ../experiments_fever_bow/bert_learned_mixin_lr2_epoch3_seed444_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode learned_mixin_baseline --seed 555 --output_dir ../experiments_fever_bow/bert_learned_mixin_lr2_epoch3_seed555_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode bias_product_baseline --seed 111 --output_dir ../experiments_fever_bow/bert_bias_product_lr2_epoch3_seed111_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode bias_product_baseline --seed 222 --output_dir ../experiments_fever_bow/bert_bias_product_lr2_epoch3_seed222_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode bias_product_baseline --seed 333 --output_dir ../experiments_fever_bow/bert_bias_product_lr2_epoch3_seed333_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode bias_product_baseline --seed 444 --output_dir ../experiments_fever_bow/bert_bias_product_lr2_epoch3_seed444_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5
#
#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode bias_product_baseline --seed 555 --output_dir ../experiments_fever_bow/bert_bias_product_lr2_epoch3_seed555_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 111 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed111_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 222 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed222_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 333 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed333_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 444 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed444_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=15 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 555 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed555_reproduce/ --which_bias fever_claim_only_bow_reproduce --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 444 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed444/ --which_bias fever_claim_only_bow --learning_rate 2e-5

#CUDA_VISIBLE_DEVICES=5 python train_fever_distill.py --do_train --do_eval --mode smoothed_distill --seed 555 --output_dir ../experiments_fever_bow/bert_smoothed_distill_lr2_epoch3_seed555/ --which_bias fever_claim_only_bow --learning_rate 2e-5
