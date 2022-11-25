This is the codes, models, and logs of the paper "Diagonal Matrix-to-Sequence Method for Seq2seq".

- dataset: Datasets containing training and testing, download URL: https://drive.google.com/file/d/1IOZgyzY9KGgvGowPiLT_xh3EUZX3obI5/view?usp=sharing

- dumped: Model files, download URL: https://drive.google.com/file/d/1VYXRZyQxuDnWl8wNKdcI_V1AcoVvfbYV/view?usp=sharing
* Only the training model of MIS in the paper is included in the zip package due to memory limitation, the rest of the models can be easily trained by code.

- gen_dataset.py: Code of generate datasets


The following command example can be used to train a model.

python main.py  --exp_name "MIS_16_200000_SMD_256d" \
                --task "MIS"\
                --sorting_type "SMD"\
                --emb_dim 256 \
                --n_enc_layers 6 \
                --n_dec_layers 6 \
                --n_heads 8 \
                --batch_size 256 \
                --epoch_size 120000\
                --stopping_criterion "test_acc,100" \
                --validation_metrics "test_acc" \
                --num_workers 4 \
                --reload_data "dataset/MIS/MIS_16_200000.train,dataset/MIS/MIS_16_200000.test" \
                --reload_model "" \
                --eval_only False \


The following command example can be used to test a model.

python main.py  --sorting_type "SMD" \
                --reload_model "dumped/MIS_16_200000_SMD_256d/dhpe85v8jw/best-test_acc.pth" \
                --reload_data = " ,/public/home/pw/workspace/LearnLawsInMatrix/dataset/eva/MIS/MIS_14_200000.test" \
                --dump_path "eval_log/" \
                --task "MIS" \
                --epoch_size 80000 \
                --eval_only True
