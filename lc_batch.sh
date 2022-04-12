#!/bin/bash --login

# Generate required datasets:
# choice:
#     0: create mixed test dataset
#     1: create saliency map dataset
#     2: create blind drug dataset
#     3: create blind cell dataset
# python preprocess.py --choice 0
# python preprocess.py --choice 1
# python preprocess.py --choice 2
# python preprocess.py --choice 3

# ep=2
ep=300

tr_file=train_sz_1
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known

tr_file=train_sz_2
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known

tr_file=train_sz_3
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known

tr_file=train_sz_4
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known

tr_file=train_sz_5
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known

tr_file=train_sz_6
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known

tr_file=train_sz_7
python main.py --mode train --device cuda:0 --epochs $ep --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --lc_dpath lc_data/split_known
