import os
import csv
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


folder = "data/"
#folder = ""
folder = "ap_data/"


def create_lc_sets(split_type: str="known"):
    """ Creates subsets for training, val, and test. """
    IC = pd.read_csv('./data/PANCANCER_IC_82833_580_170.csv')

    import pdb; pdb.set_trace()
    if split_type == 'known':
        # pass
        train_set, val_test_set = train_test_split(IC,
                                                   test_size=0.2,
                                                   random_state=42,
                                                   stratify=IC['Cell line name'])
        val_set, test_set = train_test_split(val_test_set,
                                             test_size=0.5,
                                             random_state=42,
                                             stratify=val_test_set['Cell line name'])

    elif split_type == 'leave_drug_out':
        pass
        ## scaffold
        # smiles_list = pd.read_csv('./data/IC50_GDSC/drug_smiles.csv')[['CanonicalSMILES', 'drug_name']]
        # train_set, val_set, test_set = scaffold_split(IC,
        #                                               smiles_list,
        #                                               seed=42)

    elif split_type == 'leave_cell_out':
        pass
        ## stratify
        # cell_info = IC[['Tissue', 'Cell line name']].drop_duplicates()
        # train_cell, val_test_cell = train_test_split(cell_info,
        #                                              stratify=cell_info['Tissue'],
        #                                              test_size=0.4,
        #                                              random_state=42)
        # val_cell, test_cell = train_test_split(val_test_cell,
        #                                        stratify=val_test_cell['Tissue'],
        #                                        test_size=0.5,
        #                                        random_state=42)

        # train_set = IC[IC['Cell line name'].isin(train_cell['Cell line name'])]
        # val_set = IC[IC['Cell line name'].isin(val_cell['Cell line name'])]
        # test_set = IC[IC['Cell line name'].isin(test_cell['Cell line name'])]

    fdir = os.path.dirname(os.path.abspath(__file__))
    lc_dir = Path(os.path.join(fdir, "lc_data"))
    os.makedirs(lc_dir, exist_ok=True)

    # df = IC.copy()
    # import pdb; pdb.set_trace()
    if split_type == "known":
        # outdir = lc_dir/"mix_drug_cell"
        outdir = lc_dir/"split_known"
        os.makedirs(outdir, exist_ok=True)

        # Define vars that determine train, val, and test sizes
        # size = int(len(df) * 0.8)
        # size1 = int(len(df) * 0.9)

        # df_tr = df[:size]
        # df_vl = df[size:size1]
        # df_te = df[size1:]

        df_tr = train_set.copy()
        df_vl = val_set.copy()
        df_te = test_set.copy()

        df_tr.to_csv(outdir/"train_data.csv", index=False)
        df_vl.to_csv(outdir/"val_data.csv", index=False)
        df_te.to_csv(outdir/"test_data.csv", index=False)

        df = pd.concat([df_tr, df_vl, df_te], axis=0)
        df.to_csv(outdir/"drug_cell_rsp.csv", index=False)

    elif split_type == "drug":
        df_tr = None
        df_vl = None
        df_te = None

    elif split_type == "cell":
        df_tr = None
        df_vl = None
        df_te = None

    else:
        raise ValueError

    # --------------------------------------------

    # Save splits
    # lc_init_args = {'cv_lists': cv_lists,
    #                 'lc_step_scale': args['lc_step_scale'],
    #                 'lc_sizes': args['lc_sizes'],
    #                 'min_size': args['min_size'],
    #                 'max_size': args['max_size'],
    #                 'lc_sizes_arr': args['lc_sizes_arr'],
    #                 'print_fn': print
    #                 }

    lc_init_args = {'cv_lists': None,
                    'lc_step_scale': "log",
                    'lc_sizes': 7,
                    'min_size': None,
                    'max_size': None,
                    'lc_sizes_arr': None,
                    'print_fn': print
                    }

    lc_step_scale = "log"
    lc_sizes = 7
    min_size = 1024
    max_size = df_tr.shape[0]
    # from learningcurve.lrn_crv import LearningCurve
    # lc_obj = LearningCurve(X=None, Y=None, meta=None, **lc_init_args)
    pw = np.linspace(0, lc_sizes-1, num=lc_sizes) / (lc_sizes-1)
    m = min_size * (max_size/min_size) ** pw
    m = np.array([int(i) for i in m])  # cast to int
    lc_sizes = m

    # LC subsets
    for i, sz in enumerate(lc_sizes):
        aa = df_tr[:sz]
        aa.to_csv(outdir/f"train_sz_{i+1}.csv", index=False)
    return None


if __name__ == "__main__":
    # import candle
    fdir = Path(__file__).resolve().parent

    # ftp_fname = fdir/"ftp_file_list"
    # with open(ftp_fname, "r") as f:
    #     data_file_list = f.readlines()

    # ftp_origin = "https://ftp.mcs.anl.gov/pub/candle/public/improve/reproducability/GraphDRP/data"
    # for f in data_file_list:
    #     candle.get_file(fname=f.strip(),
    #                     origin=os.path.join(ftp_origin, f.strip()),
    #                     unpack=False, md5_hash=None,
    #                     datadir=fdir/"./data",
    #                     cache_subdir="common")

    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    # parser.add_argument(
    #     '--choice', type=int, required=False, default=0,
    #     help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
    parser.add_argument(
        '--split_type', type=str, required=False, default="known", choices=["known"],
        help="Split type: known, ")
    args = parser.parse_args()

    # import pdb; pdb.set_trace()
    create_lc_sets(split_type=args.split_type)
    # create_lc_sets(split_type="drug")

    # import pdb; pdb.set_trace()
    # create_lc_datasets(split_type=args.split_type)
