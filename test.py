import glob
import sys, os
sys.path.append(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
from my_improve_utils import load_IMPROVE_pre_split_data
from utils import set_random_seed
from utils import train, validate
from models.TGDRP_candle import TGDRP, TGDRP_INIT
from models import TGDRP_candle
import pickle
import argparse
import fitlog
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
import time
import datetime
from benchmark_dataset_generator.improve_utils import *

import candle

def initialize_parameters():
   _common = TGDRP_INIT(TGDRP_candle.file_path,
                          'candle_TGSA_params.txt',
                          'pytorch',
                          prog='TGSA',
                          desc='TGSA candle')

   # Initialize parameters
   gParams = candle.finalize_parameters(_common)
   return gParams

def get_data_loader_pre_split(edge, batch_size):
    fp = open(os.path.join(improve_globals.main_data_dir, "drug_feature_graph.pkl"), "rb")
    drug_dict = pickle.load(fp)
    fp.close()

    edge_index = np.load(os.path.join(improve_globals.main_data_dir, 'edge_index_PPI_{}.npy'.format(edge)))

    fp = open(os.path.join(improve_globals.main_data_dir, "cell_feature_all.pkl"), "rb")
    cell_dict = pickle.load(fp)
    fp.close()

    fp = open(os.path.join(improve_globals.main_data_dir, "selected_gen_PPI_0.95.pkl"), "rb")
    selected_genes = pickle.load(fp)
    fp.close()
    IC = pd.read_csv(os.path.join(improve_globals.main_data_dir, 'drug_response_with_IC50.csv'), sep=",")
    # rs_df_split = load_single_drug_response_data("CCLE", split=1, split_type=["train", "val", "test"], y_col_name='auc')
    #rs_df_split = load_single_drug_response_data("CCLE", split=1, split_type=["test"], y_col_name='auc')
    #rs_df_split = load_single_drug_response_data("gCSI", split=1, split_type=["test"], y_col_name='auc')
    #rs_df_split = load_single_drug_response_data("GDSCv1", split=1, split_type=["test"], y_col_name='auc')
    rs_df_split = load_single_drug_response_data("GDSCv2", split=1, split_type=["test"], y_col_name='auc')

    test_loader = load_IMPROVE_pre_split_data(rs_df_split, drug_dict, cell_dict, edge_index, batch_size)
    # train_loader, val_loader, test_loader = load_IMPROVE_data(IC, drug_dict, cell_dict, edge_index, setup, model,
    #                                                           batch_size)

    print("testing: %d " % (len(test_loader.dataset)))
    print('mean degree of gene graph:{}'.format(len(edge_index[0]) / len(selected_genes)))
    return test_loader, edge_index, selected_genes

def get_predefine_cluster(edge_index, save_fn, selected_gene_num, thresh, device):
    if not os.path.exists(save_fn):
        g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), x=torch.zeros(selected_gene_num, 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            print(len(cluster.unique()))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        np.save(save_fn, cluster_predefine)
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    else:
        cluster_predefine = np.load(save_fn, allow_pickle=True).item()
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}

    return cluster_predefine


def run(gParameters):
    edge = gParameters["edge"] # threshold of edge
    device = gParameters["device"] # 'cuda:1'
    model = gParameters["model"] #'TGDRP'
    batch_size = gParameters["batch_size"] #128
    lr = gParameters["lr"] # 0.0001
    weight_decay = gParameters["weight_decay"] #0
    epochs = gParameters["epochs"] #300
    patience = gParameters["patience"] #3
    # setup = gParameters["setup"] #'known'
    setup = 'pre-split'  # 'known'
    pretrain = gParameters["pretrain"] #1
    weight_path = gParameters["weight_path"] #''
    mode = gParameters["mode"] #'train'

    dropout_ratio = gParameters["dropout_ratio"]  # 0.2
    seed = gParameters["seed"]  # 42
    layer_drug = gParameters["layer_drug"]  # 3
    dim_drug = gParameters["dim_drug"]  # 128
    cell_feature_num = gParameters["cell_feature_num"]  # 3
    layer = gParameters["layer"]  # 3
    hidden_dim = gParameters["hidden_dim"]  # 8

    ############################################
    # improve_globals.DATASET = "Pilot1"  # Yitan's dataset
    # improve_globals.DATASET = "Benchmark"  # Alex's dataset
    data_root_dir = improve_globals.main_data_dir
    if improve_globals.DATASET == "Pilot1":
        print("Training on Pilot1 dataset")
        output_root_dir = os.path.join(improve_globals.data_root_dir, "TGSA_output_pilot1")
    else:
        print("Training on Benchmark dataset")
        output_root_dir = os.path.join(improve_globals.data_root_dir, "TGSA_output")

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    # load_single_drug_response_data(
    #     # source: Union[str, List[str]],
    #     source: str,
    # split: Union[int, None] = None,
    # split_type: Union[str, List[str], None] = None,
    # y_col_name: str = "auc",
    # sep: str = "\t",
    # verbose: bool = True) -> pd.DataFrame:
    # """
    # Returns datarame with cancer ids, drug ids, and drug response values. Samples
    # from the original drug response file are filtered based on the specified
    # sources.
    #
    # Args:
    #     source (str or list of str): DRP source name (str) or multiple sources (list of strings)
    #     split(int or None): split id (int), None (load all samples)
    #     split_type (str or None): one of the following: 'train', 'val', 'test'
    #     y_col_name (str): name of drug response measure/score (e.g., AUC, IC50)
    #
    # Returns:
    #     pd.Dataframe: dataframe that contains drug response values
    # """


    test_loader, edge_index, selected_genes = get_data_loader_pre_split(edge, batch_size)

    predefine_cluster_fn = os.path.join(data_root_dir, 'cluster_predefine_PPI_{}.npy'.format(edge))
    cluster_predefine = get_predefine_cluster(edge_index, predefine_cluster_fn, len(selected_genes), edge,
                                              device)

    model = TGDRP(cluster_predefine, gParameters)
    # model = nn.DataParallel(model)  # TODO: use all available GPUs
    model.to(device)


    print("Running in testing mode")
    test_start = time.time()
    weight = "TGDRP_pre" if pretrain else "TGDRP"

    pth_fn = os.path.join(output_root_dir, 'trained_model', '{}.pth'.format(weight))
    if not os.path.exists(pth_fn):
        pth_dir = os.path.join(output_root_dir, 'trained_model')
        list_of_files = glob.glob(os.path.join(pth_dir, "*.pth"))
        latest_file = max(list_of_files, key=os.path.getctime)  # get the newest file
        pth_fn = os.path.join(pth_dir, latest_file)
    model.load_state_dict(torch.load(pth_fn, map_location=device)['model_state_dict'])
    test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, device)
    print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 4), round(test_MAE, 4),
                                                         round(test_r2, 4), round(test_r, 4)))
    test_end = time.time()
    test_total_time = test_end - test_start
    print("Testing time:%s s \n s" % str(test_total_time))

def main():
    gParams = initialize_parameters()
    run(gParams)

if __name__ == "__main__":
    main()
