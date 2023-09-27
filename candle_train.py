import glob
import sys, os
sys.path.append(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
from my_improve_utils import load_IMPROVE_data, EarlyStopping
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
import json
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

def get_data_loader(edge, setup, model, batch_size):
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

    train_loader, val_loader, test_loader = load_IMPROVE_data(IC, drug_dict, cell_dict, edge_index, setup, model,
                                                              batch_size)

    print("All data: %d, training: %d, validation: %d, testing: %d " % (
    len(IC), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    print('mean degree of gene graph:{}'.format(len(edge_index[0]) / len(selected_genes)))
    return train_loader, val_loader, test_loader, edge_index, selected_genes

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
    setup = gParameters["setup"] #'known'
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

    train_loader, val_loader, test_loader, edge_index, selected_genes = get_data_loader(edge, setup,
                                                                                        model, batch_size)

    predefine_cluster_fn = os.path.join(data_root_dir, 'cluster_predefine_PPI_{}.npy'.format(edge))
    cluster_predefine = get_predefine_cluster(edge_index, predefine_cluster_fn, len(selected_genes), edge,
                                              device)

    model = TGDRP(cluster_predefine, gParameters)
    # model = nn.DataParallel(model)  # TODO: use all available GPUs
    model.to(device)

    if mode == 'train':

        train_start = time.time()

        if pretrain and weight_path != '':
            pretrain_model_fn = os.path.join(output_root_dir, 'model_pretrain', '{}.pth'.format(weight_path))
            model.GNN_drug.load_state_dict(torch.load(pretrain_model_fn)['model_state_dict'])

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        log_folder = os.path.join(output_root_dir, "logs", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(gParameters)
        fitlog.add_hyper_in_file(__file__)

        dt = datetime.datetime.now()
        model_save_path = os.path.join(output_root_dir, "trained_model")
        model_fn = os.path.join(model_save_path, model._get_name() + "_{}_{:02d}-{:02d}-{:02d}.pth".format(dt.date(), dt.hour, dt.minute, dt.second))

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        stopper = EarlyStopping(mode='lower', patience=patience, filename=model_fn)
        for epoch in range(1, epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_loss = train(model, train_loader, criterion, opt, device)
            fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)

            print('Evaluating...')
            rmse, MAE, r2, r = validate(model, val_loader, device)
            print("Validation rmse:{}".format(rmse))
            fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

            early_stop = stopper.step(rmse, model)
            # Supervisor HPO
            print("\nIMPROVE_RESULT val_loss:\t{}\n".format(rmse))
            with open(Path(output_root_dir) / "scores.json", "w", encoding="utf-8") as f:
                json.dump([rmse, MAE, r2, r], f, ensure_ascii=False, indent=4)

            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        train_end = time.time()
        train_total_time = train_end - train_start
        print("Training time: %s s \n" % str(train_total_time))


def main():
    gParams = initialize_parameters()
    run(gParams)

if __name__ == "__main__":
    main()
