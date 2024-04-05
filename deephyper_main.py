import collections
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from utils import scaffold_split, _collate, _collate_drp, _collate_CDR
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
from candle_train import get_data_loader
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

class IMPROVE_Dataset(Dataset):
    def __init__(self, drug_dict, cell_dict, IC, edge_index):
        super(IMPROVE_Dataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['improve_chem_id']
        self.Cell_line_name = IC['improve_sample_id']
        # self.value = IC['ic50']
        self.value = IC['auc']
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        self.cell[self.Cell_line_name[index]].edge_index = self.edge_index
        # self.cell[self.Cell_line_name[index]].adj_t = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


def get_data_loader_TGSA(IC, drug_dict, cell_dict, edge_index, batch_size):
    train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)
    Dataset = IMPROVE_Dataset
    collate_fn = _collate
    train_dataset = Dataset(drug_dict, cell_dict, train_set, edge_index=edge_index)
    val_dataset = Dataset(drug_dict, cell_dict, val_set, edge_index=edge_index)
    test_dataset = Dataset(drug_dict, cell_dict, test_set, edge_index=edge_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    return train_loader, val_loader, test_loader

def get_predefine_cluster(edge_index, save_fn, selected_gene_num, device):
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
def run(config: dict) -> float:
    default_config = {}
    default_config["layer_drug"] = 3
    default_config["dim_drug"] = 128
    default_config["hidden_dim"] = 8
    default_config["cell_feature_num"] = 3
    default_config["layer"] = 3
    default_config["dim_cell"] = 8
    # default_config["dropout_ratio"] = 0.2
    default_config["patience"] = 3
    default_config["device"] = 'cuda:1'
    default_config["weight_path"] = 0
    default_config["output_root_dir"] = ""
    default_config["epochs"] = 300
    default_config["weight_decay"]=0
    data_root_dir = "/homes/ac.jjiang/data_dir/TGSA"

    epochs = default_config["epochs"]
    patience = default_config["patience"]
    # device = default_config["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_decay = default_config["weight_decay"]
    weight_path = default_config["weight_path"]
    output_root_dir = default_config["output_root_dir"]

    default_config["dropout_ratio"] = config["dropout_ratio"]
    default_config["batch_size"] = config["batch_size"]

    batch_size = config["batch_size"]
    lr = config["lr"]

    pretrain = True
    fp = open(os.path.join(improve_globals.main_data_dir, "drug_feature_graph.pkl"), "rb")
    drug_dict = pickle.load(fp)
    fp.close()

    edge_index = np.load(os.path.join(improve_globals.main_data_dir, 'edge_index_PPI_0.95.npy'))

    fp = open(os.path.join(improve_globals.main_data_dir, "cell_feature_all.pkl"), "rb")
    cell_dict = pickle.load(fp)
    fp.close()

    fp = open(os.path.join(improve_globals.main_data_dir, "selected_gen_PPI_0.95.pkl"), "rb")
    selected_genes = pickle.load(fp)
    fp.close()
    IC = pd.read_csv(os.path.join(improve_globals.main_data_dir, 'drug_response_with_IC50.csv'), sep=",")

    train_loader, val_loader, test_loader = get_data_loader_TGSA(IC, drug_dict, cell_dict, edge_index, batch_size)
    print("All data: %d, training: %d, validation: %d, testing: %d " % (
    len(IC), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    print('mean degree of gene graph:{}'.format(len(edge_index[0]) / len(selected_genes)))

    predefine_cluster_fn = os.path.join(data_root_dir, 'cluster_predefine_PPI_0.95.npy')
    cluster_predefine = get_predefine_cluster(edge_index, predefine_cluster_fn, len(selected_genes), device)

    model = TGDRP(cluster_predefine, default_config)
    model.to(device)

    train_start = time.time()

    # if pretrain and weight_path != '':
    #     pretrain_model_fn = os.path.join(output_root_dir, 'model_pretrain', '{}.pth'.format(weight_path))
    #     model.GNN_drug.load_state_dict(torch.load(pretrain_model_fn)['model_state_dict'])

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_folder = os.path.join(output_root_dir, "logs", model._get_name())
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    dt = datetime.datetime.now()
    model_save_path = os.path.join(output_root_dir, "trained_model")
    model_fn = os.path.join(model_save_path,
                            model._get_name() + "_{}_{:02d}-{:02d}-{:02d}.pth".format(dt.date(), dt.hour, dt.minute,
                                                                                      dt.second))

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_rmse = float('inf')  # Start with a high value since we want to minimize RMSE
    stopper = EarlyStopping(mode='lower', patience=patience, filename=model_fn)
    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_loss = train(model, train_loader, criterion, opt, device)
        fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)

        print('Evaluating...')
        rmse, MAE, r2, r = validate(model, val_loader, device)
        if rmse < best_rmse:  # Check if current RMSE is better than the best
            best_rmse = rmse  # Update the best RMSE
        print("Validation rmse:{}".format(rmse))
        fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

        early_stop = stopper.step(rmse, model)

        if early_stop:
            break

    print('EarlyStopping! Finish training!')

    train_end = time.time()
    train_total_time = train_end - train_start
    print("Training time: %s s \n" % str(train_total_time))
    print("\nIMPROVE_RESULT:\t{}\n".format(best_rmse))
    return best_rmse


if __name__ == "__main__":


    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.1, 0.7, "log-uniform"), "dropout_ratio")  #
    problem.add_hyperparameter([8, 16, 32, 64, 128], "batch_size")
    problem.add_hyperparameter((0.00001, 0.01, "log-uniform"), "lr")

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define your search and execute it
    search = CBO(problem, evaluator, random_state=42)

    results = search.search(max_evals=100)
    print(results)


