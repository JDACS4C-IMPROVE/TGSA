import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from improve_utils import load_IMPROVE_data, EarlyStopping
from utils import set_random_seed
from utils import train, validate
from models.TGDRP import TGDRP
import pickle
import argparse
import fitlog
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='device')
    parser.add_argument('--model', type=str, default='TGDRP', help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--layer_drug', type=int, default=3, help='layer for drug')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug')
    parser.add_argument('--cell_feature_num', type=int, default=3, help='dimension of cell line features')
    parser.add_argument('--layer', type=int, default=3, help='number of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--edge', type=float, default=0.95, help='threshold for cell line graph')
    parser.add_argument('--setup', type=str, default='known', help='experimental setup')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='whether use pre-trained weights (0 for False, 1 for True')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')
    parser.add_argument('--mode', type=str, default='test',
                        help='train or test')
    return parser.parse_args()


def get_data_loader(edge, setup, model, batch_size):
    fp = open("/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data/drug_feature_graph.pkl",
              "rb")
    drug_dict = pickle.load(fp)
    fp.close()

    edge_index = np.load(
        '/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data/edge_index_PPI_{}.npy'.format(
            edge))

    fp = open("/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data/cell_feature_all.pkl",
              "rb")
    cell_dict = pickle.load(fp)
    fp.close()

    fp = open("/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data/selected_gen_PPI_0.95.pkl",
              "rb")
    selected_genes = pickle.load(fp)
    fp.close()

    # edge_index = np.load('/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data/edge_index_PPI_{}.npy'.format(edge))
    IC = pd.read_csv(
        '/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data/raw_data/y_data/response.txt',
        sep="\t")

    train_loader, val_loader, test_loader = load_IMPROVE_data(IC, drug_dict, cell_dict, edge_index, setup, model,
                                                              batch_size)
    print(len(IC), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print('mean degree:{}'.format(len(edge_index[0]) / len(selected_genes)))
    sample_key = list(cell_dict.keys())[0]
    print(cell_dict[sample_key])
    num_feature = cell_dict[sample_key].x.shape[1]
    return train_loader, val_loader, test_loader, edge_index, selected_genes


'''
# Added by Jun Jiang 
why not use this?
# predefine cluster 
# "https://stringdb-static.org/download/clusters.proteins.v11.5/9606.clusters.proteins.v11.5.txt.gz"
# "https://stringdb-static.org/download/clusters.info.v11.5/9606.clusters.info.v11.5.txt.gz"
'''


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


def main():
    args = arg_parse()

    set_random_seed(args.seed)

    ############################################
    # TODO: for Debug
    args.mode = 'train'   #
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ############################################

    data_root_dir = "/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data"
    output_root_dir = "/infodev1/non-phi-data/junjiang/TGSA_output"
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    train_loader, val_loader, test_loader, edge_index, selected_genes = get_data_loader(args.edge, args.setup,
                                                                                        args.model, args.batch_size)

    predefine_cluster_fn = os.path.join(data_root_dir, 'cluster_predefine_PPI_{}.npy'.format(args.edge))
    cluster_predefine = get_predefine_cluster(edge_index, predefine_cluster_fn, len(selected_genes), args.edge,
                                              args.device)

    model = TGDRP(cluster_predefine, args)
    # model = nn.DataParallel(model)  # use all available GPUs
    model.to(args.device)

    if args.mode == 'train':
        if args.pretrain and args.weight_path != '':
            model.GNN_drug.load_state_dict(
                torch.load(os.path.join(output_root_dir, 'IMPROVE_model_pretrain', '{}.pth'.format(args.weight_path)))['model_state_dict'])

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_folder = os.path.join(os.getcwd(), "logs", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)

        stopper = EarlyStopping(mode='lower', patience=args.patience)
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_loss = train(model, train_loader, criterion, opt, args.device)
            fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)

            print('Evaluating...')
            rmse, _, _, _ = validate(model, val_loader, args.device)
            print("Validation rmse:{}".format(rmse))
            fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

            early_stop = stopper.step(rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)

        train_rmse, train_MAE, train_r2, train_r = validate(model, train_loader, args.device)
        val_rmse, val_MAE, val_r2, val_r = validate(model, val_loader, args.device)
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
        print('Train reslut: rmse:{} r2:{} r:{}'.format(train_rmse, train_r2, train_r))
        print('Val reslut: rmse:{} r2:{} r:{}'.format(val_rmse, val_r2, val_r))
        print('Test reslut: rmse:{} r2:{} r:{}'.format(test_rmse, test_r2, test_r))

        fitlog.add_best_metric(
            {'epoch': epoch - args.patience,
             "train": {'RMSE': train_rmse, 'MAE': train_MAE, 'pearson': train_r, "R2": train_r2},
             "valid": {'RMSE': stopper.best_score, 'MAE': val_MAE, 'pearson': val_r, 'R2': val_r2},
             "test": {'RMSE': test_rmse, 'MAE': test_MAE, 'pearson': test_r, 'R2': test_r2}})

    elif args.mode == 'test':
        weight = "TGDRP_pre" if args.pretrain else "TGDRP"
        model.load_state_dict(
            torch.load(os.path.join(output_root_dir, 'IMPROVE_weights', '{}.pth'.format(weight)), map_location=args.device)['model_state_dict'])
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
        print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 4), round(test_MAE, 4),
                                                             round(test_r2, 4), round(test_r, 4)))


if __name__ == "__main__":
    main()
