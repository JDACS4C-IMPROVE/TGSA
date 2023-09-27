import torch
import pandas as pd
from utils import scaffold_split, _collate, _collate_drp, _collate_CDR
from torch.utils.data import Dataset, DataLoader
import os, datetime
from sklearn.model_selection import train_test_split, KFold
from benchmark_dataset_generator.improve_utils import *

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

class IMPROVE_Dataset_name(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super().__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['improve_chem_id']
        self.Cell_line_name = IC['Cell line name']
        self.value = IC['ic50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


class IMPROVE_Dataset_CDR(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super().__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['improve_chem_id']
        self.Cell_line_name = IC['improve_sample_id']
        self.value = IC['ic50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])

def load_IMPROVE_pre_split_data(rs_df_split, drug_dict, cell_dict, edge_index, batch_size):
    # rs_df = load_single_drug_response_data("CCLE", split=1, split_type=["test"], y_col_name='auc')
    print(rs_df_split.size)
    Dataset = IMPROVE_Dataset
    collate_fn = _collate

    rs_dataset = Dataset(drug_dict, cell_dict, rs_df_split, edge_index=edge_index)
    rs_loader = DataLoader(rs_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=4)
    return rs_loader

def load_IMPROVE_data(IC, drug_dict, cell_dict, edge_index, setup, model, batch_size):
    if setup == 'known':
        IC_sample = IC.sample(frac=0.05, random_state=42)
        train_set, val_test_set = train_test_split(IC_sample, test_size=0.2, random_state=42)
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)

    elif setup == 'leave_drug_out':
        ## scaffold
        smiles_list = pd.read_csv(
            os.path.join(improve_globals.x_data_dir, "drug_smiles.csv"))[
            ['canSMILES', 'improve_chem_id']]
        train_set, val_set, test_set = scaffold_split(IC, smiles_list, seed=42)

    elif setup == 'leave_cell_out':
        ## stratify
        cell_info = IC[['source', 'improve_sample_id']].drop_duplicates()
        train_cell, val_test_cell = train_test_split(cell_info, stratify=cell_info['source'], test_size=0.4,
                                                     random_state=42)
        val_cell, test_cell = train_test_split(val_test_cell, stratify=val_test_cell['source'], test_size=0.5,
                                               random_state=42)

        train_set = IC[IC['improve_sample_id'].isin(train_cell['improve_sample_id'])]
        val_set = IC[IC['improve_sample_id'].isin(val_cell['improve_sample_id'])]
        test_set = IC[IC['improve_sample_id'].isin(test_cell['improve_sample_id'])]

    else:
        raise ValueError

    if model == 'TCNN':
        Dataset = IMPROVE_Dataset_name
        collate_fn = None
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif model == 'GraphDRP':
        Dataset = IMPROVE_Dataset_name
        collate_fn = _collate_drp
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif model == 'DeepCDR':
        Dataset = IMPROVE_Dataset_CDR
        collate_fn = _collate_CDR
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    else:
        Dataset = IMPROVE_Dataset
        collate_fn = _collate
        train_dataset = Dataset(drug_dict, cell_dict, train_set, edge_index=edge_index)
        val_dataset = Dataset(drug_dict, cell_dict, val_set, edge_index=edge_index)
        test_dataset = Dataset(drug_dict, cell_dict, test_set, edge_index=edge_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4
                              )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4
                            )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=4)

    return train_loader, val_loader, test_loader


class EarlyStopping:
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

