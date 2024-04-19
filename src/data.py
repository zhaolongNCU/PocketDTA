import torch
import os
import numpy as np
import pickle
import math
import pandas as pd
from .utils import get_logger
from pathlib import Path
import pytorch_lightning as pl
from .featurizers import Featurizer
from .featurizers.protein import FOLDSEEK_MISSING_IDX
import typing as T
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader, SequentialSampler,SubsetRandomSampler
import torch_geometric


logg = get_logger()


class BinaryDataset(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        #targets_id,
        labels,
        drug_seq_featurizer: Featurizer,
        drug_struc_featurizer: Featurizer,
        target_seq_featurizer: Featurizer,
        ):

        self.drugs = drugs
        self.targets = targets
        self.labels = labels

        self.drug_seq_featurizer = drug_seq_featurizer
        self.drug_struc_featurizer = drug_struc_featurizer        
        self.target_seq_featurizer = target_seq_featurizer


    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        drug_seq = self.drug_seq_featurizer(self.drugs.iloc[i])        
        drug_struc = self.drug_struc_featurizer(self.drugs.iloc[i])    
        target_seq = self.target_seq_featurizer(self.targets.iloc[i])  

        label = torch.tensor(self.labels.iloc[i])

        return drug_seq, drug_struc, target_seq, label

class ProteinGraphDataset(Dataset):
    def __init__(self, tops,targets,save_dir,domain):
        super(ProteinGraphDataset, self).__init__()
        self.targets = targets
        if domain:
            path = os.path.join(save_dir,f'Domain_coord_graph_{tops}seqid_dict.pickle')
        else:
            path = os.path.join(save_dir,f'coord_graph_{tops}seqid_dict.pickle')
        print(path)
        with open(path,'rb') as f:
            self.DoGsite3_dict = pickle.load(f)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):

        return self.DoGsite3_dict[self.targets.iloc[i]]   



def get_task_dir(task_name: str):
    task_paths = {
        'davis': './dataset/Davis',
        'kiba': './dataset/KIBA',
    }

    return Path(task_paths[task_name.lower()]).resolve()

def drug_target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):

    d_seq_emb = [a[0] for a in args]
    d_struc_emb = [a[1] for a in args]
    t_seq_emb = [a[2] for a in args]
    labs = [a[3] for a in args]

    drugs_seq = torch.stack(d_seq_emb, 0)
    drugs_struc = torch.stack(d_struc_emb,0)
    targets_seq = pad_sequence(t_seq_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)
    labels = torch.stack(labs, 0)

    return drugs_seq,drugs_struc, targets_seq, labels


def create_fold_setting_cold(df, fold_seed, frac, entities):
    if isinstance(entities, str):
        entities = [entities]
    train_frac, val_frac, test_frac = frac
    test_entity_instances = [df[e].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values for e in entities]

    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]                  

    if len(test) == 0:
        raise ValueError("No test samples found. Try another seed, increasing the test frac or a ""less stringent splitting strategy.")
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [train_val[e].drop_duplicates().sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed).values for e in entities]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError("No validation samples found. Try another seed, increasing the test frac ""or a less stringent splitting strategy.")

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class DTADataModule(pl.LightningDataModule):
    def __init__(self,data_dir: str,
        drug_seq_featurizer: Featurizer,
        drug_struc_featurizer: Featurizer,
        target_seq_featurizer: Featurizer,
        #target_struc_featurizer: Featurizer,
        tops: str = 'top3',
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        use_cold_spilt: bool = False,
        use_test: bool = False,
        cold: str = 'Drug',
        batch_size: int = 32,
        shuffle: bool = False,    
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",):
        self.tops = tops
        self.use_cold_spilt = use_cold_spilt
        self.cold = cold
        self.use_test = use_test
        self._loader_train_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }


        self._loader_kwargs = {
            "batch_size": batch_size,
            #"shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._loader_struc_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("process.csv")
        self._seed = seed

        self._drug_column = "Drug"
        self._target_column = "Target"
        self._label_column = "Y"

        self.drug_seq_featurizer = drug_seq_featurizer
        self.drug_struc_featurizer = drug_struc_featurizer
        self.target_seq_featurizer = target_seq_featurizer
        #self.target_struc_featurizer = target_struc_featurizer

    def prepare_data(self):

        if (self.drug_seq_featurizer.path.exists()and self.target_seq_featurizer.path.exists()):
            logg.warning("Drug and target seq featurizers already exist")
            return

        if (self.drug_struc_featurizer.path.exists()):
            logg.warning("Drug and target seq featurizers already exist")
            return

        self._dataframes = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

        all_drugs = self._dataframes[self._drug_column].unique()       
        all_targets = self._dataframes[self._target_column].unique()   #
        all_targets_id = self._dataframes['target_key'].unique() 

        if self._device.type == "cuda":
            self.drug_seq_featurizer.cuda(self._device)
            self.drug_struc_featurizer.cuda(self._device)
            self.target_seq_featurizer.cuda(self._device)
            #self.target_struc_featurizer.cuda(self._device)

        if not self.drug_seq_featurizer.path.exists():
            self.drug_seq_featurizer.write_to_disk(all_drugs)       

        if not self.drug_struc_featurizer.path.exists():
            self.drug_struc_featurizer.write_to_disk(all_drugs)        

        if not self.target_seq_featurizer.path.exists():
            self.target_seq_featurizer.write_to_disk(all_targets)
        
        #if not self.target_struc_featurizer.path.exists():
            #self.target_struc_featurizer.write_to_disk(all_targets_id)

        self.drug_seq_featurizer.cpu()
        self.target_seq_featurizer.cpu()
        self.drug_struc_featurizer.cpu()
        #self.target_struc_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):
 
        self._dataframes = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

        if self.use_cold_spilt:
            self.df_train,self.df_val,self.df_test = create_fold_setting_cold(self._dataframes,fold_seed=self._seed,
                                                                              frac=[0.8, 0.1, 0.1],entities=self.cold)
            print(len(self.df_test))
            logg.info(f'Processing {self.cold} cold start experiments')
        elif self.use_test:
            self.df_train,self.df_val,self.df_test = self._dataframes,self._dataframes,self._dataframes
            print(len(self.df_test))
        else:
            self.df_train, temp = train_test_split(self._dataframes, test_size=0.2, random_state=self._seed)
            self.df_val, self.df_test = train_test_split(temp, test_size=0.5, random_state=self._seed)
        print(self.df_train.iloc[0])
        self.df_train = self.df_train.sample(frac=1)         
        self.df_val = self.df_val.sample(frac=1)
        self.df_test = self.df_test.sample(frac=1)

        all_drugs = self._dataframes[self._drug_column].unique()
        all_targets = self._dataframes[self._target_column].unique()
        #all_targets_id = self._dataframes['target_key'].unique()

        if self._device.type == "cuda":
            self.drug_seq_featurizer.cuda(self._device)
            self.drug_struc_featurizer.cuda(self._device)
            self.target_seq_featurizer.cuda(self._device)
            #self.target_struc_featurizer.cuda(self._device)

        self.drug_seq_featurizer.preload(all_drugs)          
        self.drug_seq_featurizer.cpu()

        self.drug_struc_featurizer.preload(all_drugs)
        self.drug_struc_featurizer.cpu()

        self.target_seq_featurizer.preload(all_targets)
        self.target_seq_featurizer.cpu()


    def train_dataloader(self,seed,domain):

        self.df_train_shuffle = self.df_train.sample(frac=1,random_state=seed)  
        print(self.df_train_shuffle.iloc[0])
        self.data_train_pack = BinaryDataset(
                                self.df_train_shuffle[self._drug_column],
                                self.df_train_shuffle[self._target_column],
                                self.df_train_shuffle[self._label_column],
                                self.drug_seq_featurizer,
                                self.drug_struc_featurizer,
                                self.target_seq_featurizer,
                                )
        self.data_train_structure = ProteinGraphDataset(
                                self.tops,
                                self.df_train_shuffle['target_key'],
                                self._data_dir,
                                domain,)
        dataloader_pack = DataLoader(self.data_train_pack, sampler=SequentialSampler(self.data_train_pack),**self._loader_kwargs)
        dataloader_struc = torch_geometric.loader.DataListLoader(self.data_train_structure,**self._loader_struc_kwargs)
        dataloader = zip(dataloader_pack,dataloader_struc)
        train_len = len(dataloader_pack)
        return dataloader,train_len

    def val_dataloader(self,domain):
        #self.df_test_shuffle = self.df_test.sample(frac=1)
        self.data_val_pack = BinaryDataset(
                                self.df_val[self._drug_column],
                                self.df_val[self._target_column],
                                self.df_val[self._label_column],
                                self.drug_seq_featurizer,
                                self.drug_struc_featurizer,
                                self.target_seq_featurizer,
                                )
        self.data_val_structure = ProteinGraphDataset(
                                self.tops,
                                self.df_val['target_key'],
                                self._data_dir,
                                domain,)
        dataloader_pack = DataLoader(self.data_val_pack, sampler=SequentialSampler(self.data_val_pack),**self._loader_kwargs)
        dataloader_struc = torch_geometric.loader.DataListLoader(self.data_val_structure,**self._loader_struc_kwargs)
        dataloader = zip(dataloader_pack,dataloader_struc)
        val_len = len(dataloader_pack)
        return dataloader,val_len

    def test_dataloader(self,domain):
        self.data_test_pack = BinaryDataset(
                                self.df_test[self._drug_column],
                                self.df_test[self._target_column],
                                self.df_test[self._label_column],
                                self.drug_seq_featurizer,
                                self.drug_struc_featurizer,
                                self.target_seq_featurizer,
                                )
        self.data_test_structure = ProteinGraphDataset(
                                self.tops,
                                self.df_test['target_key'],
                                self._data_dir,
                                domain,)
        dataloader_pack = DataLoader(self.data_test_pack, sampler=SequentialSampler(self.data_test_pack),**self._loader_kwargs)
        dataloader_struc = torch_geometric.loader.DataListLoader(self.data_test_structure,**self._loader_struc_kwargs)
        dataloader = zip(dataloader_pack,dataloader_struc)
        test_len = len(dataloader_pack)
        return dataloader,test_len

