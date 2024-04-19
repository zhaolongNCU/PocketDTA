import os
import pickle
import dgl
import math
import torch
#import pysmiles
import deepchem as dc
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from pathlib import Path
from .base import Featurizer
from ..utils import get_logger, canonicalize

from mol2vec.features import (
    mol2alt_sentence,
    mol2sentence,
    MolSentence,
    sentences2vec,
)
from gensim.models import word2vec
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot
from src.compound_gnn_model import GNNComplete
from src.molecule_datasets import mol_to_graph_data_obj_simple,mol_to_graph_data_obj_simple_molebert
from src.molebert_gnn import GNN

logg = get_logger()



class Informax_Featurizer(Featurizer):
    def __init__(self,name:str="3Dinformax",shape: int = 256,save_dir: Path = Path().absolute(),):
        super().__init__(name, shape, save_dir) 
        
        informax_dict_path = os.path.join(save_dir,f'3Dinformax_emb_dict.pt')
        self.informax_embeddings_dict = torch.load(informax_dict_path)

    def _transform(self, smile: str) -> torch.Tensor:
        # smile：'COCc1ccccc1C1C(C(=O)C(C)C)C(=O)C(=O)N1c1ccc(cc1)-c1cccs1'
        feats = self.informax_embeddings_dict[smile]                       

        return feats



class GraphMVPFeaturizer(Featurizer):
    def __init__(self,name:str="GraphMVP",shape: int = 300,save_dir: Path = Path().absolute(),):
        super().__init__(name, shape, save_dir) 
        last_part = os.path.basename(save_dir)
        #print(last_part)

        if last_part == 'KIBA':
            self._max_len = 36
        else:
            self._max_len = 40         

        self._shape = shape

        self.model = GNNComplete(num_layer=5,emb_dim=300,JK='last',drop_ratio=0.1,gnn_type='gin')
        GraphMVP_path = os.path.join(save_dir,f'{name}.pth')

        self.model.load_state_dict(torch.load(GraphMVP_path))
        self.model.to('cuda:0')

    def _transform(self, smile: str) -> torch.Tensor:

        rdkit_mol = AllChem.MolFromSmiles(smile)
        data = mol_to_graph_data_obj_simple(rdkit_mol)
        feats = torch.zeros((self._max_len,self._shape))
        with torch.no_grad():
            self.model.eval()
            graph_embedding = self.model(data.to('cuda:0'))
        if graph_embedding.shape[0] >= self._max_len:
            feats = graph_embedding[:self._max_len,:]
        else:
            feats[:graph_embedding.shape[0],:] = graph_embedding  #36*300

        return feats


class MoleBERTFeaturizer(Featurizer):
    def __init__(self,name:str="MoleBERT",shape: int = 300,save_dir: Path = Path().absolute(),):
        super().__init__(name, shape, save_dir) 
        last_part = os.path.basename(save_dir)


        if last_part == 'KIBA':
            self._max_len = 36
        else:
            self._max_len = 40         #Davis:40

        self._shape = shape

        self.model = GNN(5,300)
        MoleBERT_path = os.path.join(save_dir,f'{name}.pth')
        print(MoleBERT_path)
        self.model.from_pretrained(MoleBERT_path)
        self.model.to('cuda:0')

    def _transform(self, smile: str) -> torch.Tensor:
        rdkit_mol = AllChem.MolFromSmiles(smile)
        data = mol_to_graph_data_obj_simple_molebert(rdkit_mol)
        feats = torch.zeros((self._max_len,self._shape))
        with torch.no_grad():
            self.model.eval()
            graph_embedding = self.model(data.to('cuda:0'))
        if graph_embedding.shape[0] >= self._max_len:
            feats = graph_embedding[:self._max_len,:]
        else:
            feats[:graph_embedding.shape[0],:] = graph_embedding  

        return feats


class Mol2VecFeaturizer(Featurizer):
    def __init__(self, radius: int = 1, save_dir: Path = Path().absolute()):
        super().__init__("Mol2Vec", 300,save_dir)
        Mol2Vec_path = os.path.join(save_dir,'model_300dim.pkl')
        self._radius = radius
        self._model = word2vec.Word2Vec.load(Mol2Vec_path)

    def _transform(self, smile: str) -> torch.Tensor:

        molecule = Chem.MolFromSmiles(smile)
        try:
            sentence = MolSentence(mol2alt_sentence(molecule, self._radius))
            wide_vector = sentences2vec(sentence, self._model, unseen="UNK")
            feats = wide_vector.mean(axis=0)
        except Exception:
            feats = np.zeros(self.shape)

        feats = torch.from_numpy(feats).squeeze().float()
        #print(feats.shape)
        return feats


class MorganFeaturizer(Featurizer):
    def __init__(self,shape: int = 2048,radius: int = 2,save_dir: Path = Path().absolute(),):
        super().__init__("Morgan", shape, save_dir) 

        self._radius = radius

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile)    
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self._radius, nBits=self.shape)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features) 
        except Exception as e:
            logg.error(
                f"rdkit not found this smiles for morgan: {smile} convert to all 0 features"
            )
            logg.error(e)
            features = np.zeros((self.shape,))
        return features   #（2048，）

    def _transform(self, smile: str) -> torch.Tensor:

        feats = (torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float())    #2048
        if feats.shape[0] != self.shape:
            logg.warning("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats


