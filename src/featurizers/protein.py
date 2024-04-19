import sys
import pickle
import os
import torch
import hashlib
import pickle as pk
import typing as T
from pathlib import Path
from ..protein_to_graph import coord_to_graph
from .base import Featurizer
from ..utils import get_logger


logg = get_logger()
MODEL_CACHE_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
FOLDSEEK_MISSING_IDX = 20
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class ESMFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("ESM", 1280, save_dir)

        import esm

        torch.hub.set_dir(r'/home/inspur/zdp409100230054/.cache/torch/hub')    
        last_part = os.path.basename(save_dir)


        if last_part == 'KIBA':
            self._max_len = 1310
        else:
            self._max_len = 1210         

        (self._esm_model,self._esm_alphabet,) = esm.pretrained.esm2_t33_650M_UR50D()  
        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self._register_cuda("model", self._esm_model)

    def _transform(self, seq: str):
        seq = seq.upper()
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter(
            [("sequence", seq)]
        )
        batch_tokens = batch_tokens.to(self.device)
        results = self._cuda_registry["model"][0](
            batch_tokens, repr_layers=[33], return_contacts=True
        )
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        tokens = token_representations[0, 1 : len(seq) + 1]

        return tokens.mean(0)




class ProtBertFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
        super().__init__("ProtBert", 1024, save_dir)

        from transformers import AutoTokenizer, AutoModel, pipeline

        self._max_len = 1310
        self.per_tok = per_tok

        self._protbert_tokenizer = AutoTokenizer.from_pretrained(
            r"/home/inspur/zdp409100230054/PocketDTA/models/huggingface/transformers/models--Rostlab--prot_bert",
            do_lower_case=False,

        )
        self._protbert_model = AutoModel.from_pretrained(
            r"/home/inspur/zdp409100230054/PocketDTA/models/huggingface/transformers/models--Rostlab--prot_bert",

        )
        self._protbert_feat = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
        )

        self._register_cuda("model", self._protbert_model)
        self._register_cuda("featurizer", self._protbert_feat, self._feat_to_device)

    def _feat_to_device(self, pipe, device):
        from transformers import pipeline

        if device.type == "cpu":
            d = -1
        else:
            d = device.index

        pipe = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=d,
        )
        self._protbert_feat = pipe
        return pipe

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):

        embedding = torch.tensor(self._cuda_registry["featurizer"][0](self._space_sequence(seq))) 
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx] 

        return feats.mean(0) 

class ProtT5XLUniref50Featurizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
        super().__init__("ProtT5XLUniref50", 1024, save_dir)
        last_part = os.path.basename(save_dir)
        if last_part == 'KIBA':
            self._max_len = 1310
        else:
            self._max_len = 1210         
        self.per_tok = per_tok

        (
            self._protbert_model,
            self._protbert_tokenizer,
        ) = ProtT5XLUniref50Featurizer._get_T5_model()
        self._register_cuda("model", self._protbert_model)

    @staticmethod
    def _get_T5_model():
        from transformers import T5Tokenizer, T5EncoderModel


        local_model_path = r'/home/inspur/zdp409100230054/PocketDTA/models/huggingface/transformers/models--Rostlab--prot_t5_xl_uniref50'

        model = T5EncoderModel.from_pretrained(local_model_path)
        model = model.eval()  

        tokenizer = T5Tokenizer.from_pretrained(local_model_path, do_lower_case=False)

        return model, tokenizer

    @staticmethod
    def _space_sequence(x):
        return " ".join(list(x))

    def _transform(self, seq: str):


        token_encoding = self._protbert_tokenizer.batch_encode_plus(
            ProtT5XLUniref50Featurizer._space_sequence(seq),
            add_special_tokens=True,
            padding="longest",
        )
        input_ids = torch.tensor(token_encoding["input_ids"])
        attention_mask = torch.tensor(token_encoding["attention_mask"])

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            embedding = self._cuda_registry["model"][0](
                input_ids=input_ids, attention_mask=attention_mask
            )
            embedding = embedding.last_hidden_state
            seq_len = len(seq)
            start_Idx = 1
            end_Idx = seq_len + 1
            seq_emb = embedding[0][start_Idx:end_Idx]

        return seq_emb.mean(0)