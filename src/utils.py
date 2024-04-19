import sys
import logging as lg
import typing as T
from pathlib import Path
import random
import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

logLevels = {0: lg.ERROR, 1: lg.WARNING, 2: lg.INFO, 3: lg.DEBUG}
LOGGER_NAME = "DTA"

def get_logger():
    return lg.getLogger(LOGGER_NAME)

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def config_logger(
    file: T.Union[Path, None],
    fmt: str,
    level: bool = 2,
    use_stdout: bool = True,
):
    """
    Create and configure the logger

    :param file: Can be a Path or None -- if a Path, log messages will be written to the file at Path
    :type file: T.Union[Path, None]
    :param fmt: Formatting string for the log messages
    :type fmt: str
    :param level: Level of verbosity
    :type level: int
    :param use_stdout: Whether to also log messages to stdout
    :type use_stdout: bool
    :return:
    """

    module_logger = lg.getLogger(LOGGER_NAME)
    module_logger.setLevel(logLevels[level])
    formatter = lg.Formatter(fmt)

    if file is not None:
        fh = lg.FileHandler(file)
        fh.setFormatter(formatter)
        module_logger.addHandler(fh)

    if use_stdout:
        sh = lg.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        module_logger.addHandler(sh)

    lg.propagate = False

    return module_logger

def get_featurizer(featurizer_string, *args, **kwargs):
    from . import featurizers  

    featurizer_string_list = featurizer_string.split(",") 
    if len(featurizer_string_list) > 1:
        featurizer_list = [getattr(featurizers, i.strip()) for i in featurizer_string_list]
        return featurizers.ConcatFeaturizer(featurizer_list, *args, **kwargs)
    else:
        return getattr(featurizers, featurizer_string_list[0])(*args, **kwargs) 


