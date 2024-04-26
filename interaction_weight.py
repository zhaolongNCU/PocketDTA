import os 
import time
import torch
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
from train_test import *
from utils_dta import *
from src.gvp_gnn import StructureEncoder
from src import model as model_types
from src.utils import (get_logger,
                       config_logger,
                       get_featurizer,
                       set_random_seed,)
from src.data import (get_task_dir,
                      DTADataModule,
                      )
from src.model import proSeqEncoder

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
def str_to_list(arg):
    return arg.split(',')


logg = get_logger()

parser = ArgumentParser(description="DTA Training.")
parser.add_argument("--config", help="YAML config file", default="configs/default_config.yaml")
parser.add_argument("--task",choices=["KIBA","bindingdb","Davis","metz","pdbbind",],type=str,
                    help="Task name. Could be kiba, bindingdb, davis, metz, pdbbind.",dest="task")
parser.add_argument("--drug-seq-featurizer", help="Drug seq featurizer", dest="drug_seq_featurizer")
parser.add_argument("--drug-struc-featurizer", help="Drug struc featurizer", dest="drug_struc_featurizer")
parser.add_argument("--GraphMVPName", help="GraphMVPName", dest="GraphMVPName")
parser.add_argument("--target-seq-featurizer", help="Target seq featurizer", dest="target_seq_featurizer")
parser.add_argument("--t", "--tops",type=str,help="DoGSite tops",dest="tops")
parser.add_argument("--epochs", "--epochs",type=int, help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", type=int, help="batch size",dest="batch_size")
parser.add_argument("--lr", "--learning-rate",type=float,help="initial learning rate",dest="learning_rate",)
parser.add_argument("--r", "--replicate", type=int, help="Replicate", dest="replicate")
parser.add_argument("--d", "--device", type=int, help="CUDA device", dest="device")
parser.add_argument("--g", "--struc-encoder-layer-num", type=int, help="GVP-GNN layers", dest="struc_encoder_layer_num")
parser.add_argument("--h", "--n-heads", type=int, help="n_heads", dest="n_heads")
parser.add_argument("--drugdim", "--drug-dim", type=int, help="drug_dim", dest="drug_dim")
parser.add_argument("--targetdim", "--target-dim", type=int, help="target_dim", dest="target_dim")
parser.add_argument("--h-dim", "--h-dim", type=int, help="h_dim", dest="h_dim")
parser.add_argument("--w", "--weight-decay", type=float, help="weight_decay", dest="weight_decay")
parser.add_argument("--ld", "--lr-decay", type=float, help="lr_decay", dest="lr_decay")
parser.add_argument("--domain", "--domain", type=str2bool, default=True, help="domain", dest="domain")
parser.add_argument("--use-drug-seq", type=str2bool, default=True)
parser.add_argument("--use-drug-struc", type=str2bool, default=True)
parser.add_argument("--use-target-seq", type=str2bool, default=True)
parser.add_argument("--use-target-struc", type=str2bool, default=True)
parser.add_argument("--model", help="Model", dest="model_architecture")
parser.add_argument("--use-cold-spilt", type=str2bool, default=False)
parser.add_argument("--cold", type=str_to_list, default=['Drug','target_key'])
parser.add_argument("--use-test", type=str2bool, default=False)

#Get configuration
args = parser.parse_args()
print(args.config)
config = OmegaConf.load(args.config)
arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
config.update(arg_overrides)
logg.info({k:v for k,v in config.items()})
print(config.tops)
print(config.struc_encoder_layer_num)
print(config.n_heads)
print(config.drug_dim)
print(config.target_dim)
print(config.node_h_dim)
print(config.model_architecture)
print(config.cold)

if config.cold == ['Drug']:
    cold_name = 'cold_drug'
elif config.cold == ['target_key']:
    cold_name = 'cold_target'
elif config.cold == ['Drug','target_key']:
    cold_name = 'cold_drug_target'    

#path
root_path = os.getcwd()
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

path_model = os.path.join(root_path,f'output/{config.task}-model-{current_time}-{config.replicate}')
best_model_file = r'/home/inspur/zdp409100230054/PocketDTA/output_best/256-1280-128_gvp_3调参/h_6/lr_0.001/w_0.0001/Davis-model-2024-03-04-02:40:35-lr-0.001-batch-32-h-6-drugdim-256-targetdim-1280-h_dim-128/best_model_epoch/model--epoch--194.pth'
if not os.path.exists(path_model):
    os.system("mkdir -p %s" % path_model)

result_file = 'test_results--%s.txt' % current_time
#model_file = 'model--%s.pth' % current_time
affinity_file = 'affinity_test.csv'
log_file = 'testing--%s.log' % current_time
result_test_file = f'results_test_seed{config.replicate}--{current_time}.txt'
file_results = os.path.join(path_model, result_file)
#file_model = os.path.join(path_model, model_file)
file_affinity = os.path.join(path_model, affinity_file)
file_test_results = os.path.join(path_model, result_test_file)
file_log = os.path.join(path_model,log_file)
#f_results = open(file_results, 'a')

config_logger(file_log,"%(asctime)s [%(levelname)s] %(message)s",config.verbosity,use_stdout=True,)


logg.info(f'config:{config}')

# Set CUDA device
device_no = config.device
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")

logg.info(f"Using CUDA device {device}")
logg.info(f"Weight_decay {config.weight_decay}")
logg.info(f"lr_decay {config.lr_decay}")
# Set random state
logg.debug(f"Setting random state {config.replicate}")
#torch.manual_seed(config.replicate)
#np.random.seed(config.replicate)
set_random_seed(config.replicate,deterministic=True)

# Load DataModule
logg.info("Preparing DataModule")
task_dir = get_task_dir(config.task)
#print(task_dir)

drug_seq_featurizer = get_featurizer(config.drug_seq_featurizer, save_dir=task_dir)
drug_struc_featurizer = get_featurizer(config.drug_struc_featurizer, name=config.GraphMVPName,save_dir=task_dir)
target_seq_featurizer = get_featurizer(config.target_seq_featurizer, save_dir=task_dir)
#target_struc_featurizer = get_featurizer(config.target_struc_featurizer, save_dir=task_dir)

datamodule = DTADataModule(
            task_dir,
            drug_seq_featurizer,
            drug_struc_featurizer,
            target_seq_featurizer,
            #target_struc_featurizer,
            tops=config.tops,
            device=device,
            seed=config.replicate,
            use_cold_spilt=config.use_cold_spilt,
            use_test=config.use_test,
            cold=config.cold,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            )

# Load Dataset
datamodule.prepare_data()
datamodule.setup()

config.drug_seq_shape = drug_seq_featurizer.shape 
config.drug_struc_shape = drug_struc_featurizer.shape            
config.target_seq_shape = target_seq_featurizer.shape        
#print(config.drug_seq_shape,config.drug_struc_shape,config.target_seq_shape)


# Model
logg.info("Initializing model")
#print(config.node_in_dim)

# Parameters of protein sequence encoder
seq_dim = 1024
seq_hid_dim = 256              #512
seq_encoder_layer_num = 1      #1
kernel_size = 7
seq_dropout = 0.3
max_pro_seq_len = 1310


drug_seq_dim = 300
drug_seq_hid_dim = config.drug_dim              #512
drug_seq_encoder_layer_num = 1      #1
drug_kernel_size = 5
drug_seq_dropout = 0.3
max_drug_seq_len = 61

drug_seq_encoder = proSeqEncoder(drug_seq_dim, drug_seq_hid_dim, drug_seq_encoder_layer_num, drug_kernel_size,
                                  max_drug_seq_len, dropout=drug_seq_dropout)

target_struc_encoder = StructureEncoder(node_in_dim=(config.node_in_dim[0],config.node_in_dim[1]), 
                                 node_h_dim=(config.node_h_dim[0],config.node_in_dim[1]),      
                                 edge_in_dim=(config.edge_in_dim[0],config.edge_in_dim[1]), 
                                 edge_h_dim=(config.edge_h_dim[0],config.edge_h_dim[1]),
                                 seq_in=False, num_layers=config.struc_encoder_layer_num,
                                 drop_rate=config.struc_dropout,)

model_best = getattr(model_types, config.model_architecture)(drug_seq_encoder,config.target_seq_shape,target_struc_encoder,
                                                        drug_seq_dim=config.drug_seq_shape,
                                                        drug_dim=config.drug_dim,target_dim=config.target_dim,
                                                        gvp_output_dim=config.node_h_dim[0],
                                                        h_dim=config.h_dim,n_heads=config.n_heads,
                                                        use_drug_seq=config.use_drug_seq, use_drug_struc=config.use_drug_struc, 
                                                        use_target_seq=config.use_target_seq, use_target_struc=config.use_target_struc,)

model_best = model_best.to(device)
logg.info(model_best)

tester = Tester(model_best, config.batch_size)


min_mse_test = float('inf')

best_epoch_test = 0

results_test = ('tTime\tLoss_test\tMSE_test\tCI_test\tRm2_test\tPearson_test\tSpearman_test')
with open(file_test_results,'w') as f:
    f.write(results_test+'\n')

best_model_state_dict = torch.load(best_model_file)
model_best.load_state_dict(best_model_state_dict)
logg.info(f"Loading model from {best_model_file}")

best_model = model_best.to(device)
best_tester = Tester_BAN(best_model, config.batch_size)
testing_generator,test_len = datamodule.test_dataloader(domain=config.domain)
start_time_test = time.time()
G_test, P_test,att = best_tester.test(testing_generator, device,test_len)
print('True',G_test)
print('Predict',P_test)

del model_best

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch

def process_tensor(tensor):
    heads_mean = tensor.mean(dim=1).squeeze(0)

    drug_mean = heads_mean.mean(dim=1, keepdim=True).squeeze(1)

    drug_mean_shape = drug_mean.shape[0]+1
    print(drug_mean_shape)
    target_mean = heads_mean.mean(dim=0, keepdim=True).squeeze(0)
    target_mean_shape = target_mean.shape[0]+1
    print(target_mean_shape)
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    print()
    drug_normalized = normalize(drug_mean[1:drug_mean_shape])
    print(drug_normalized.shape)
    target_normalized = normalize(target_mean[1:target_mean_shape])
    print(target_normalized.shape)

    return drug_normalized, target_normalized


drug_softmax, target_softmax = process_tensor(att)


print(drug_softmax), print(target_softmax)
