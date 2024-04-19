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

#Get configuration
args = parser.parse_args()
print(args.config)
config = OmegaConf.load(args.config)
arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
config.update(arg_overrides)
logg.info({k:v for k,v in config.items()})

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
if not os.path.exists(path_model):
    os.system("mkdir -p %s" % path_model)
result_file = 'results--%s.txt' % current_time
model_file = 'model--%s.pth' % current_time
loss_file = 'loss--%s.csv' % current_time
log_file = 'training--%s.log' % current_time
result_test_file = f'results_test_seed{config.replicate}--{current_time}.txt'
file_results = os.path.join(path_model, result_file)
file_model = os.path.join(path_model, model_file)
file_loss = os.path.join(path_model, loss_file)
file_test_results = os.path.join(path_model, result_test_file)
file_log = os.path.join(path_model,log_file)
f_results = open(file_results, 'a')

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

set_random_seed(config.replicate,deterministic=True)

logg.info("Preparing DataModule")
task_dir = get_task_dir(config.task)


drug_seq_featurizer = get_featurizer(config.drug_seq_featurizer, save_dir=task_dir)
drug_struc_featurizer = get_featurizer(config.drug_struc_featurizer, name=config.GraphMVPName,save_dir=task_dir)
target_seq_featurizer = get_featurizer(config.target_seq_featurizer, save_dir=task_dir)

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

# Model
logg.info("Initializing model")

seq_dim = 1024
seq_hid_dim = 256              
seq_encoder_layer_num = 1      
kernel_size = 7
seq_dropout = 0.3
max_pro_seq_len = 1310

drug_seq_dim = 300
drug_seq_hid_dim = config.drug_dim             
drug_seq_encoder_layer_num = 1  
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

model = getattr(model_types, config.model_architecture)(drug_seq_encoder,config.target_seq_shape,target_struc_encoder,
                                                        drug_seq_dim=config.drug_seq_shape,
                                                        drug_dim=config.drug_dim,target_dim=config.target_dim,
                                                        gvp_output_dim=config.node_h_dim[0],
                                                        h_dim=config.h_dim,n_heads=config.n_heads,
                                                        use_drug_seq=config.use_drug_seq, use_drug_struc=config.use_drug_struc, 
                                                        use_target_seq=config.use_target_seq, use_target_struc=config.use_target_struc,)

model = model.to(device)
logg.info(model)

trainer = Trainer(model, config.learning_rate, config.weight_decay, config.batch_size, config.gradient_accumulation)
tester = Tester(model, config.batch_size)


if "checkpoint" in config:                      
    state_dict = torch.load(config.checkpoint)
    model.load_state_dict(state_dict)
    logg.info(f"Loading checkpoint model from {config.checkpoint}")

start_time = time.time()


'''start training.'''
logg.info("Beginning Training")

min_mse_val = float('inf')

best_epoch = 0

loss_train_epochs, loss_val_epochs = [], []

last_part = os.path.basename(task_dir)


for epoch in range(1,config.epochs+1):
    print('NUM_epoch:',epoch)

    logg.info("Getting DataLoaders")
    training_generator,train_len = datamodule.train_dataloader(seed=epoch,domain=config.domain)  
    validation_generator,val_len = datamodule.val_dataloader(domain=config.domain)
    start_time_epoch = time.time()
    if epoch % config.decay_interval == 0:
        trainer.optimizer.param_groups[0]['lr'] *= config.lr_decay

    loss_train, G_train, P_train = trainer.train(training_generator, device,train_len)

    print('G:',G_train)
    print('P:',P_train)
    loss_val, G_val, P_val = tester.test(validation_generator, device,val_len)

    logg.info(f'epoch:{epoch}_trainmse:{loss_train}_testmse:{loss_val}')


    end_time_epoch = time.time()
    seconds = end_time_epoch-start_time_epoch
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    spend_time_epoch = "%02d:%02d:%02d" % (h, m, s)
    loss_train_epoch = "%.3f" % loss_train
    loss_val_epoch = "%.3f" % loss_val
    loss_train_epochs.append(float(loss_train_epoch))
    loss_val_epochs.append(float(loss_val_epoch))

    if loss_val < min_mse_val:
        trainer.save_model(model,file_model)
        best_model = copy.deepcopy(model)
        min_mse_val = loss_val
        best_epoch = epoch
        logg.info(f"The best model is in epoch %d. MSE val:%s\n" % (epoch,loss_val_epoch))


end_time = time.time()
seconds = end_time-start_time
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
spend_time = "%02d:%02d:%02d" % (h, m, s)
final_print = "All epochs spend %s, where the best model is in epoch %d" % (spend_time, best_epoch)
params = f'trainable params in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}'
seconds_epoch = f'Every epoch time:{seconds/config.epochs}'
logg.info(final_print)
logg.info(params)
f_results.write(final_print+'\n')
f_results.write(seconds_epoch+'\n')
f_results.write(params+'\n')
f_results.close()

#plot
logg.info("Plotting")
dict_loss = {}
dict_loss['epochs'] = list(range(1, config.epochs+1))  
dict_loss['loss_train_all'],dict_loss['loss_val_all'] = loss_train_epochs, loss_val_epochs   

df_loss = pd.DataFrame(dict_loss)
df_loss.to_csv(file_loss, index=False)
plot_train_val_metric(list(range(1, config.epochs+1)), loss_train_epochs, loss_val_epochs, path_model, 'Loss', config.task)


#Test
logg.info("Beginning testing")
best_tester = Tester(best_model, config.batch_size)
testing_generator,test_len = datamodule.test_dataloader(domain=config.domain)
results_test = ('Time\tLoss_test\tMSE_test\tCI_test\tRm2_test\tPearson_test\tSpearman_test')
with open(file_test_results,'w') as f:
    f.write(results_test+'\n')
start_time_test = time.time()
loss_test, G_test, P_test = best_tester.test(testing_generator, device,test_len)
end_time_test = time.time()
seconds = end_time_test-start_time_test
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
spend_time_test = "%02d:%02d:%02d" % (h, m, s)
mse_test,ci_test,rm2_test,pearson_test,spearman_test = calculate_metrics(G_test,P_test)
results_test = [spend_time_test,loss_test,mse_test,ci_test,rm2_test,pearson_test,spearman_test]
with open(file_test_results, 'a') as f:
    f.write('\t'.join(map(str, results_test)) + '\n')

logg.info(f'The best model is epoch {best_epoch} in all epoch Test MSE: {mse_test}')
