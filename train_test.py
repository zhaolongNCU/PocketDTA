import torch
import torch.nn as nn
import time
import os
import pickle
from tqdm import tqdm
from Radam import RAdam
from lookahead import Lookahead
import torch_geometric
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch_size, gradient_accumulation):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        self.optimizer_inner = RAdam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, la_steps=5, la_alpha=0.5)    
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation

    def train(self,dataloader,device,all_count):
        self.model.train()
        Loss = nn.MSELoss()
        loss_total = 0
        self.optimizer.zero_grad()
        current_count = 0
        all_count = all_count
        #print(all_count)
        spent_time_accumulation = 0
        all_predict_labels, all_real_labels = torch.Tensor(), torch.Tensor()
        step = 0
        for step, batch in enumerate(dataloader):
            step = step
            # print(step)
            start_time_batch = time.time()
            batch1, batch2 = batch
            # print(len(batch1))
            batch_split = torch.tensor(list(range(batch1[1].shape[0]))).to(device)      #batch1[0].shape[0]=batch_size
            drug_seq, drug_struc,target_seq, labels = batch1
            # print(labels)
            drug_seq, drug_struc, target_seq, labels = drug_seq.to(device), drug_struc.to(device), target_seq.to(device), labels.to(device)
            target_struc = [batch_each_sample.to(device) for batch_each_sample in batch2]
            data_pack = (drug_seq,drug_struc,target_seq,target_struc,batch_split,device)
            predict_labels = self.model(data_pack)
            loss = Loss(predict_labels.float(), labels.to(device).float())               #labels:B
            all_predict_labels = torch.cat((all_predict_labels,predict_labels.cpu()),0)
            loss_total += loss.item()
            loss /= self.gradient_accumulation
            loss.backward()
            all_real_labels = torch.cat((all_real_labels,labels.cpu()),0)                #all_sample
            if (step+1) % self.gradient_accumulation == 0 or (step+1) == all_count:      
                self.optimizer.step()
                self.optimizer.zero_grad()
            end_time_batch = time.time()
            seconds = end_time_batch-start_time_batch
            spent_time_accumulation += seconds
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_batch = "%02d:%02d:%02d" % (h, m, s)
            m, s = divmod(spent_time_accumulation, 60)
            h, m = divmod(m, 60)
            have_spent_time = "%02d:%02d:%02d" % (h, m, s)   

            current_count += 1
            if current_count == all_count:
                print("Finish batch: %d/%d---batch time: %s, have spent time: %s" % (current_count, all_count, spend_time_batch, have_spent_time))
            else:
                print("Finish batch: %d/%d---tatch time: %s, have spent time: %s" % (current_count, all_count, spend_time_batch, have_spent_time), end='\r')
            #del data_pack
            #torch.cuda.empty_cache()
        #print(step+1)
        return loss_total/(step+1),all_real_labels.detach().numpy().flatten(),all_predict_labels.detach().numpy().flatten()
    
    def save_model(self, model, filename):
        # model_to_save = model
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), filename)

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataloader, device,all_count):
        self.model.eval()
        #dataset, dataset_structure = dataset_tuple
        #datasampler = SequentialSampler(dataset)
        #dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size)
        #dataloaderstruc = torch_geometric.loader.DataListLoader(dataset_structure, num_workers=4, batch_size=self.batch_size)
        Loss = nn.MSELoss()
        loss_total = 0
        all_count = all_count
        all_predict_labels, all_real_labels = torch.Tensor(), torch.Tensor()
        for step, batch in enumerate(dataloader):
            batch1, batch2 = batch
            batch_split = torch.tensor(list(range(batch1[1].shape[0]))).to(device) #batch1[0].shape[0]=batch_size
            drug_seq, drug_struc, target_seq, labels = batch1
            drug_seq, drug_struc, target_seq, labels = drug_seq.to(device), drug_struc.to(device), target_seq.to(device), labels.to(device)
            target_struc = [batch_each_sample.to(device) for batch_each_sample in batch2]
            data_pack = (drug_seq,drug_struc,target_seq,target_struc,batch_split,device)            
            with torch.no_grad():
                predict_labels = self.model(data_pack)
            loss = Loss(predict_labels.float(), labels.to(device).float())
            all_predict_labels = torch.cat((all_predict_labels,predict_labels.cpu()),0)
            all_real_labels = torch.cat((all_real_labels,labels.cpu()),0)
            loss_total += loss.item()
        return loss_total/all_count,all_real_labels.detach().numpy().flatten(),all_predict_labels.detach().numpy().flatten()


class Tester_BAN(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataloader, device,all_count):
        self.model.eval()
        #dataset, dataset_structure = dataset_tuple
        #datasampler = SequentialSampler(dataset)
        #dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size)
        #dataloaderstruc = torch_geometric.loader.DataListLoader(dataset_structure, num_workers=4, batch_size=self.batch_size)
        Loss = nn.MSELoss()
        loss_total = 0
        all_count = all_count
        all_predict_labels, all_real_labels = torch.Tensor(), torch.Tensor()
        for step, batch in enumerate(dataloader):
            batch1, batch2 = batch
            batch_split = torch.tensor(list(range(batch1[1].shape[0]))).to(device) #batch1[0].shape[0]=batch_size
            drug_seq, drug_struc, target_seq, labels = batch1
            drug_seq, drug_struc, target_seq, labels = drug_seq.to(device), drug_struc.to(device), target_seq.to(device), labels.to(device)
            target_struc = [batch_each_sample.to(device) for batch_each_sample in batch2]
            data_pack = (drug_seq,drug_struc,target_seq,target_struc,batch_split,device)            
            with torch.no_grad():
                predict_labels,att = self.model(data_pack)
            #print(att.shape)
            #print(att)
            #loss = Loss(predict_labels.float(), labels.to(device).float())
            all_predict_labels = torch.cat((all_predict_labels,predict_labels.cpu()),0)
            all_real_labels = torch.cat((all_real_labels,labels.cpu()),0)
            #loss_total += loss.item()
        return all_real_labels.detach().numpy().flatten(),all_predict_labels.detach().numpy().flatten(),att