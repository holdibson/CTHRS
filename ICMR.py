import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import one_hot
from evaluate import  calculate_top_map
import load_dataloader
from metric import ContrastiveLoss
from model import FuseTransEncoder, ImageMlp, TextMlp
from os import path as osp
from utils import load_checkpoints, save_checkpoints
from torch.optim import lr_scheduler
import datetime
class Solver(object):
    def __init__(self, config):
        self.config=config
        self.dataset  = config.dataset
        self.total_epoch = config.epoch
        self.batch_size = config.batch_size
        self.nbits = config.hash_lens
        self.label_dim=config.label_dim
        self.model_dir = config.model_dir
        self.alpha = config.alpha
        self.a1=config.a1
        self.a2=config.a2
        self.a3=config.a3
        self.task = config.task
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        self.img_feat_lens = 512
        self.txt_feat_lens = 512
        num_layers, self.token_size, nhead = 2, 1024, 4
 
        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(self.device)
        self.ImageMlp = ImageMlp(self.img_feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.txt_feat_lens, self.nbits).to(self.device)
        
        paramsFuse_to_update = list(self.FuseTrans.parameters()) 
        paramsImage = list(self.ImageMlp.parameters()) 
        paramsText = list(self.TextMlp.parameters()) 
        
        total_param = sum([param.nelement() for param in paramsFuse_to_update])+sum([param.nelement() for param in paramsImage])+sum([param.nelement() for param in paramsText])
        print("total_param:",total_param)

        self.optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))
        
        if self.dataset == "rsicd":
            self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
            self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp,milestones=[20,40,60,80], gamma=1.5)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp,milestones=[20,40,60,80], gamma=1.5)
        elif self.dataset == "rsitmd":
            self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
            self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp,milestones=[10,30,50,70,90], gamma=1.8)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp,milestones=[10,30,50,70,90], gamma=1.8)
        else:    
            self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=2*1e-3, betas=(0.5, 0.999))
            self.optimizer_TextMlp = optim.Adam(paramsText, lr=2*1e-3, betas=(0.5, 0.999))
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp,milestones=[10,30,50,70,90], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp,milestones=[10,30,50,70,90], gamma=1.5)    
        self.train_loader,self.query_loader,self.retrieval_loader=load_dataloader.dataloader(self.dataset,self.batch_size)
     
     
    def train(self):
        if self.task == 0: # train real
            print("Training Fusion Transformer...")
            for epoch in range(self.total_epoch):
                print("epoch:",epoch+1)
                train_loss,idx = self.trainfusion()
                if((epoch+1)%10==0):
                    print("Testing...")
                    img2text, text2img = self.evaluate() 
                    print('I2T:',img2text, ', T2I:',text2img)
            save_checkpoints(self)
           
        elif self.task == 1: # train hash 
            print("Training Hash Fuction...")
            I2T_MAP = []
            T2I_MAP = []
            lossfile=""
            mapfile=""
            if self.dataset == "rsicd":
                lossfile="rsicdloss.txt"
                mapfile="rsicdmap.txt"
            elif self.dataset == "rsitmd":
                lossfile="rsitmdloss.txt"
                mapfile="rsitmdmap.txt"
            else:
                lossfile="ucmloss.txt"
                mapfile="ucmmap.txt"
            with open(lossfile, 'a+', encoding='utf-8') as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("----------------{0}--------{1}-----{2}-----{3}---\n".format(self.nbits,current_time,self.ImageMlp_scheduler.milestones,self.ImageMlp_scheduler.gamma))
                f.write("----------------img-lr:{0}---------------------\n".format(self.optimizer_ImageMlp.state_dict()['param_groups'][0]['lr']))
                f.write("----------------txt-lr:{0}---------------------\n".format(self.optimizer_TextMlp.state_dict()['param_groups'][0]['lr']))
                f.write("----------------{0}---------------------\n".format(self.config))

            with open(mapfile, 'a+', encoding='utf-8') as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("----------------{0}--------{1}-----{2}-----{3}---\n".format(self.nbits,current_time,self.ImageMlp_scheduler.milestones,self.ImageMlp_scheduler.gamma))
                f.write("----------------img-lr:{0}---------------------\n".format(self.optimizer_ImageMlp.state_dict()['param_groups'][0]['lr']))
                f.write("----------------txt-lr:{0}---------------------\n".format(self.optimizer_TextMlp.state_dict()['param_groups'][0]['lr']))
                f.write("----------------{0}---------------------\n".format(self.config))
            for epoch in range(self.total_epoch):
                print("epoch:",epoch+1)
                train_loss,idx = self.trainhash()
                print(train_loss)
                with open(lossfile, 'a+', encoding='utf-8') as f:
                     f.write("{0}\n".format(train_loss))
                if((epoch+1)%10==0):
                    print("Testing...")
                    img2text, text2img = self.evaluate() 
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    print('I2T:',img2text, ', T2I:',text2img)
                    with open(mapfile, 'a+', encoding='utf-8') as f:
                         f.write("I2T:{0},T2I:{1} \n".format(img2text,text2img))
            print(I2T_MAP,T2I_MAP)
            save_checkpoints(self)
                
        elif self.task == 2: # test real
            file_name = self.dataset + '_fusion.pth'
            ckp_path = osp.join(self.model_dir,'real', file_name)
            load_checkpoints(self, ckp_path)

        elif self.task == 3: # test hash 
            
            file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
            ckp_path = osp.join(self.model_dir,'hash', file_name)
            load_checkpoints(self, ckp_path)

        print("Final Testing...")
        i2t, t2i = self.evaluate() 
        print('I2T:',i2t, ', T2I:',t2i)
        return (i2t + t2i)/2., i2t, t2i
      
    def evaluate(self):
        self.FuseTrans.eval()
        self.ImageMlp.eval()
        self.TextMlp.eval()
        qu_BI, qu_BT, qu_L = [], [], []
        re_BI, re_BT, re_L = [], [], []
        with torch.no_grad():
            for _,(data_I, data_T, data_L) in enumerate(self.query_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                temp_tokens = torch.concat((data_I, data_T), dim = 1)
                img_query ,txt_query =  self.FuseTrans(temp_tokens)
                if self.task == 1 or self.task == 3:
                    img_query = self.ImageMlp(img_query)
                    txt_query = self.TextMlp(txt_query)
                img_query, txt_query = img_query.cpu().numpy(), txt_query.cpu().numpy()
                qu_BI.extend(img_query)
                qu_BT.extend(txt_query)
                data_L=data_L.long()
                if len(data_L.shape) == 1:
                    data_L = one_hot(data_L, num_classes=self.label_dim).to(self.device)
                else:
                    data_L.to(self.device)
                qu_L.extend(data_L.cpu().numpy())  

            for _,(data_I, data_T, data_L) in enumerate(self.retrieval_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                temp_tokens = torch.concat((data_I, data_T), dim = 1)
                img_retrieval ,txt_retrieval =  self.FuseTrans(temp_tokens)
                if self.task ==1 or self.task ==3:
                    img_retrieval = self.ImageMlp(img_retrieval)
                    txt_retrieval = self.TextMlp(txt_retrieval)
                img_retrieval, txt_retrieval = img_retrieval.cpu().numpy(), txt_retrieval.cpu().numpy()
                re_BI.extend(img_retrieval)
                re_BT.extend(txt_retrieval)
                data_L=data_L.long()
                if len(data_L.shape) == 1:
                    data_L = one_hot(data_L, num_classes=self.label_dim).to(self.device)
                else:
                    data_L.to(self.device)
                re_L.extend(data_L.cpu().numpy())
        
        re_BI = np.array(re_BI)
        re_BT = np.array(re_BT)
        re_L = np.array(re_L)
        
        qu_BI = np.array(qu_BI)
        qu_BT = np.array(qu_BT)
        qu_L = np.array(qu_L)
        
        
        if self.task ==1 or self.task ==3:   # hashing
            qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
            qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
            re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
            re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()
        elif self.task ==0 or self.task ==2:  # real value
            qu_BI = torch.tensor(qu_BI).cpu().numpy()
            qu_BT = torch.tensor(qu_BT).cpu().numpy()
            re_BT = torch.tensor(re_BT).cpu().numpy()
            re_BI = torch.tensor(re_BI).cpu().numpy()
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=20)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=20)

        return MAP_I2T, MAP_T2I 
    
    def trainfusion(self):
        self.FuseTrans.train()
        running_loss = 0.0
        for idx, (img, txt,_) in enumerate(self.train_loader):
            temp_tokens = torch.concat((img, txt), dim = 1).to(self.device)
            temp_tokens = temp_tokens.unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            img=img.to(self.device)
            txt=txt.to(self.device)
            loss1 = self.ContrastiveLoss(img, img_embedding)
            loss2 = self.ContrastiveLoss(txt, text_embedding)
            loss=loss1+loss2
            self.optimizer_FuseTrans.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            running_loss += loss.item()
        return running_loss,idx
    
    def trainhash(self):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        for idx, (img, txt,_) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            temp_tokens = torch.concat((img, txt), dim = 1)
            temp_tokens = temp_tokens.unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            SF = F.normalize(img_embedding).mm(F.normalize(text_embedding).t())
            SF_I = F.normalize(img_embedding).mm(F.normalize(img_embedding).t())
            SF_T = F.normalize(text_embedding).mm(F.normalize(text_embedding).t())
            self.ContrastiveLoss = ContrastiveLoss(batch_size=img.shape[0], device=self.device)
            loss1 = self.ContrastiveLoss(img, img_embedding)
            loss2 = self.ContrastiveLoss(txt, text_embedding)
            decomposed_loss=loss1+loss2
            
            img_embedding = self.ImageMlp(img_embedding)
            text_embedding = self.TextMlp(text_embedding)
            SH = F.normalize(img_embedding).mm(F.normalize(text_embedding).t())
            SH_I = F.normalize(img_embedding).mm(F.normalize(img_embedding).t())
            SH_T = F.normalize(text_embedding).mm(F.normalize(text_embedding).t())

            loss_intra=0.5*(F.mse_loss(SF_I, SH_I)+F.mse_loss(SF_T, SH_T))+0.5*(F.mse_loss(self.alpha*SF, SH_I)+F.mse_loss(self.alpha*SF, SH_T))
            loss_inter=F.mse_loss(self.alpha*torch.ones(img_embedding.shape[0]).to(self.device), SH.diag())+F.mse_loss(self.alpha*SF, SH)
            loss_erra=loss_intra+loss_inter
            
            loss_con = self.ContrastiveLoss(img_embedding, text_embedding)

            loss =self.a1*decomposed_loss+self.a2*loss_con+self.a3*loss_erra
            
            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()
        
            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()
        return running_loss,idx