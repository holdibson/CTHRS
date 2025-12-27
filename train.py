import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import load_dataloader
from torch.nn.functional import one_hot
from metric import ContrastiveLoss
from model import FuseTransEncoder, ImageMlp, TextMlp
from os import path as osp
from utils import load_checkpoints, save_checkpoints,calculate_top_map
from torch.optim import lr_scheduler
class Train(object):
    def __init__(self, config):
        self.config=config
        self.dataset  = config.dataset
        self.total_epoch = config.epoch
        self.batch_size = config.batch_size
        self.bits = config.bits
        self.label_dim=config.label_dim
        self.model_dir = config.model_dir
        self.eta = config.eta
        self.alpha=config.alpha
        self.beta=config.beta
        self.gama=config.gama
        self.task = config.task
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        self.img_feat_lens = 512
        self.txt_feat_lens = 512
        num_layers, self.token_size, nhead = 2, 1024, 4
 
        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(self.device)
        self.ImageMlp = ImageMlp(self.img_feat_lens, self.bits).to(self.device)
        self.TextMlp = TextMlp(self.txt_feat_lens, self.bits).to(self.device)
        
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
        if self.task == 0: # train 
            print("Training...")
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
                f.write("----------------{0}--------{1}-----{2}-----{3}---\n".format(self.bits,current_time,self.ImageMlp_scheduler.milestones,self.ImageMlp_scheduler.gamma))
                f.write("----------------img-lr:{0}---------------------\n".format(self.optimizer_ImageMlp.state_dict()['param_groups'][0]['lr']))
                f.write("----------------txt-lr:{0}---------------------\n".format(self.optimizer_TextMlp.state_dict()['param_groups'][0]['lr']))
                f.write("----------------{0}---------------------\n".format(self.config))

            with open(mapfile, 'a+', encoding='utf-8') as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("----------------{0}--------{1}-----{2}-----{3}---\n".format(self.bits,current_time,self.ImageMlp_scheduler.milestones,self.ImageMlp_scheduler.gamma))
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
            print('i2t:{}, t2i:{}'.format(I2T_MAP,T2I_MAP))
            save_checkpoints(self)

        elif self.task == 1: # test  
            file_name = self.dataset + '_hash_' + str(self.bits)+".pth"
            ckp_path = osp.join(self.model_dir,'hash', file_name)
            load_checkpoints(self, ckp_path)

        i2t, t2i = self.evaluate() 
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
                Fuse_tokens = torch.concat((data_I, data_T), dim = 1)
                img_query ,txt_query =  self.FuseTrans(Fuse_tokens)
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
                Fuse_tokens = torch.concat((data_I, data_T), dim = 1)
                img_retrieval ,txt_retrieval =  self.FuseTrans(Fuse_tokens)
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
        
        qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
        qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
        re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
        re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=20)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=20)

        return MAP_I2T, MAP_T2I 
    
    def trainhash(self):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        a = getattr(self,'a',0.01) 
        T = 0.1 
        for idx, (img, txt,_) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            Fuse_tokens = torch.concat((img, txt), dim = 1)
            Fuse_tokens = Fuse_tokens.unsqueeze(0)
            img_f, text_f = self.FuseTrans(Fuse_tokens)
            
            SF = F.normalize(img_f).mm(F.normalize(text_f).t())
            SF_I = F.normalize(img_f).mm(F.normalize(img_f).t())
            SF_T = F.normalize(text_f).mm(F.normalize(text_f).t())
            
            p = F.softmax(SF_I / T, dim=1)
            log_q = F.log_softmax(SF_T / T, dim=1)
            kl_t2i = F.kl_div(log_q, p, reduction='batchmean')
            loss_kl = kl_t2i
    
            self.ContrastiveLoss = ContrastiveLoss(batch_size=img.shape[0], device=self.device)
            loss1 = self.ContrastiveLoss(img, img_f)
            loss2 = self.ContrastiveLoss(txt, text_f)
            decomposed_loss=loss1+loss2
            
            img_h = self.ImageMlp(img_f)
            text_h = self.TextMlp(text_f)
            
            SH = F.normalize(img_h).mm(F.normalize(text_h).t())
            SH_I = F.normalize(img_h).mm(F.normalize(img_h).t())
            SH_T = F.normalize(text_h).mm(F.normalize(text_h).t())

            loss_intra=0.5*(F.mse_loss(SF_I, SH_I)+F.mse_loss(SF_T, SH_T))+0.5*(F.mse_loss(self.eta*SF, SH_I)+F.mse_loss(self.eta*SF, SH_T))
            loss_inter=F.mse_loss(self.eta*torch.ones(img_h.shape[0]).to(self.device), SH.diag())+F.mse_loss(self.eta*SF, SH)
            loss_erra=loss_intra+loss_inter
            
            loss_con = self.ContrastiveLoss(img_h, text_h)

            loss =self.alpha*decomposed_loss+a*loss_kl+self.beta*loss_con+self.gama*loss_erra
            
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