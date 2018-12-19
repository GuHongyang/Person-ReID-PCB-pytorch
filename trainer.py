import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from metrics import cmc_map,cmc,mean_ap
import numpy as np
import csv
import os
import time
import shutil


class Trainer:
    def __init__(self,args,model,data):
        self.args=args
        self.model=model

        self.train_data=data.dataset['train']
        self.test_data=data.dataset['test']
        self.query_data=data.dataset['query']

        self.train_dataloader=data.dataloader['train']
        self.test_dataloader=data.dataloader['test']
        self.query_dataloader=data.dataloader['query']

        savedir=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        os.mkdir(savedir)
        f=open(savedir+'/log.csv','w',newline='')
        self.savedir=savedir
        self.writer=csv.writer(f)
        self.writer.writerow(['Epoch','Loss','mAP','Rank1','Rank3','Rank5','Rank10'])
        self.best_mAP=0

        self.str=[]

        f1 = open(savedir+'/config.txt','w')
        keys=self.args.__dict__.keys()
        for k in keys:
            f1.write('{}:{}\n'.format(k,self.args.__dict__[k]))


        if self.args.load_model!='':
            self.model.get_module().load_state_dict(torch.load(self.args.load_model))



    def train(self):



        if self.args.lr_base!=0:
            params=[
                {'params': filter(lambda p: id(p) in self.model.get_module().ignored_params, self.model.parameters())},
                {'params':filter(lambda p: id(p) in self.model.get_module().base_params, self.model.parameters()),
                 'lr':self.args.lr_base}
            ]
        else:
            params=self.model.parameters()

        if self.args.optimizer=='SGD':
            opti=SGD(params=params,
                     lr=self.args.lr,
                     momentum=self.args.momentum,
                     dampening=self.args.dampening,
                     weight_decay=self.args.weight_decay,
                     nesterov=self.args.nesterov)
        elif self.args.optimizer=='Adam':
            opti = Adam(params=params,
                       lr=self.args.lr,
                        weight_decay=self.args.weight_decay
                        )
        else:
            raise Exception

        lr_s=StepLR(opti,step_size=self.args.decay_every,gamma=self.args.gamma)


        loss_cert=nn.CrossEntropyLoss()



        epoch_bar=tqdm(range(1,self.args.epochs+1))
        for epoch in epoch_bar:
            lr_s.step()

            LOSS=0
            batch_bar=tqdm(self.train_dataloader)
            self.model.train()
            self.model.get_module().mode='train'
            for batch in batch_bar:
                x,y=batch
                x=x.cuda()
                y=y.cuda()

                y_=self.model(x)

                loss=0
                for i in range(self.args.num_part):
                    loss+=loss_cert(y_[i],y)
                    if self.args.share_classifier:
                        break
                loss/=self.args.num_part

                opti.zero_grad()
                loss.backward()
                opti.step()

                LOSS+=loss.item()

                batch_bar.set_description('loss={:.4f}'.format(loss.item()))

            LOSS/=len(self.train_dataloader)
            epoch_bar.set_description('LOSS={:.4f},LR={}'.format(LOSS,lr_s.get_lr()[0]))

            if epoch%self.args.test_every==0:
                self.str=[epoch,LOSS]
                self.test()
            else:
                self.writer.writerow([epoch,LOSS])

        if epoch%self.args.test_every!=0:
            self.test()



    def test(self):
        print('\n\n[TEST]')
        self.model.eval()
        self.model.get_module().mode = 'test'

        gf=self.extract_feature(self.test_dataloader).numpy()
        qf=self.extract_feature(self.query_dataloader).numpy()

        TMP=np.dot(qf,gf.T)
        dist = np.sqrt(2-2*TMP)

        r,m_ap = cmc_map(dist, self.query_data.ids, self.test_data.ids, self.query_data.cameras, self.test_data.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)

        if self.best_mAP<m_ap:
            self.best_mAP=m_ap
            torch.save(self.model.get_module().state_dict(),self.savedir+'/best_model.pkl')
        torch.save(self.model.get_module().state_dict(), self.savedir + '/latest_model.pkl')

        self.str.extend([m_ap,r[0], r[2], r[4], r[9]])
        self.writer.writerow(self.str)

        print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
                m_ap,
                r[0], r[2], r[4], r[9]
            )
        )




    def fliphor(self, x):
        inv_idx = torch.arange(x.size(3)-1,-1,-1).long()  # N x C x H x W
        return x.index_select(3,inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        if self.args.test_flip:
            T=2
        else:
            T=1

        for (inputs, labels) in tqdm(loader):
            ff = torch.FloatTensor(inputs.size(0), self.args.num_part*self.args.feats).zero_()
            for i in range(T):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to('cuda')
                outputs = self.model(input_img)
                f = outputs.data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
        return features






