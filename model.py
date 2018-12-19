import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()

        module=import_module(args.model)
        self.model=getattr(module,args.model)(args)

        if not args.cpu:
            self.model=self.model.to('cuda')


        if args.nGPU>1:
            self.model=nn.DataParallel(self.model,device_ids=range(args.nGPU))

        self.args=args

    def forward(self, x):
        return self.model(x)


    def get_module(self):
        if self.args.nGPU>1 and not self.args.cpu:
            return self.model.module
        else:
            return self.model


