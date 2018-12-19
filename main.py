from data import Data
from model import Model
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description='Person ReID Frame')

"""
System parameters
"""
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only',default=False)
parser.add_argument('--nGPU', type=int, default=4, help='number of GPUs')

"""
Data parameters
"""
parser.add_argument("--datadir", type=str, default="/home/guhongyang/DATASETS/Market-1501-v15.09.15", help='dataset directory')
parser.add_argument("--batchtrain", type=int, default=64, help='input batch size for test')
parser.add_argument("--batchtest", type=int, default=128, help='input batch size for test')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--num_classes', type=int, default=751, help='num classes')
parser.add_argument('--test_flip',type=bool,default=True)


"""
Model parameters
"""
parser.add_argument('--model',type=str,default='PCB')
parser.add_argument('--feats',type=int,default=256,help='reduce dims')
parser.add_argument('--num_part',type=int,default=6,help='num of part for PCB')
parser.add_argument('--load_model',type=str,default='')
parser.add_argument('--pool',type=str,default='avg',choices=['avg','max'])
parser.add_argument('--share_classifier',type=bool,default=False,help='cat all the features into a classifier')
parser.add_argument('--share_feature',type=bool,default=False)


"""
Train parameters
"""
parser.add_argument('--epochs',type=int,default=60)
parser.add_argument('--test_every',type=int,default=10)


"""
Optimizer parameters
"""
parser.add_argument('--optimizer',type=str,default='SGD',choices=['SGD','Adam'])
parser.add_argument('--lr',type=float,default=0.1)
parser.add_argument('--lr_base',type=float,default=0.01)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--dampening',type=float,default=0)
parser.add_argument('--weight_decay',type=float,default=5e-4)
parser.add_argument('--nesterov',type=bool,default=True)


"""
Learning rate parameters
"""
parser.add_argument('--decay_every',type=int,default=20)
parser.add_argument('--gamma',type=float,default=0.1)



args = parser.parse_args()



from torchsummary import summary

datas=Data(args)
model=Model(args)
trainer=Trainer(args,model,datas)
trainer.train()
