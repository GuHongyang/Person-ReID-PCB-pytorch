import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch.nn.init as init





def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class PCB(nn.Module):
    def __init__(self,args):
        super(PCB,self).__init__()

        self.part = args.num_part

        resnet = resnet50(pretrained=True)

        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        if args.pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((self.part, 1))
        elif args.pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d((self.part,1))
        else:
            raise Exception

        self.Feature=nn.ModuleList()
        for i in range(self.part):
            Conv=nn.Sequential(
                nn.Conv2d(2048,args.feats,kernel_size=1),
                nn.BatchNorm2d(args.feats),
                nn.ReLU(inplace=True)
            )
            Conv.apply(weights_init_kaiming)

            self.Feature.append(Conv)

            if args.share_feature:
                break

        self.Classifer=nn.ModuleList()
        for i in range(self.part):
            if args.share_classifier:
                Classifer = nn.Sequential(
                    nn.Linear(args.feats*args.num_part, args.num_classes, bias=True)
                )
            else:
                Classifer=nn.Sequential(
                    nn.Linear(args.feats,args.num_classes,bias=True)
                )

            Classifer.apply(weights_init_classifier)

            self.Classifer.append(Classifer)

            if args.share_classifier:
                break


        self.ignored_params=list(map(id,self.Feature.parameters()))
        self.ignored_params.extend(list(map(id,self.Classifer.parameters())))
        self.base_params=list(map(id,self.backbone.parameters()))

        self.mode='train'


    def forward(self, x):
        f0=self.pool(self.backbone(x))
        f=[]
        for i in range(self.part):
            if len(self.Feature)==1:
                f.append(self.Feature[0](f0[:, :, i:i + 1, :]).squeeze(2).squeeze(2))
            else:
                f.append(self.Feature[i](f0[:,:,i:i+1,:]).squeeze(2).squeeze(2))

        if self.mode=='test':
            return torch.cat(f,1)

        y=[]
        for i in range(self.part):
            if len(self.Classifer)==1:
                y.append(self.Classifer[0](torch.cat(f,1)))
                break
            else:
                y.append(self.Classifer[i](f[i]))

        return y


