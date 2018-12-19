from torch.utils.data import dataloader
from importlib import import_module
from torchvision import transforms
from dataset import Dataset


class Data:
    def __init__(self, args):

        transform = {'train': transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
            'test': transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'query': transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])}


        self.transforms=transform

        self.dataset={ name:Dataset(args,transform[name],name) for name in ['train','test','query'] }

        self.dataloader={}
        self.dataloader['train']=dataloader.DataLoader(self.dataset['train'],
                                                          shuffle=True,
                                                          batch_size=args.batchtrain,
                                                          num_workers=args.nThread)
        self.dataloader['test']=dataloader.DataLoader(self.dataset['test'],
                                                          shuffle=False,
                                                          batch_size=args.batchtest,
                                                          num_workers=args.nThread)
        self.dataloader['query'] = dataloader.DataLoader(self.dataset['query'],
                                                        shuffle=False,
                                                        batch_size=args.batchtest,
                                                        num_workers=args.nThread)

