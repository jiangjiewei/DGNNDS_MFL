import torch
import pickle
import os
import gc
import re

def load_pretrain_model(model, args):
    model_dict = model.resnet_101.state_dict()
    print('loading pretrained model from imagenet:')
    resnet_pretrained = torch.load(args.pretrain_model)
    pretrain_dict = {k:v for k, v in resnet_pretrained.items() if not k.startswith('fc')}
    model_dict.update(pretrain_dict)
    model.resnet_101.load_state_dict(model_dict)
    del resnet_pretrained
    del pretrain_dict
    gc.collect()
    return model

def load_pretrain_model_dense(model, args):
    model_dict = model.densenet121.state_dict()
    # print(model_dict.keys())
    # with open("model_dict.txt", 'w+') as fp:
    #     fp.write(str(model_dict.keys()))
    print('loading pretrained model from imagenet:')
    densenet_pretrained = torch.load(args.pretrain_model)
    # pretrain_dict = {k:v for k, v in densenet_pretrained.items() if not k.startswith('classifier')}
    pretrain_dict = {re.sub(r'\.(\d)\.', r'\1.', k): v for k, v in densenet_pretrained.items()}
    # print(pretrain_dict.keys())
    # with open("pretrain_dict.txt", 'w+') as fp:
    #     fp.write(str(pretrain_dict.keys()))
    model_dict.update(pretrain_dict)
    model.densenet121.load_state_dict(model_dict)
    del densenet_pretrained
    del pretrain_dict
    gc.collect()
    return model
