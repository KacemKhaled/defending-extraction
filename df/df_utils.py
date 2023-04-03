# Adapted from : https://github.com/mmazeika/model-stealing-defenses
import torch
import numpy as np
from wrn import WideResNet
from torchvision import datasets, transforms, models
from gtsrb import GTSRB
from lenet import LeNet5
from cub200_dataset import CUB200
from caltech256 import Caltech256
from pascal import PascalVOC_Dataset, encode_labels
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy
import tqdm
from model import resnet18, resnet34, resnet_8x #, vgg16_bn, vgg16#, VGG16
from torchvision.models import vgg16
import wandb
import cfg
import time
from model.resnet_q import resnet18 as resnet18_q
from model.resnet_q import resnet50 as resnet50_q
from model.resnet_q import resnet34 as resnet34_q

from cifarlike import SVHN


"""
helper functions:
https://github.com/cs230-stanford/cs230-stanford.github.io
"""

import json
import logging
import os
import shutil
import torch

import numpy as np
import scipy.misc

from io import BytesIO  # Python 3.x


def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 3, 32, 32)):
    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)

    return model


def log_metrics(acc, loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "train_loss": loss, "train_acc":acc}, step=example_ct)
    # wandb.log({"epoch": epoch, "train_loss": loss, "train_acc":acc})


def log_metrics_val(metrics):
    wandb.log(metrics)

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def get_model_accuracy(eval_data, model_path):
    print(eval_data)
    test_data, num_classes = load_data(eval_data, train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, num_workers=12, shuffle=False,
                                              pin_memory=False)

    model = create_model(eval_data, num_classes)
    assert os.path.exists(model_path), 'Expected model in path: {}'.format(model_path)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    val_metrics = evaluate(model, test_loader)
    print(f"{model_path.split('/')[-1]} : {val_metrics}")
    json_path = model_path[:-3] + '.json'
    with open(json_path, 'w') as f:
        json.dump(val_metrics, f)

    return float(val_metrics['acc'] / 100)

############################## TRAINING/EVAL ##############################

def evaluate(model, data_loader,loss_fn=F.cross_entropy, device=torch.device("cuda:0"),suffix="",*args):
    model.eval()
    model.to(device)

    # summary for current eval loop
    summ = []
    print("Validation")

    with torch.no_grad():
        # compute metrics over the dataset
        # with tqdm.tqdm(total=len(data_loader)) as t: 
        for data_batch, labels_batch in data_loader:
            # if params.cuda:
            data_batch = data_batch.to(device)         # (B,3,32,32)
            labels_batch = labels_batch.to(device)     # (B,)

            # compute model output
            output_batch = model(data_batch)
            # print(output_batch, labels_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {f'val_acc{suffix}': acc.item(), f'val_loss{suffix}': loss.item()}
            summ.append(summary_batch)
            # tqdm setting
            # t.set_postfix(loss=f'{loss.item():05.3f}',acc=f'{acc:05.3f}')
            # t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean


def get_file_size(file_path):
    """ Get file in size in MB"""
    size = os.path.getsize(file_path)
    return size / (1024*1024)


def create_model(dataset_name, num_classes, drop_rate=0.3, arch='resnet', quantized=False):
    dataset_name = dataset_name.split('_')[0]
    if dataset_name in ['cifar10', 'cifar100', 'svhn', 'gtsrb']:
        if quantized:
            model = resnet18_q(num_classes=num_classes, pretrained=False)
        elif 'vgg' in arch:
            model = vgg16(num_classes=num_classes).cuda()
        elif 'wideresnet' in arch:
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=drop_rate).cuda()
        elif 'resnet34_8x' in arch:
            model = resnet_8x.ResNet34_8x(num_classes=num_classes)
        elif 'resnet34' in arch:
            model = resnet34(num_classes=num_classes).cuda()
        elif 'resnet' in arch:
            model = resnet18(num_classes=num_classes).cuda()
        else:
            raise ValueError('{} is an invalid architecture!'.format(arch))
    elif dataset_name in ['mnist', 'fashionmnist']:
        # TODO add if quantized and elif use_vgg_countermeasure:
        model = LeNet5().cuda()
        # model = torch.nn.DataParallel(model)
    elif dataset_name in ['cub200', 'caltech256', 'pascal']:
        model_head = 1000
        if quantized:
            model = resnet34_q(num_classes=model_head, pretrained=True)
        elif 'vgg' in arch:
            # model = models.resnet18(pretrained=False, num_classes=num_classes).cuda()
            model = vgg16(num_classes=model_head,pretrained=True).cuda()
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            # model = torch.nn.DataParallel(model)
        elif 'resnet' in arch:
            model = models.resnet34(num_classes=model_head, pretrained=True).cuda()
            model.fc = torch.nn.Linear(model.fc.weight.shape[1], num_classes)
            # model = torch.nn.DataParallel(model)
        elif 'wideresnet' in arch:
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=drop_rate).cuda()
        else:
            raise ValueError('{} is an invalid architecture!'.format(arch))
    else:
        raise ValueError('{} is an invalid dataset!'.format(dataset_name))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


def distillation_loss_clf(logits, distill_target, gt_target, temperature):
    """
    Takes the mean across a batch of the distillation loss described in
    "Distilling the Knowledge in a Neural Network" by Hinton et al.
    
    Divides the normalized logits of the model undergoing training by the temperature parameter.
    This way, at test time, the temperature is reverted and the model is better calibrated.
    
    :param logits: tensor of shape (N, C); the predicted logits
    :param distill_target: tensor of shape (N, C); the target posterior
    :param gt_target: long tensor of shape (N,); the gt class labels
    :param temperature: scalar; the temperature parameter
    :returns: cross entropy with temperature applied to the target logits
    """
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    
    # distillation loss
    target_logits = torch.clamp(distill_target, min=1e-12, max=1).log() / temperature
    target = torch.softmax(target_logits, dim=1)
    distill_loss = -1 * (normalized_logits/temperature * target).sum(1).mean(0)
    
    # # normal loss
    # normal_loss = F.cross_entropy(logits, gt_target, reduction='mean')
    
    return distill_loss# * (temperature ** 2) + normal_loss


def cross_entropy_loss(logits, distill_target, gt_target, temperature):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


############################## DATA LOADING ##############################

def load_data(dataset_name, train=True, deterministic=False, seed=1,budget=50000):
    if train == False:
        dataset_name = dataset_name.split('_')[0]  # this might cause a bug now that we aren't splitting datasets in two
    
    if dataset_name in ['cub200']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        num_classes = 200
        # osp.join(cfg.DATASET_ROOT, '256_ObjectCategories')
        dataset = CUB200(root=cfg.DATASET_ROOT,
                         train=train, transform=transform, download=False)
    
    elif dataset_name in ['caltech256']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        num_classes = 257
        dataset = Caltech256(root=cfg.DATASET_ROOT,
                         train=train, transform=transform)

        # dataset = datasets.ImageFolder(root=cfg.DATASET_ROOT+'/256_ObjectCategories',
        #                               transform=transform)

    elif dataset_name in ['imagenet_cub200']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        split = 'train' if train else 'val'
        num_classes = 200
        dataset = datasets.ImageFolder(f'{cfg.DATASET_ROOT}/ImageNet_CUB200/{split}', transform=transform)
        shuffle_indices = np.arange(len(dataset))
        rng = np.random.RandomState(seed)
        rng.shuffle(shuffle_indices)
        dataset = torch.utils.data.Subset(dataset, shuffle_indices[:30000])  # for comparability with Caltech256

    elif dataset_name in ['pascal']:
        # Imagnet values
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

        #    mean=[0.485, 0.456, 0.406]
        #    std=[0.229, 0.224, 0.225]
        if train and not deterministic:
            transform = transforms.Compose([transforms.Resize((300, 300)),
                                                   # transforms.RandomChoice([
                                                   #         transforms.CenterCrop(300),
                                                   #         transforms.RandomResizedCrop(300, scale=(0.80, 1.0)),
                                                   #         ]),
                                                  transforms.RandomChoice([
                                                      transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                      transforms.RandomGrayscale(p=0.25)
                                                  ]),
                                                  transforms.RandomHorizontalFlip(p=0.25),
                                                  transforms.RandomRotation(25),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std),
                                                  ])
        else:
            transform = transforms.Compose([transforms.Resize(330),
                                                    transforms.CenterCrop(300),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std),
                                                    ])

        num_classes = 20
        split = 'train' if train else 'val'


        # Create validation dataloader
        dataset = PascalVOC_Dataset(cfg.DATASET_ROOT,
                                          year='2012',
                                          image_set=split,
                                          download=False,
                                          transform=transform,
                                          target_transform=encode_labels)

        shuffle_indices = np.arange(len(dataset))
        # rng = np.random.RandomState(seed)
        # rng.shuffle(shuffle_indices)
        # if dataset_name == 'gtsrb_1':
        #     dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
        # elif dataset_name == 'gtsrb_2':
        #     dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])

    elif dataset_name in ['imagenet_cifar10', 'imagenet_cifar100']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        
        split = 'train' if train else 'val'
        if dataset_name in ['imagenet_cifar10']:
            num_classes = 10
            dataset = datasets.ImageFolder(f'{cfg.DATASET_ROOT}/ImageNet_CIFAR10/{split}', transform=transform)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            dataset = torch.utils.data.Subset(dataset, shuffle_indices[:budget])
        elif dataset_name in ['imagenet_cifar100']:
            num_classes = 100
            dataset = datasets.ImageFolder(f'{cfg.DATASET_ROOT}/ImageNet_CIFAR100/{split}', transform=transform)
    
    elif dataset_name in ['svhn','cifar10','gtsrb', 'cifar10_1', 'cifar10_2', 'cifar100', 'cifar100_1', 'cifar100_2']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        
        if dataset_name in ['cifar10', 'cifar10_1', 'cifar10_2']:
            num_classes = 10
            dataset = datasets.CIFAR10(cfg.DATASET_ROOT, train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'cifar10_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'cifar10_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
        elif dataset_name in ['cifar100', 'cifar100_1', 'cifar100_2']:
            num_classes = 100
            dataset = datasets.CIFAR100(cfg.DATASET_ROOT, train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'cifar100_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'cifar100_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
        elif dataset_name in ['svhn', 'svhn_1', 'svhn_2']:
            num_classes = 10
            split = 'train' if train else 'test'
            dataset = datasets.SVHN(cfg.DATASET_ROOT, split=split, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'svhn_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'svhn_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])

        elif dataset_name in ['gtsrb', 'gtsrb_1', 'gtsrb_2']:
            num_classes = 43
            split = 'train' if train else 'test'
            dataset = GTSRB(cfg.DATASET_ROOT, train=train, transform=transform, download=True)
            # dataset = GTSRB(cfg.DATASET_ROOT, train=train, transform=transform)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'gtsrb_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'gtsrb_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])



    elif dataset_name in ['mnist', 'mnist_1', 'mnist_2', 'fashionmnist', 'fashionmnist_1', 'fashionmnist_2']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        
        if dataset_name in ['mnist', 'mnist_1', 'mnist_2']:
            num_classes = 10
            dataset = datasets.MNIST(cfg.DATASET_ROOT, train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'mnist_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'mnist_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
        elif dataset_name in ['fashionmnist', 'fashionmnist_1', 'fashionmnist_2']:
            num_classes = 10
            dataset = datasets.FashionMNIST(cfg.DATASET_ROOT, train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'fashionmnist_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'fashionmnist_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
    elif dataset_name in ['Imagenet1k32']:

        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),  # this is the old version
                # transforms.Resize((224, 224)), # this is a new line
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        dataset = ImageNet1k32(root=cfg.DATASET_ROOT,
                         train=True, transform=transform, download=False)

    return dataset, num_classes


class ZippedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, distillation_targets):
            super().__init__()
            self.dataset = dataset
            self.distillation_targets = distillation_targets
            assert len(dataset) == len(distillation_targets), 'Should have same length'
    
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx], self.distillation_targets[idx]


def make_distillation_dataset(dataset, distillation_targets):
    """
    Takes a PyTorch dataset D1 and outputs a new dataset D2, where D2[i] = (D1[i], distillation_targets[i])
    
    :param dataset: a PyTorch dataset
    :param distillation_targets: a list of distillation targets
    :returns: the new dataset
    """
    return ZippedDataset(dataset, distillation_targets)


def get_watermark_batch(test_data, num_classes, num_watermarks=1, seed=1):
    rng = np.random.RandomState(seed)

    watermark_by = torch.zeros(num_watermarks, num_classes).cuda()
    for i in range(num_watermarks):
        watermark_by[i][rng.choice(num_classes)] = 1

    watermark_bx = []
    indices = rng.choice(len(test_data), size=num_watermarks, replace=False)
    for i in range(num_watermarks):
        watermark_bx.append(test_data[indices[i]][0].cuda())
    watermark_bx = torch.stack(watermark_bx)
    
    return watermark_bx, watermark_by