from threading import currentThread
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import tqdm
# from tqdm import tqdm
from torch.backends import cudnn
import pickle
import argparse
from defenses import *
from df_utils import *
from torchmetrics import Accuracy
import logging
import os
import wandb
import cfg


logging.basicConfig(level=logging.INFO)
cudnn.benchmark = True

class ParamsCfg():
    def __init__(self,cuda,learning_rate,num_epochs) -> None:
        self.cuda=cuda
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs




# ************************** training function **************************
def train_epoch(model, optim, sched, loss_fn, data_loader, params):
    model.train()
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    with tqdm.tqdm(total=len(data_loader), miniters=int(len(data_loader)/100)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if params.cuda:
                train_batch = train_batch.cuda()        # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output and loss
            output_batch = model(train_batch)           # logit without softmax
            loss = loss_fn(output_batch, labels_batch)
            acc = 100.0 * np.sum(np.argmax(output_batch.detach().cpu().numpy(), axis=1) == labels_batch.cpu().numpy()) / float(labels_batch.cpu().numpy().shape[0])

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

            # update the average loss
            loss_avg.update(loss.item())
            acc_avg.update(acc.item())

            # tqdm setting
            t.set_postfix(loss=f'{loss_avg():05.3f}',acc=f'{acc_avg():05.3f}')
            # t.update()
    return loss_avg()





def train_and_eval(model, optim, sched, loss_fn, train_loader, dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate

    for epoch in range(params.num_epochs):
        # LR schedule *****************
        # lr = adjust_learning_rate(optim, epoch, lr, params)

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss = train_epoch(model, optim, sched, loss_fn, train_loader, params)
        logging.info("- Train loss : {:05.3f}".format(train_loss))

        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, loss_fn, dev_loader, params)     # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)

        # save last epoch model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, f'{args.dataset}.pt')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))


############################## DATASET AND TRAINING CODE ##################################
def train(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, print_every=100):
    """
    Trains the provided model
    
    :param model: the student model to train with distillation
    :param loader: the data loader for distillation; the dataset was created with make_distillation_dataset
    :param loss_fn: the loss function to use for distillation
    :param num_epochs: the number of epochs to train for
    """
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    examples_ct = 0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        val_metrics = evaluate(model, test_loader)     # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        print(f"Epoch : {epoch} \n\t- Eval metrics : {metrics_string}")
        log_metrics_val(val_metrics)
        wandb.log({'epoch': epoch})
        model.train()
        # with tqdm.tqdm(total=len(loader), miniters=int(len(loader)/1000)) as t:
        for i, (bx, by) in enumerate(loader):
            bx = bx.cuda()
            by = by.cuda()

            # forward pass
            logits = model(bx)
            loss = loss_fn(logits, by)

            acc = 100.0 * np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == by.cpu().numpy()) / float(by.cpu().numpy().shape[0])


            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # update the average loss
            loss_avg.update(loss.item())
            acc_avg.update(acc.item())

            # tqdm setting
            # t.set_postfix(loss=f'{loss_avg():05.3f}',acc=f'{acc_avg():05.3f}')
            # t.update()
            examples_ct += len(bx)
            if i % print_every == 0:
                # print(i, loss.item())
                log_metrics(acc, loss, examples_ct, epoch)

    
    # loss, acc = evaluate(model, loss_fn, test_loader, params)
    val_metrics = evaluate(model,  test_loader)     # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
    print("- Eval metrics : " + metrics_string)
    log_metrics_val(val_metrics)
    # print('Final:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss, acc))
    model.eval()


################################################################

def cross_entropy_loss(logits, gt_target):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


def main(args):
    wandb.init(
        project=cfg.WB_PROJECT,
        entity=cfg.WB_ENTITY,
        group=args.dataset, # or args.eval_data
        name=args.exp_name,
        tags=['teacher',args.dataset],
        config=args, settings=wandb.Settings(start_method="fork"))

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_filepath = args.save_path


    train_data, _ = load_data(args.dataset, train=True)
    test_data, num_classes = load_data(args.dataset, train=False)

    def misinformation_loss(logits, gt_target):
        """
        :param logits: tensor of shape (N, C); the predicted logits
        :param gt_target: long tensor of shape (N,); the gt class labels
        :returns: cross entropy loss
        """
        smax = torch.softmax(logits, dim=1)
        loss = -1 * (((1 - smax) * torch.nn.functional.one_hot(gt_target, num_classes=num_classes)).sum(1) + 1e-12).log().mean(0)
        return loss
    if args.dataset in ['cub200', 'caltech256','pascal']:
        batch_size = 256
    else:
        batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=cfg.NUM_WORKERS, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=cfg.NUM_WORKERS, shuffle=False, pin_memory=False)

    if args.misinformation:
        loss = misinformation_loss
    else:
        loss = cross_entropy_loss

    ################# TRAINING ####################
    print('\nTraining model on: {}'.format(args.dataset))

    teacher = create_model(args.dataset, num_classes, arch=args.arch)
    # teacher = ResNet18(num_class=num_classes)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # teacher.to(device)
    num_epochs = args.epochs # 50
    lr = 0.01 if args.dataset == 'cub200' else  0.001 #0.05
    optimizer = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))


    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    # if not os.path.exists(args.save_path):
    train(teacher, train_loader, test_loader, optimizer, scheduler, loss, num_epochs=num_epochs)
    print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
    torch.save(teacher.state_dict(), args.save_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get queryset used by the adversary.')
    parser.add_argument('--dataset', type=str, help='dataset that we transfer knowledge with')
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--misinformation', type=int,
        help='if "1", train a network with the misinformation loss for the Adaptive Misinformation method', default=0)
    parser.add_argument('--exp_name', type=str, help='name for exepriment', default='train')
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=32)
    parser.add_argument('--arch', type=str, help='Use VGG architecture for the model', default='resnet')


    args = parser.parse_args()
    print(args)

    main(args)
