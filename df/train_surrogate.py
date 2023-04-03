import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import tqdm
from torch.backends import cudnn
import pickle
import argparse
from defenses import *
from df_utils import *
import os
import cfg

cudnn.benchmark = True




############################## DATASET AND TRAINING CODE ##################################

def train_with_distillation(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, temperature=3, oracle_defense=None, print_every=100):
    """
    Trains the provided model on the distillation dataset
    
    :param model: the student model to train with distillation
    :param loader: the data loader for distillation; the dataset was created with make_distillation_dataset
    :param loss_fn: the loss function to use for distillation
    :param num_epochs: the number of epochs to train for
    :param oracle_defense: if true, use an online perturbation; constitutes an oracular defense method
    """
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    examples_ct = 0


    for epoch in range(num_epochs):
        # with torch.no_grad():
        # loss, acc = evaluate(model, test_loader)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        val_metrics = evaluate(model, test_loader)     # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        print("- Eval metrics : " + metrics_string)
        log_metrics_val(val_metrics)
        wandb.log({'epoch': epoch})
        model.train()

        # print('Epoch: {}, Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, loss, acc))
        
        model.train()
        with tqdm.tqdm(total=len(loader), miniters=int(len(loader)/1000)) as t:
            for i, (tmp1, distill_targets) in enumerate(loader):
                bx, by = tmp1
                bx = bx.cuda()
                by = by.cuda()
                distill_targets = distill_targets.cuda()
                
                if oracle_defense is not None:
                    teacher, epsilons, all_ones = oracle_defense
                    model.eval()
                    distill_targets = method_gradient_redirection(bx, teacher, [model], epsilons=epsilons, backprop_modules=None, all_ones=all_ones)[:,0,:].cuda()
                    model.train()

                # forward pass
                logits = model(bx)
                loss = loss_fn(logits, distill_targets, by, temperature)
                
                # acc = 100.0 * np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == by.cpu().numpy()) / float(by.cpu().numpy().shape[0])
                acc = 100.0 * np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == np.argmax(distill_targets.detach().cpu().numpy(), axis=1)) / float(by.cpu().numpy().shape[0])


                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update the average loss
                loss_avg.update(loss.item())
                acc_avg.update(acc.item())

                # tqdm setting
                t.set_postfix(loss=f'{loss_avg():05.3f}',acc=f'{acc_avg():05.3f}')
                # t.update()

                examples_ct += len(bx)
                if i % print_every == 0:
                    # print(i, loss.item())
                    log_metrics(acc, loss, examples_ct, epoch)

    # with torch.no_grad():
    # loss, acc = evaluate(model, test_loader)
    # print('Final:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss, acc))
    val_metrics = evaluate(model,  test_loader)     # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
    print("- Eval metrics : " + metrics_string)
    log_metrics_val(val_metrics)
    model.eval()


################################################################

def main(args):
    wandb.init(
        project=cfg.WB_PROJECT,
        entity=cfg.WB_ENTITY,
        group=args.eval_data,  # or args.eval_data
        tags=['surrogate', args.eval_data],
        name=args.exp_name,
        config=args, settings=wandb.Settings(start_method="fork"))

    transfer_data, _ = load_data(args.transfer_data, train=True)
    transfer_data_deterministic, _ = load_data(args.transfer_data, train=True, deterministic=True)
    eval_data, num_classes = load_data(args.eval_data, train=False)
    
    test_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.num_gpus*128, num_workers=cfg.NUM_WORKERS, shuffle=False, pin_memory=True)

    loss = distillation_loss_clf

    ################# GENERATING PERTURBATIONS #################
    print('\nGenerating perturbations on transfer set')

    teacher = create_model(args.eval_data, num_classes)
    teacher_path = f'{cfg.MODEL_DIR}/trained_models/{args.eval_data}_teacher.pt'
    assert os.path.exists(teacher_path), 'Expected model in teacher path: {}'.format(teacher_path)
    teacher.load_state_dict(torch.load(teacher_path))
    teacher.eval()

    perturbations, _ = generate_perturbations(transfer_data_deterministic, teacher, None, method_no_perturbation, epsilons=None,
                                           batch_size=args.num_gpus*128, num_workers=cfg.NUM_WORKERS)

    transfer_data = make_distillation_dataset(transfer_data, perturbations[:,0,:])

    ################# TRAINING #################
    print('\nUsing testing dataset: {}\n(training/transfer dataset is different)'.format(args.eval_data))

    student = create_model(args.eval_data, num_classes)
    num_epochs = args.epochs # 50 # 50
    lr = 0.01 if args.eval_data == 'cub200' else 0.1
    batch_size = 32 if args.eval_data == 'cub200' else 128
    temperature = args.distillation_temperature
    train_loader = torch.utils.data.DataLoader(transfer_data, shuffle=True, pin_memory=True,
                                        batch_size=batch_size, num_workers=cfg.NUM_WORKERS)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))
    early_stopping_epoch = num_epochs if args.early_stopping_epoch == -1 else args.early_stopping_epoch

    train_with_distillation(student, train_loader, test_loader, optimizer, scheduler, loss,
                            num_epochs=early_stopping_epoch, temperature=temperature, oracle_defense=None)

    print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
    torch.save(student.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get queryset used by the adversary.')
    parser.add_argument('--transfer_data', type=str, help='dataset that we train the teacher/adversary with')
    parser.add_argument('--eval_data', type=str, help='dataset that we eval the teacher/adversary with')
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--distillation_temperature', type=float, help='temperature to use for distillation', default=1)
    parser.add_argument('--early_stopping_epoch', type=int, help='epoch to use for early stopping; -1 means no early stopping', default=-1)

    parser.add_argument('--exp_name', type=str, help='name for exepriment', default='train surrogate')
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=32)

    args = parser.parse_args()
    print(args)

    main(args)
