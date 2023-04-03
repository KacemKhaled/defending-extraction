import torch
import wandb
from torch import nn, optim
import numpy as np
import time
import tqdm
from itertools import chain
from torch.backends import cudnn
import argparse
from defenses import *
from df_utils import *
import pickle
import os
from itertools import chain
import cfg
import pandas as pd

cudnn.benchmark = True
print('GPU Available:',torch.cuda.is_available())
# surrogate.module. --> surrogate.

############################### CIFAR EXPERIMENTS #################################

def main(args):

    run = wandb.init(
        project=cfg.WB_PROJECT,
        entity=cfg.WB_ENTITY,
        group=args.eval_data,  # or args.eval_data
        tags=['queries', args.defense, args.eval_data, args.transfer_data],
        name=args.exp_name,
        config=args, settings=wandb.Settings(start_method="fork"))
    assert not (args.defense == 'None' and len(args.epsilons.split()) != 1), 'set --epsilons 0 with --defense None'
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    # if eval_perturbations flag is True, swap transfer dataset for eval dataset's test split,
    # but don't change args.transfer_data (b/c some defenses use surrogates specific to the transfer set)
    transfer_data, _ = load_data(args.eval_data if args.eval_perturbations else args.transfer_data,
                                 train=not args.eval_perturbations, deterministic=True)
    eval_data, num_classes = load_data(args.eval_data, train=False)
    quantized = "quantized" in args.load_path

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    device = cpu_device if quantized else cuda_device

    if quantized:
        teacher = load_torchscript_model(model_filepath=args.load_path, device=cpu_device)
    else:
        teacher = create_model(args.eval_data, num_classes, quantized=quantized)
        # teacher_path = cfg.MODEL_DIR+'/trained_models/{}_teacher.pt'.format(args.eval_data.split('_')[0])
        teacher_path = args.load_path
        assert os.path.exists(teacher_path), 'Expected model in teacher path: {}'.format(teacher_path)
        teacher.load_state_dict(torch.load(teacher_path))
        teacher.eval()
    # Compute defender accuracy



    num_params = 0
    for p in teacher.parameters():  # used for override_grad; assuming teacher and student have same architecture
        num_params += p.numel()

    print('\nRunning defense: {}\nwith epsilons: {}'.format(args.defense, args.epsilons))
    epsilons = [float(eps) for eps in args.epsilons.split()]
    elapsed_time_ave = 0.0
    elapsed_time_clean = 0.0
    if args.defense == 'MAD':
        surrogate = create_model(args.eval_data, num_classes)
        surr_root = cfg.MODEL_DIR+'/trained_models'
        surrogate.load_state_dict(torch.load('{}/{}_to_{}_surrogate_{}epochs.pt'.format(surr_root, args.transfer_data, args.eval_data, 0)))
        surrogate.eval()

        if args.eval_data in ['cifar10', 'cifar100', 'cub200', 'svhn', 'gtsrb']:
            backprop_modules = [surrogate.fc]  # only target the final layer, per the official MAD GitHub repository
        elif args.eval_data in ['mnist', 'fashionmnist']:
            backprop_modules = [surrogate.f5]
        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, surrogate,
                                               method_orekondy, epsilons=epsilons, batch_size=args.num_gpus*64,  device=device, # this was 4
                                               backprop_modules=backprop_modules)
    elif args.defense in ['NEW','NEW-MIN-IP']:
        # method_gradient_redirection_no_surrogate
        if args.defense in ['NEW']:
            override_grad = -1 * torch.ones(num_params).cuda()  # b/c we are using sample_surrogates
        else:
            print('using MIN-IP method')
            override_grad = False
        backprop_modules = None
        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, None,
                                               method_gradient_redirection_no_surrogate, epsilons=epsilons, device=device,
                                               sample_surrogates=False, override_grad=override_grad,
                                               batch_size=args.num_gpus*64, num_workers=cfg.NUM_WORKERS,
                                               backprop_modules=backprop_modules)

    elif args.defense in ['GRAD']:
        # ============================ LOAD SURROGATES ============================ #
        surrogates = []
        surrogate = create_model(args.eval_data, num_classes)
        surr_root = cfg.MODEL_DIR+'/trained_models'
        tmp_transfer_data = args.eval_data if (len(args.defense.split('_')) == 3 and args.defense.split('_')[2] == 's1') else args.transfer_data
        surrogate_path = '{}/{}_to_{}_surrogate_{}epochs.pt'.format(surr_root, tmp_transfer_data, args.eval_data, args.defense.split('_')[1])
        surrogate.load_state_dict(torch.load(surrogate_path))
        print('Loaded surrogate from', surrogate_path)
        surrogate.eval()
        surrogates.append(surrogate)
        

        # ============================ SELECT BACKPROP_MODULES ============================ #
        # backprop_modules is used to specify which parameters to target in the GRAD^2 or MAD defenses
        # (NOTE: this are not used in the GRAD^2 defense from the paper and is included for completeness)
        backprop_modules = [None for _ in surrogates]

        # ============================ SELECT OVERRIDE_GRAD ============================ #
        # override_grad is used to specify a fixed target direction for the gradient redirection. By default, the gradient_redirection function
        # uses the negative parameters of the student network as a target (min-inner-product, or MIN-IP); this overrides that functionality.
        # This is how the ALL-ONES and watermark experiments from the paper are specified; there are some additional settings as well
        # that didn't make it into the paper.
        tmp_num_params = num_params
        override_grad = [-1 * torch.ones(tmp_num_params).cuda() for i in range(len(surrogates))]  # b/c we are using sample_surrogates
        
        

        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, surrogates,
                                               method_gradient_redirection, epsilons=epsilons, device=device,
                                               sample_surrogates=True, override_grad=override_grad,
                                               batch_size=args.num_gpus*64, num_workers=cfg.NUM_WORKERS,
                                               backprop_modules=backprop_modules)


    elif args.defense == 'DCP':
        gamma = 0.1 if args.eval_data == 'cifar100' else 0.2
        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, None, method_dcp, epsilons=epsilons, device=device,
                                                                 batch_size=args.num_gpus * 64, num_workers=cfg.NUM_WORKERS,
                                                                 gamma=gamma)

    elif args.defense == 'ReverseSigmoid':
        gamma = 0.1 if args.eval_data == 'cifar10' else 0.2
        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, None, method_reverse_sigmoid, epsilons=epsilons, device=device,
                                               batch_size=args.num_gpus * 64, num_workers=cfg.NUM_WORKERS,
                                               gamma=gamma)

    elif args.defense == 'AdaptiveMisinformation':
        oe_model = None
        misinformation_model = create_model(args.eval_data, num_classes)
        misinformation_model_path = cfg.MODEL_DIR+'/trained_models/{}_misinformation.pt'.format(args.eval_data.split('_')[0])
        assert os.path.exists(misinformation_model_path), 'Expected model in misinformation_model path: {}'.format(misinformation_model_path)
        misinformation_model.load_state_dict(torch.load(misinformation_model_path))
        misinformation_model.eval()

        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, None, method_adaptive_misinformation, epsilons=epsilons, device=device,
                                               batch_size=args.num_gpus*64, num_workers=cfg.NUM_WORKERS, oe_model=oe_model, misinformation_model=misinformation_model)
    elif args.defense == 'None':
        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, None, method_no_perturbation, epsilons=None, device=device,
                                               batch_size=args.num_gpus*64, num_workers=cfg.NUM_WORKERS)
    elif args.defense == 'Random':
        perturbations, elapsed_time_ave = generate_perturbations(transfer_data, teacher, None, method_rand_perturbation, epsilons=epsilons, device=device,
                                               batch_size=args.num_gpus*64, num_workers=cfg.NUM_WORKERS, num_classes=num_classes)

    if args.eval_perturbations:
        data, _ = load_data(args.eval_data, train=False)
        print(data)
        # print(data.targets)
        if args.eval_data == 'cub200':
            targets = [data.data.target.iloc[i] - 1 for i in range(len(data))]
        elif args.eval_data == 'svhn':
            targets = data.labels
        else:
            targets = data.targets

    clean_posteriors, elapsed_time_clean = generate_perturbations(transfer_data, teacher, None, method_no_perturbation, epsilons=None,  device=device,
                                           batch_size=args.num_gpus*64, num_workers=cfg.NUM_WORKERS)
    clean_posteriors = clean_posteriors[:, 0, :].data.cpu().numpy()

    latency_times = {"elapsed_time_for_perturbation": elapsed_time_ave, "elapsed_time_no_perturbation": elapsed_time_clean}

    summary = {
        'avg_inference_time_und': np.mean(elapsed_time_clean),
        'min_inference_time_und': min(elapsed_time_clean),
        'max_inference_time_und': max(elapsed_time_clean),
        'std_inference_time_und': np.std(elapsed_time_clean),
        'avg_inference_time_def': np.mean(elapsed_time_ave),
        'std_inference_time_def': np.std(elapsed_time_ave),
        'min_inference_time_def': min(elapsed_time_ave),
        'max_inference_time_def': max(elapsed_time_ave),
    }

    run.log(latency_times)
    run.log(summary)
    summary['dataset'] = args.eval_data
    summary['exp'] = args.exp_name

    df_summary = pd.DataFrame.from_records([summary], index='dataset')

    tbl = wandb.Table(data=df_summary)
    run.log({"table":tbl})

    latencies = zip(elapsed_time_clean,elapsed_time_ave)

    df = pd.DataFrame(latencies,
                      columns=['Time_clean', f'Time_{args.defense}'])

    for i,l in enumerate(latencies):
        run.log({'latency_und':l[0],'latency_defended':l[1],'batch':i})

    df.to_csv(args.save_path.replace('.pkl', '_times.csv'))
    with open(args.save_path.replace('.pkl', '_times.json'), 'w') as f:
        json.dump(latency_times, f)

    perturbations_dict = {}

    for i, eps in enumerate(args.epsilons.split()):
        perturbations_dict[eps] = perturbations[:, i, :].data.cpu().numpy()
        if args.eval_perturbations:
            # evaluation mode: we can compute defender error (1 - acc) on eval data
            post = torch.FloatTensor(perturbations_dict[eps])
            pred = torch.argmax(post, dim=1)
            correct = 100.0 * (pred == torch.LongTensor(targets)).float()
            defender_accuracy = correct.mean()
            wandb.log({"epsilon": float(eps), "defense": args.defense, "defender acc": defender_accuracy, "defender err": 100-defender_accuracy})
        else:
            # here we compute effective epsilon (l1) distance
            effective_epsilon = (torch.FloatTensor(perturbations_dict[eps]) - torch.FloatTensor(clean_posteriors)).abs().sum(1).mean(0).item()
            wandb.log({"epsilon": float(eps), "defense": args.defense, "l1-distance": effective_epsilon })


    print('\nSaving outputs to: {}\n'.format(args.save_path))



    with open(args.save_path, 'wb') as f:
        pickle.dump(perturbations_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get transfer_data used by the adversary.')
    parser.add_argument('--transfer_data', type=str, help='dataset that we query the teacher on')
    parser.add_argument('--eval_data', type=str, help='dataset that we will eval the teacher/adversary on')
    parser.add_argument('--defense', type=str, help='method used by the teacher for defense')
    parser.add_argument('--epsilons', type=str, help='epsilons to use for perturbations', default='')
    parser.add_argument('--save_path', type=str, help='path for saving perturbed posteriors')
    parser.add_argument('--load_path', type=str, help='path for loading the teacher model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for generating perturbations', default=1)
    parser.add_argument('--eval_perturbations', action='store_true', help='if true, generate perturbations on val set of transfer dataset')
    parser.add_argument('--watermark_seed', type=int, help='random seed to use for selecting the watermark input-output pair', default=3)

    parser.add_argument('--exp_name', type=str, help='name for exepriment', default='perturbations')
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=32)
    parser.add_argument('--gamma', type=int, help='gamma fo generating perturbations', default=0)


    args = parser.parse_args()
    print(args)

    main(args)
