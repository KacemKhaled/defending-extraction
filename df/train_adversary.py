# Modified from : https://github.com/mmazeika/model-stealing-defenses
mport argparse
import pickle

from torch import optim
from torch.backends import cudnn
from torchmetrics import Accuracy
from defenses import *
from df_utils import *

cudnn.benchmark = True


############################## DATASET AND TRAINING CODE ##############################

def train_with_distillation(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, temperature=3,
                            oracle_defense=None, print_every=100,
                            save_every_epoch=False, save_path=None, use_argmax_countermeasure=False):
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
    accuracy = Accuracy()
    examples_ct = 0

    for epoch in range(num_epochs):
        if (save_every_epoch == True) and (save_path != None):
            model.eval()
            torch.save(model.state_dict(), args.save_path.split('.pt')[0] + f'_{epoch}epochs.pt')
            model.train()

        # with torch.no_grad():
        # loss, acc = evaluate(model, test_loader)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        val_metrics = evaluate(model, test_loader)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        print("- Eval metrics : " + metrics_string)
        log_metrics_val(val_metrics)
        wandb.log({'epoch': epoch})
        model.train()
        # print(f'Epoch: {epoch}, Test Loss: {loss_avg():05.3f}, Test Acc: {acc_avg():05.3f}')
        with tqdm.tqdm(total=len(loader), miniters=int(len(loader) / 1000)) as t:

            for i, (tmp1, distill_targets) in enumerate(loader):
                bx, by = tmp1
                bx = bx.cuda()
                by = by.cuda()
                distill_targets = distill_targets.cuda()
                if use_argmax_countermeasure:
                    tmp = torch.zeros_like(distill_targets)
                    tmp[range(len(distill_targets)), distill_targets.argmax(dim=1)] = 1
                    distill_targets = tmp

                if oracle_defense is not None:
                    teacher, epsilons, override_grad, watermark, backprop_modules = oracle_defense
                    model.eval()
                    if watermark is not False:  # in watermarking experiments, get the watermark grad here and pass it in as an override
                        watermark_bx, watermark_by = watermark
                        override_grad = -1 * get_Gty(watermark_bx, model, watermark_by).detach()
                    distill_targets = method_gradient_redirection(bx, teacher, [model], epsilons=epsilons,
                                                                  backprop_modules=backprop_modules,
                                                                  override_grad=override_grad)[:, 0, :].cuda()
                    model.train()
                    model.zero_grad()

                # forward pass
                logits = model(bx)
                loss = loss_fn(logits, distill_targets, by, temperature)

                acc = 100.0 * np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == np.argmax(
                    distill_targets.detach().cpu().numpy(), axis=1)) / float(by.cpu().numpy().shape[0])



                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_avg.update(loss.item())
                acc_avg.update(acc.item())

                examples_ct += len(bx)
                if i % print_every == 0:
                    # print(i, loss.item())
                    log_metrics(acc, loss, examples_ct, epoch)

    val_metrics = evaluate(model, test_loader)  # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
    print("- Eval metrics : " + metrics_string)
    log_metrics_val(val_metrics)
    model.eval()

    if save_every_epoch == True and (save_path is not None):
        model.eval()
        torch.save(model.state_dict(), args.save_path.split('.pt')[0] + f'_{num_epochs}epochs.pt')
        model.train()
    elif save_every_epoch == False and save_path is not None:
        print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
        torch.save(model.state_dict(), args.save_path)

    model.eval()
    return val_metrics


################################################################

def main(args):
    run = wandb.init(
        project=cfg.WB_PROJECT,
        entity=cfg.WB_ENTITY,
        group=args.eval_data,  # or args.eval_data
        tags=['adversary', args.eval_data, args.defense],
        name=args.exp_name,
        config=args, settings=wandb.Settings(start_method="fork"))

    # epsilons = [int(e) for e in args.epsilons.split(' ')]

    use_argmax_countermeasure = False
    if args.load_path.split('.pkl')[0].split('_')[-1] == 'argmax':
        args.load_path = args.load_path[:-11] + '.pkl'
        use_argmax_countermeasure = True


    transfer_data, _ = load_data(args.transfer_data, train=True)
    eval_data, num_classes = load_data(args.eval_data, train=False)
    if args.eval_data in ['cub200', 'caltech256','pascal']:
        batch_size = 32
    else:
        batch_size = 64
    test_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, num_workers=cfg.NUM_WORKERS,
                                              shuffle=False, pin_memory=True)

    with open(args.load_path, 'rb') as f:
        perturbations_dict = pickle.load(f)
    perturbations = torch.FloatTensor(perturbations_dict[args.epsilon])
    print(len(transfer_data), len(perturbations))
    # print(transfer_data[0], perturbations[0])
    transfer_data = make_distillation_dataset(transfer_data, perturbations)

    if args.loss == 'cross-entropy':
        loss = cross_entropy_loss
    elif args.loss == 'distillation':
        loss = distillation_loss_clf

    ################# TRAINING #################
    print('\nTraining model on: {}\nwith transfer_data: {}\nwith eval_data: {}\n'.format(args.load_path,
                                                                                         args.transfer_data,
                                                                                         args.eval_data))
    print(f"epsilon: {args.epsilon}")
    student = create_model(args.eval_data, num_classes, arch=args.arch)
    num_epochs = args.epochs  # 50 # 50
    lr = 0.01 if args.eval_data == 'cub200' else 0.1
    batch_size = 32 if args.eval_data == 'cub200' else 64

    # lr = 0.01 if use_vgg_countermeasure else lr  # assumes CIFAR data
    temperature = args.distillation_temperature
    loader = torch.utils.data.DataLoader(transfer_data, shuffle=True, pin_memory=True,
                                         batch_size=batch_size, num_workers=cfg.NUM_WORKERS)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(loader))

    if args.oracle != 'None':
        teacher = create_model(args.eval_data, num_classes)
        teacher_path = cfg.MODEL_DIR + '/trained_models/{}_teacher.pt'.format(args.eval_data)
        assert os.path.exists(teacher_path), 'Expected model in teacher path: {}'.format(teacher_path)
        teacher.load_state_dict(torch.load(teacher_path))
        teacher.eval()

        num_params = 0
        for p in teacher.parameters():  # used for override_grad; assuming teacher and student have same architecture
            num_params += p.numel()

        override_grad = False
        watermark = False
        if args.oracle in ['ALL-ONES', 'ALL-ONES_focused']:
            override_grad = -1 * torch.ones(num_params).cuda()
        elif args.oracle in ['WATERMARK']:
            watermark_seed = int(args.save_path.split('_')[-1].split('.')[0])
            watermark_bx, watermark_by = get_watermark_batch(eval_data, num_classes, num_watermarks=1,
                                                             seed=watermark_seed)
            watermark = (watermark_bx, watermark_by)

        backprop_modules = [student.module.conv1] if args.oracle in ['MIN-IP_focused', 'ALL-ONES_focused'] else None
        oracle_args = (teacher, [float(args.epsilon)], override_grad, watermark, backprop_modules)
    else:
        oracle_args = None
    if args.skip_training:
        # load the model and find the accuracy
        student_path = args.save_path
        assert os.path.exists(student_path), 'Evaluation Only Mode: Expected model in student path: {}'.format(
            student_path)
        student.load_state_dict(torch.load(student_path))
        student.eval()

        val_metrics = evaluate(student, test_loader)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        print("- Eval metrics : " + metrics_string)
        log_metrics_val(val_metrics)

    else:
        # We train the model from scratch
        val_metrics = train_with_distillation(student, loader, test_loader, optimizer, scheduler, loss,
                                              num_epochs=num_epochs, temperature=temperature, oracle_defense=oracle_args,
                                              save_every_epoch=args.save_every_epoch, save_path=args.save_path,
                                              use_argmax_countermeasure=use_argmax_countermeasure)

    with open(args.load_path, 'rb') as f:
        perturbations_dict = pickle.load(f)
    perturbations = torch.FloatTensor(perturbations_dict[args.epsilon])
    # if args.eval_perturbations:

    data, _ = load_data(args.eval_data, train=False)
    if args.eval_data == 'cub200':
        targets = [data.data.target.iloc[i] - 1 for i in range(len(data))]
    elif args.eval_data == 'svhn':
        targets = data.labels
    else:
        targets = data.targets

    with open(args.clean_posteriors_path, 'rb') as f:
        clean_posteriors = pickle.load(f)
    clean_posteriors = torch.FloatTensor(clean_posteriors['0.0'])

    with open(args.load_path.replace('.pkl', '_val.pkl'), 'rb') as f:
        val_posteriors = pickle.load(f)
    val_posteriors = torch.FloatTensor(val_posteriors[args.epsilon])

    # evaluation mode: we can compute defender error (1 - acc) on eval data
    post = val_posteriors
    pred = torch.argmax(post, dim=1)
    correct = 100.0 * (pred == torch.LongTensor(targets)).float()
    defender_accuracy = correct.mean()
    # compute agreement:
    student.eval()
    # summary for current eval loop
    outs = []
    print("Validation")
    with torch.no_grad():
        for data_batch, labels_batch in tqdm.tqdm(test_loader):
            # if params.cuda:
            data_batch = data_batch.cuda()
            # compute model output
            output_batch = student(data_batch)
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.unsqueeze(1).detach()
            outs.append(output_batch.cpu().detach())
    outs = torch.cat(outs, dim=0)
    outs = torch.FloatTensor(outs[:, 0, :].data.cpu().numpy())
    outs = torch.argmax(outs, dim=1)
    agreement = 100.0 * (pred == outs).float().mean()

    # here we compute effective epsilon (l1) distance
    effective_epsilon = (perturbations - clean_posteriors).abs().sum(1).mean(0).item()

    metrics_to_log = \
        {  # "defense": args.defense,
            "epsilon": args.epsilon,
            "l1_distance": effective_epsilon,
            "defender_acc": defender_accuracy,
            "adversary_acc": val_metrics['val_acc'],
            "agreement": agreement,
            "defender_err": 100 - defender_accuracy,
            "adversary_err": 100 - val_metrics['val_acc'],
        }
    metrics_to_log = {key: float(val) for key, val in metrics_to_log.items()}
    table = wandb.Table(
        columns=["epsilon", "defense", "l1_distance", 'defender_acc', "adversary_acc", "agreement", "defender_err",
                 "adversary_err",  "agreement_err"
                 ])
    table.add_data(args.epsilon, args.defense, effective_epsilon, defender_accuracy, val_metrics['val_acc'], agreement,
                   100 - defender_accuracy, 100 - val_metrics['val_acc'], 100 - agreement)
    run.log({"extraction_results": table})

    json_path = args.log_file

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            metrics[args.epsilon] = metrics_to_log
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=1)
        print('updated json file', json_path)
    else:
        with open(json_path, 'w', encoding='utf-8') as f:
            metrics = {args.epsilon: metrics_to_log}
            json.dump(metrics, f, indent=1)
        print('created json file', json_path)

    if args.save_every_epoch == False:
        print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
        torch.save(student.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get queryset used by the adversary.')
    parser.add_argument('--transfer_data', type=str, help='dataset that we transfer knowledge with')
    parser.add_argument('--eval_data', type=str, help='dataset that we eval with; the teacher was trained on this')
    parser.add_argument('--load_path', type=str, help='path for loading perturbed posteriors (at multiple epsilons)')
    parser.add_argument('--clean_posteriors_path', type=str, help='path for loading clean posteriors')
    parser.add_argument('--oracle', type=str, help='whether to use oracle',
                        choices=['None', 'MIN-IP', 'ALL-ONES', 'MIN-IP_focused', 'ALL-ONES_focused', 'WATERMARK'],
                        default='None')
    parser.add_argument('--epsilon', type=str, help='epsilon of perturbations to use for training')
    parser.add_argument('--loss', type=str, help='loss to use for training', choices=['cross-entropy', 'distillation'],
                        default='distillation')
    parser.add_argument('--distillation_temperature', type=float, help='temperature to use for distillation', default=1)
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--log_file', type=str, help='path for logging model extraction metrics')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--save_every_epoch', action='store_true', help='saves a model at every intermediate epoch')

    parser.add_argument('--exp_name', type=str, help='name for exepriment', default='train adversary')
    parser.add_argument('--defense', type=str, help='name for defense', default='None')
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=32)
    parser.add_argument('--arch', type=str, help='Use VGG architecture for the model', default='resnet')
    parser.add_argument('--skip_training', action='store_true', help='Skip training')


    args = parser.parse_args()
    print(args)

    main(args)
