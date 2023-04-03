# modified from :
# https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/
# https://leimao.github.io/blog/PyTorch-Static-Quantization/

import argparse
import copy
import random

import torch.optim as optim
import torchvision

from df_utils import *
from model.resnet_q import resnet18 as resnet18_q


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model_q(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


############################## DATASET AND TRAINING CODE ##################################
def train_model(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, print_every=100,suffix=""):
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
        val_metrics = evaluate(model, test_loader, suffix=suffix)  # {'acc':acc, 'loss':loss} TODO here put device
        # val_metrics = {'val_acc': acc, 'val_loss': loss}

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

            acc = 100.0 * np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == by.cpu().numpy()) / float(
                by.cpu().numpy().shape[0])

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
    val_metrics = evaluate(model, test_loader, suffix=suffix)  # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
    print("- Eval metrics : " + metrics_string)
    log_metrics_val(val_metrics)
    # print('Final:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss, acc))
    model.eval()
    return model


################################################################

def cross_entropy_loss(logits, gt_target):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


def train_model_q(model, train_loader, test_loader, device):
    # The training configurations were not carefully selected.
    learning_rate = 1e-2
    num_epochs = 5

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model_q(model=model, test_loader=test_loader, device=device,
                                                  criterion=criterion)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch,
                                                                                                             train_loss,
                                                                                                             train_accuracy,
                                                                                                             eval_loss,
                                                                                                             eval_accuracy))

    return model


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


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model


def create_model_q(num_classes=10):
    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = resnet18_q(num_classes=num_classes, pretrained=False)

    return model


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


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


def main(args):
    wandb.init(
        project=cfg.WB_PROJECT,
        entity=cfg.WB_ENTITY,
        group=args.dataset,  # or args.eval_data
        name=args.exp_name,
        tags=['quantized', args.dataset, 'compressed'],
        config=args, settings=wandb.Settings(start_method="fork"))

    random_seed = 0
    # num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    config = {"server": "fbgemm", "mobile": "qnnpack"}

    model_dir = os.path.join(cfg.MODEL_DIR, 'compressed_models')  # "saved_models"
    model_filename = f"resnet_{args.dataset}.pt"  # "resnet18_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)

    quantized_model_filename = f"quantized_resnet_{args.dataset}_{args.quantization}.pt"
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    set_random_seeds(random_seed=random_seed)



    # train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=32, eval_batch_size=32)

    train_data, _ = load_data(args.dataset, train=True)
    test_data, num_classes = load_data(args.dataset, train=False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_gpus * 32, num_workers=cfg.NUM_WORKERS,
                                               shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.num_gpus * 32, num_workers=cfg.NUM_WORKERS,
                                              shuffle=False, pin_memory=False)

    # Create an untrained model.
    model = create_model(dataset_name=args.dataset, num_classes=num_classes, quantized=True)
    # Train model.
    # model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device)
    num_epochs = args.epochs  # 50
    lr = 0.01 if args.dataset == 'cub200' else 0.001  # 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    loss = cross_entropy_loss

    if not os.path.exists(model_filepath) or args.replace:

        train_model(model, train_loader, test_loader, optimizer, scheduler, loss, num_epochs=num_epochs, suffix="_pre_q")
        # Save model.
        save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    else:
        # Load a pretrained model.
        print("Model exists, skipping the training")


    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)

    # Move the model to CPU since static quantization does not support CUDA currently.
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)

    if args.quantization == "ptq":
        # Post training quantization
        model.eval()
        # The model has to be switched to evaluation mode before any layer fusion.
        # Otherwise the quantization will not work correctly.
        fused_model.eval()

        # Fuse the model in place rather manually.
        # if args.eval_data not in ['mnist', 'fashionmnist']
        fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                                                    inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

        # Print FP32 model.
        print(model)
        # Print fused model.
        print(fused_model)

        # Model and fused model should be equivalent.
        #
        # assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06,
        #                          num_tests=100,
        #                          input_size=(1, 3, 32, 32)), "Fused model is not equivalent to the original model!"

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        quantized_model = QuantizedModel(model_fp32=fused_model)
        # Using un-fused model will fail.
        # Because there is no quantized layer implementation for a single batch normalization layer.
        # quantized_model = QuantizedResNet18(model_fp32=model)
        # Select quantization schemes from
        # https://pytorch.org/docs/stable/quantization-support.html
        quantization_config = torch.quantization.get_default_qconfig(config[args.config])
        # Custom quantization configurations
        # quantization_config = torch.quantization.default_qconfig
        # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

        quantized_model.qconfig = quantization_config

        # Print quantization configurations
        print(quantized_model.qconfig)

        torch.quantization.prepare(quantized_model, inplace=True)

        # Use training data for calibration.
        calibrate_model(model=quantized_model, loader=train_loader, device=cpu_device)

        quantized_model = torch.quantization.convert(quantized_model, inplace=True)


    elif args.quantization == "qat":
        # Quantization aware training
        model.train()
        # The model has to be switched to training mode before any layer fusion.
        # Otherwise the quantization aware training will not work correctly.
        fused_model.train()

        # Fuse the model in place rather manually.
        fused_model = torch.quantization.fuse_modules(fused_model,
                                                      [["conv1", "bn1", "relu"]],
                                                      inplace=True)
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(
                        basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                        inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block,
                                                            [["0", "1"]],
                                                            inplace=True)

        # Print FP32 model.
        print(model)
        # Print fused model.
        print(fused_model)

        # Model and fused model should be equivalent.
        model.eval()
        fused_model.eval()
        assert model_equivalence(
            model_1=model,
            model_2=fused_model,
            device=cpu_device,
            rtol=1e-03,
            atol=1e-06,
            num_tests=100,
            input_size=(
                1, 3, 32,
                32)), "Fused model is not equivalent to the original model!"

        # Prepare the model for quantization aware training. This inserts observers in
        # the model that will observe activation tensors during calibration.
        quantized_model = QuantizedModel(model_fp32=fused_model)
        # Using un-fused model will fail.
        # Because there is no quantized layer implementation for a single batch normalization layer.
        # quantized_model = QuantizedResNet18(model_fp32=model)
        # Select quantization schemes from
        # https://pytorch.org/docs/stable/quantization-support.html
        quantization_config = torch.quantization.get_default_qconfig(config[args.config])
        # Custom quantization configurations
        # quantization_config = torch.quantization.default_qconfig
        # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

        quantized_model.qconfig = quantization_config

        # Print quantization configurations
        print(quantized_model.qconfig)

        # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
        torch.quantization.prepare_qat(quantized_model, inplace=True)

        # # Use training data for calibration.
        print("Training QAT Model...")
        quantized_model.train()
        # train_model_q(model=quantized_model,
        #             train_loader=train_loader,
        #             test_loader=test_loader,
        #             device=cuda_device,
        #             learning_rate=1e-3,
        #             num_epochs=10)

        num_epochs = args.epochs  # 50
        lr = 0.01 if args.dataset == 'cub200' else 0.001  # 0.05
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
        loss = cross_entropy_loss

        train_model(quantized_model, train_loader, test_loader, optimizer, scheduler, loss, num_epochs=num_epochs, suffix="_qat")

        quantized_model.to(cpu_device)

        # Using high-level static quantization wrapper
        # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
        # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    else:
        print('Unknown quantization method')
        raise NotImplementedError

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize(model=quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

    quantized_model.eval()

    # Print quantized model.
    # print(quantized_model)

    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)

    model_size = get_file_size(model_filepath)
    quantized_jit_model_size = get_file_size(quantized_model_filepath)

    # fp32_val_metrics = evaluate(model, test_loader, device=cpu_device)
    # fp32_eval_accuracy = fp32_val_metrics['val_acc']

    _, fp32_eval_accuracy = evaluate_model_q(model, test_loader, device=cpu_device)
    # {'val_acc': acc, 'val_loss': loss}

    # int8_val_metrics = evaluate(quantized_jit_model,test_loader, device=cpu_device)
    # # {'val_acc': acc, 'val_loss': loss}
    # int8_eval_accuracy = int8_val_metrics['val_acc']

    _, int8_eval_accuracy = evaluate_model_q(quantized_jit_model, test_loader, device=cpu_device)



    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1, 3, 32, 32),
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device,
                                                           input_size=(1, 3, 32, 32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device,
                                                               input_size=(1, 3, 32, 32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1, 3, 32, 32),
                                                           num_samples=100)
    metrics = dict(
        fp32_eval_accuracy=fp32_eval_accuracy,
        int8_eval_accuracy=int8_eval_accuracy,
        fp32_model_size=model_size,
        int8_model_size=quantized_jit_model_size,
        fp32_cpu_inference_latency=fp32_cpu_inference_latency * 1000,
        int8_cpu_inference_latency=int8_cpu_inference_latency * 1000,
        int8_jit_cpu_inference_latency=int8_jit_cpu_inference_latency * 1000,
        fp32_gpu_inference_latency=fp32_gpu_inference_latency * 1000
    )
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))
    wandb.log(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize model')
    parser.add_argument('--dataset', type=str, help='dataset that we transfer knowledge with')
    parser.add_argument('--original_model_path', type=str, help='path for saving model')
    parser.add_argument('--compressed_model_path', type=str, help='path for saving compressed model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
    parser.add_argument('--exp_name', type=str, help='name for exepriment', default='quantization')
    parser.add_argument('--quantization', type=str, help='Quantization technique', default='ptq',
                        choices=['ptq', 'qat'])
    parser.add_argument('--config', type=str, help='Quantization config', default='server',
                        choices=['mobile', 'server'])
    parser.add_argument('--replace', action='store_true', help='replace original model')

    # Quantization Aware Training (QAT)
    # Post Training Static Quantization (PTQ static)
    # https://pytorch.org/docs/stable/quantization.html

    args = parser.parse_args()

    main(args)
