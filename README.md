# Efficient Defense Against Model Stealing Attacks on Convolutional Neural Networks

### Abstract
Model stealing attacks have become a serious concern for deep learning models, where an attacker can steal a trained model by querying its black-box API. This can lead to intellectual property theft and other security and privacy risks. The current state-of-the-art defenses against model stealing attacks suggest adding perturbations to the prediction probabilities. However, they suffer from heavy computations and make impracticable assumptions about the adversary. They require the training of auxiliary models. This can be time-consuming and resource-intensive which hinders the deployment of these defenses in real-world applications. In this paper, we propose a simple yet effective and efficient defense alternative. We introduce a heuristic approach to perturb the output probabilities. The proposed defense can be easily integrated into models without additional training. We show that our defense is effective in defending against three state-of-the-art stealing attacks. We evaluate our approach on large and quantized (i.e., compressed) Convolutional Neural Networks (CNNs) trained on several vision datasets. Our technique outperforms the state-of-the-art defenses with a ×37 faster inference latency without requiring any additional model and with a low impact on the model's performance. We validate that our defense is also effective for quantized CNNs targeting edge devices.



## How To Use
### Environment setting:

Environment can be set:
- with `virtualenv` using:

```bash
virtualenv --no-download defending-extraction
source defending-extraction/bin/activate   
pip install -r requirements.txt   
pip install -e .
cd df
```
to deactivate the environment use `deactivate`

- or, with `conda` using:
```bash
conda env create -f environment.yml
conda activate defending-extraction
pip install -e .

```

to deactivate the environment use `conda deactivate`


```bash
cd defending-extraction/
conda activate defending-extraction
cd df/
```

### Path and general configuration variables
```shell
OUTPUTS="/Your-Path-to-Workspace/defending-extraction/df/outputs" 
datasets="gtsrb svhn cifar10 cifar100 cub200" 
defenses="None DCP ReverseSigmoid AdaptiveMisinformation Random GRAD MAD"
adv_arch="resnet34"
suffix="_${adv_arch}"
```

### Running multiple experiments
```shell
chmod +x run_all.sh
./run_all.sh
```


### Training teachers
```shell
misinformation=0
exp_name="${dataset}"
save_path="${OUTPUTS}/trained_models/${dataset}_teacher.pt"

python train_teacher.py \
  --dataset $dataset \
  --save_path $save_path \
  --num_gpus 1 \
  --misinformation $misinformation  \
  --exp_name $exp_name  \
  --epochs $epochs_teach  > "logs/${exp_name}.txt"
```


### Quantization
```shell
quantization="ptq" # or "qat"
config="server"

python quantize.py \
  --dataset $dataset \
  --original_model_path $original_model_path \
  --compressed_model_path $compressed_model_path \
  --num_gpus 1 \
  --quantization $quantization  \
  --exp_name $exp_name  \
  --epochs $epochs_teach  \
  --config $config \
  --replace > "logs/${exp_name}.txt"
```

### Training surrogates
```shell
ese=1
exp_name="${transfer_data}_to_${eval_data}_${ese}epochs_surrogate"
save_path="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_surrogate_${ese}epochs.pt"

python train_surrogate.py \
    --transfer_data $transfer_data \
    --eval_data $eval_data \
    --save_path $save_path \
    --num_gpus 1 \
    --early_stopping_epoch $ese \
    --exp_name $exp_name \
    --epochs $epochs_surr > "logs/${exp_name}.txt"
```



### Generating perturbed posteriors 
```shell
exp_name="transferset_${transfer_data}_to_${eval_data}_${defense}"
load_path="${OUTPUTS}/trained_models/${dataset}_teacher.pt"
save_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}.pkl"

python get_queries.py \
          --transfer_data $transfer_data \
          --eval_data $eval_data \
          --defense $defense \
          --epsilons "${epsilons}" \
          --save_path $save_path \
          --exp_name $exp_name \
          --load_path $load_path  > "logs/${exp_name}.txt"

# ===== generating perturbed posteriors on the validation data to evaluate the defender accuracy ==== #
exp_name="transferset_${transfer_data}_to_${eval_data}_${defense}_val"
load_path="${OUTPUTS}/trained_models/${dataset}_teacher.pt"
save_path_val="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}_val.pkl"
  
python get_queries.py \
      --transfer_data $transfer_data \
      --eval_data $eval_data \
      --defense $defense \
      --epsilons "${epsilons}" \
      --save_path $save_path_val \
      --eval_perturbations \
      --exp_name $exp_name \
      --load_path $load_path  > "logs/${exp_name}.txt"
```



### Training adversaries on perturbed posteriors - KnockoffNets
```shell
exp_name="${eval_data}_adv_from_${transfer_data}_${defense}_eps${epsilon}${suffix}"
save_path="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}_eps${epsilon}${suffix}.pt"
log_file="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}${suffix}.json"

python train_adversary.py \
        --transfer_data $transfer_data \
        --eval_data $eval_data \
        --load_path $load_path \
        --epsilon $epsilon \
        --save_path $save_path \
        --num_gpus 1 \
        --oracle $oracle  \
        --exp_name $exp_name \
        --defense $defense \
        --epochs $epochs_adv \
        --log_file $log_file \
        --clean_posteriors_path $clean_posteriors_path \
        --arch $adv_arch  > "logs/${exp_name}.txt"
```


### Training adversaries - DFME and MAZE
```shell
adv_arch=resnet34_8x 
suffix="_${adv_arch}"

## DFME
maze=0
python train_adversary_dfme.py \
            --eval_data $eval_data \
            --epsilon $epsilon \
            --model_filename $model_filename \
            --exp_name $exp_name \
            --defense $defense \
            --epochs 200 \
            --log_file $log_file \
            --arch $adv_arch  \
            --MAZE $maze > "logs/${exp_name}.txt"
## MAZE
maze=1
python train_adversary_dfme.py \
            --eval_data $eval_data \
            --epsilon $epsilon \
            --model_filename $model_filename \
            --exp_name $exp_name \
            --defense $defense \
            --epochs 200 \
            --log_file $log_file \
            --arch $adv_arch  \
            --MAZE $maze > "logs/${exp_name}.txt"
           
```


## Acknowledgment

A significant part of our code base that implements the previous state-of-the-art work is from : https://github.com/mmazeika/model-stealing-defenses and https://github.com/cake-lab/datafree-model-extraction






