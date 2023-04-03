#!/bin/sh

OUTPUTS="/Your-Path-to-Workspace/defending-extraction/df/outputs"

epochs_teach=50
epochs_mis=5
epochs_adv=50
epochs_surr=10

quantization=""  # ptq or qat
config="server"

compressed_model_path="${OUTPUTS}/compressed_models/quantized_resnet_${dataset}_${quantization}.pt"
original_model_path="${OUTPUTS}/compressed_models/resnet_${dataset}.pt"

suffix="_resnet34"
#suffix=""
adv_arch="resnet34"

datasets="gtsrb svhn cifar10 cifar100 cub200"
defenses="None DCP ReverseSigmoid AdaptiveMisinformation Random GRAD MAD"

epsilons_default="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5"
#epsilons_DCP="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5" # same as default
epsilons_None="0.0"
epsilons_ReverseSigmoid="0.0 0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28"
epsilons_AdaptiveMisinformation="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"  # tau parameters (named epsilon for convenience)
epsilon_Temp="0.5 0.6 0.7 0.8 0.9 1.1 1.2 1.5 3 5 7 10 15 20"  # Temp parameters (named epsilon for convenience)
epsilons_PreNorm="0.0"
epsilons_PostNorm="0.0"
epsilons_MAD="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
epsilons_Random="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
epsilons_GRAD="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"


for dataset in $datasets # mnist #cifar10 cifar100  # cub200 # mnist
do
  #   # ==================== DO NOT COMMENT THIS ==================== #
  eval_data=$dataset
  if [ "$eval_data" = "cub200" ]; then
              transfer_data=pascal
    fi
    if [ "$eval_data" = "cifar10" ]; then
      transfer_data=cifar100
    fi
    if [ "$eval_data" = "cifar100" ]; then
      transfer_data=cifar10
    fi
    if [ "$eval_data" = "svhn" ]; then
      transfer_data=cifar10
    fi
    if [ "$eval_data" = "gtsrb" ]; then
      transfer_data=cifar10
    fi
    if [ "$eval_data" = "mnist" ]; then
      transfer_data=fashionmnist
    fi
    if [ "$eval_data" = "fashionmnist" ]; then
      transfer_data=mnist
    fi

   # ==================== training teachers ==================== #
    misinformation=0
    exp_name="${dataset}"
    save_path="${OUTPUTS}/trained_models/${dataset}_teacher.pt"

    echo  "logs/${exp_name}.txt"

    python train_teacher.py \
      --dataset $dataset \
      --save_path $save_path \
      --num_gpus 1 \
      --misinformation $misinformation  \
      --exp_name $exp_name  \
      --epochs $epochs_teach  > "logs/${exp_name}.txt"

##
  misinformation=1

    exp_name="${dataset}_misinformation"
    save_path="${OUTPUTS}/trained_models/${dataset}_misinformation.pt"
    python train_teacher.py \
      --dataset $dataset \
      --save_path $save_path \
      --num_gpus 1 \
      --misinformation $misinformation \
      --exp_name $exp_name \
      --epochs $epochs_mis



#      # #   Quantization
## Uncomment for quantization purposes
#    quantization="qat" # ptq or qat
#    config="server"
#
#    compressed_model_path="${OUTPUTS}/compressed_models/quantized_resnet_${dataset}_${quantization}.pt"
#    original_model_path="${OUTPUTS}/compressed_models/resnet_${dataset}.pt"
#
#    exp_name="${dataset}_quantization_${quantization}"
#
#    echo "logs/${exp_name}.txt"
#
#    python quantize.py \
#      --dataset $dataset \
#      --original_model_path $original_model_path \
#      --compressed_model_path $compressed_model_path \
#      --num_gpus 1 \
#      --quantization $quantization  \
#      --exp_name $exp_name  \
#      --epochs $epochs_teach  \
#      --config $config \
#      --replace > "logs/${exp_name}.txt"
#
#    quantization="ptq" # ptq or qat
#    config="server"
#
#    compressed_model_path="${OUTPUTS}/compressed_models/quantized_resnet_${dataset}_${quantization}.pt"
#    original_model_path="${OUTPUTS}/compressed_models/resnet_${dataset}.pt"
#    exp_name="${dataset}_quantization_${quantization}"
#
#    echo  "logs/${exp_name}.txt"
#
#    python quantize.py \
#      --dataset $dataset \
#      --original_model_path $original_model_path \
#      --compressed_model_path $compressed_model_path \
#      --num_gpus 1 \
#      --quantization $quantization  \
#      --exp_name $exp_name  \
#      --epochs $epochs_teach  \
#      --config $config > "logs/${exp_name}.txt"
#
#    quantization="qat"
#    quantization=""

  echo "Finished training teachers"
#
#  # ==================== training surrogates ==================== #
    eval_data=$dataset

    for idx in 0 1
    do
      ese=$((10*idx))
       exp_name="${transfer_data}_to_${eval_data}_${ese}epochs_surrogate"
      save_path="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_surrogate_${ese}epochs.pt"
      echo "logs/${exp_name}.txt"

      python train_surrogate.py \
      --transfer_data $transfer_data \
      --eval_data $eval_data \
      --save_path $save_path \
      --num_gpus 1 \
      --early_stopping_epoch $ese \
      --exp_name $exp_name \
      --epochs $epochs_surr > "logs/${exp_name}.txt"
    done


  echo "Finished training surrogates"

  # ==================== generating perturbed posteriors ==================== #

    eval_data=$dataset

    for defense in $defenses
    do
      epsilons=$epsilons_default
      if [ "$defense" = "None" ]; then
        epsilons=$epsilons_None
      fi

      if [ "$defense" = "ReverseSigmoid" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      if [ "$defense" = "AdaptiveMisinformation" ]; then
        epsilons=$epsilons_AdaptiveMisinformation # tau parameters (named epsilon for convenience)
      fi
      if [ "$defense" = "MAD" ]; then
        epsilons=$epsilons_MAD
      fi
      if [ "$defense" = "GRAD" ]; then
        epsilons=$epsilons_GRAD
      fi


      if [ "$quantization" = "" ]; then
        exp_name="transferset_${transfer_data}_to_${eval_data}_${defense}"
        load_path="${OUTPUTS}/trained_models/${dataset}_teacher.pt"
        save_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}.pkl"
      else
        exp_name="transferset_${transfer_data}_to_${eval_data}_${defense}_${quantization}"
        load_path=$compressed_model_path
        save_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}_${quantization}.pkl"
      fi

      python get_queries.py \
          --transfer_data $transfer_data \
          --eval_data $eval_data \
          --defense $defense \
          --epsilons "${epsilons}" \
          --save_path $save_path \
          --exp_name $exp_name \
          --load_path $load_path  > "logs/${exp_name}.txt"

#      # ===== generating perturbed posteriors on the validation data to evaluate the defender accuracy ==== #

    if [ "$quantization" = "" ]; then
        exp_name="transferset_${transfer_data}_to_${eval_data}_${defense}_val"
        load_path="${OUTPUTS}/trained_models/${dataset}_teacher.pt"
        save_path_val="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}_val.pkl"
      else
        exp_name="transferset_${transfer_data}_to_${eval_data}_${defense}_${quantization}_val"
        load_path=$compressed_model_path
        save_path_val="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}_${quantization}_val.pkl"
      fi

      echo  "logs/${exp_name}.txt"

      python get_queries.py \
      --transfer_data $transfer_data \
      --eval_data $eval_data \
      --defense $defense \
      --epsilons "${epsilons}" \
      --save_path $save_path_val \
      --eval_perturbations \
      --exp_name $exp_name \
      --load_path $load_path  > "logs/${exp_name}.txt"


    done

  echo "Probably finished generating posteriors  for dataset: $dataset"

  # ==================== training adversaries on perturbed posteriors ==================== #

  oracle="None"  # Unused in the experiments in the paper; can be ignored (generates perturbations in each batch using the adversary's true parameters instead of a surrogate)

    eval_data=$dataset

    for defense in $defenses
    # AdaptiveMisinformation #PreNorm PostNorm # Temp  Random ReverseSigmoid Random  None # GRAD NEW NEW-MIN-IP MIN-IP_10 MAD AdaptiveMisinformation ReverseSigmoid Random  None # Random_vgg
    do
      if [ "$quantization" = "" ]; then
        load_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}.pkl"
        clean_posteriors_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_None.pkl"
      else
        load_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}_${quantization}.pkl"
        clean_posteriors_path="${OUTPUTS}/generated_perturbations/${transfer_data}_to_${eval_data}_None_${quantization}.pkl"
      fi

      epsilons=$epsilons_default
      if [ "$defense" = "None" ]; then
        epsilons=$epsilons_None
      fi
      if [ "$defense" = "ReverseSigmoid" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      if [ "$defense" = "ReverseSigmoid_0.1" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      if [ "$defense" = "Softplus" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      if [ "$defense" = "Gelu" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      if [ "$defense" = "Tanh" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      # AM_RS
      if [ "$defense" = "AdaptiveMisinformation" ]; then
        # epsilons="0.0"  # tau parameters (named epsilon for convenience)
        epsilons=$epsilons_AdaptiveMisinformation # tau parameters (named epsilon for convenience)
      fi
      if [ "$defense" = "Temp" ]; then
        epsilons=$epsilon_Temp  # Temp parameters (named epsilon for convenience)
      fi
      if [ "$defense" = "PreNorm" ]; then
        epsilons=$epsilons_PreNorm
      fi
      if [ "$defense" = "PostNorm" ]; then
        epsilons=$epsilons_PostNorm
      fi
      if [ "$defense" = "MAD" ]; then
        epsilons=$epsilons_MAD
      fi
      if [ "$defense" = "GRAD" ]; then
        epsilons=$epsilons_GRAD
      fi
      if [ "$defense" = "Random" ]; then
        epsilons=$epsilons_Random
      fi

      for epsilon in $epsilons
      do
        if [ "$quantization" = "" ]; then
          exp_name="${eval_data}_adv_from_${transfer_data}_${defense}_eps${epsilon}${suffix}"
          save_path="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}_eps${epsilon}${suffix}.pt"
          log_file="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}${suffix}.json"
        else
          exp_name="${eval_data}_adv_from_${transfer_data}_${defense}_eps${epsilon}_${quantization}${suffix}"
          save_path="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}_eps${epsilon}_${quantization}${suffix}.pt"
          log_file="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}_${quantization}${suffix}.json"
        fi


        echo  "logs/${exp_name}.txt"
#
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
      done

    done

  echo "Done runnning jobs for dataset: $dataset"

    # ==================== training adversaries with DFME technique ==================== #

  oracle="None"  # Unused in the experiments in the paper; can be ignored (generates perturbations in each batch using the adversary's true parameters instead of a surrogate)

    eval_data=$dataset

    for defense in $defenses
    do
      epsilons=$epsilons_default
      if [ "$defense" = "None" ]; then
        epsilons=$epsilons_None
      fi
      if [ "$defense" = "ReverseSigmoid" ]; then
        epsilons=$epsilons_ReverseSigmoid
      fi
      if [ "$defense" = "AdaptiveMisinformation" ]; then
        epsilons=$epsilons_AdaptiveMisinformation # tau parameters (named epsilon for convenience)
      fi
      if [ "$defense" = "MAD" ]; then
        epsilons=$epsilons_MAD
      fi
      if [ "$defense" = "GRAD" ]; then
        epsilons=$epsilons_GRAD
      fi
      if [ "$defense" = "Random" ]; then
        epsilons=$epsilons_Random
      fi


#      adv_arch=resnet34_8x #resnet18
      adv_arch=wideresnet #resnet18
#      adv_arch=resnet18 #resnet18
      suffix="_${adv_arch}"


      for epsilon in $epsilons
      do
        if [ "$quantization" = "" ]; then

          exp_name="${eval_data}_adv_DFME_${defense}_eps${epsilon}${suffix}"
          model_filename="student_DFME_${eval_data}_${defense}_eps${epsilon}${suffix}.pt"
          log_file="${OUTPUTS}/trained_models/student_DFME_${eval_data}_${defense}${suffix}.json"
        else
          exp_name="${eval_data}_adv_from_${transfer_data}_${defense}_eps${epsilon}_${quantization}${suffix}"
  #        exp_name="${eval_data}_adv_from_${transfer_data}_${defense}"
          save_path="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}_eps${epsilon}_${quantization}${suffix}.pt"
          log_file="${OUTPUTS}/trained_models/${transfer_data}_to_${eval_data}_${defense}_${quantization}${suffix}.json"
        fi

        echo  "logs/${exp_name}.txt"

        python train_adversary_dfme.py \
            --eval_data $eval_data \
            --epsilon $epsilon \
            --model_filename $model_filename \
            --exp_name $exp_name \
            --defense $defense \
            --epochs 200 \
            --log_file $log_file \
            --arch $adv_arch  \
            --MAZE 0 > "logs/${exp_name}.txt"



      done

    done

  echo "Done runnning DFME jobs for dataset: $dataset"
done
echo "Done runnning all jobs!"
