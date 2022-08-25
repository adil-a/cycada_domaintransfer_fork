#!/bin/bash
#SBATCH --job-name="cycada"
#SBATCH -o cycada.out
#SBATCH -e cycada.err
#SBATCH  --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p t4v2
#SBATCH --qos=high

python cyclegan/train.py --name cycada_cylreal2cyl_noIdentity \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --no_flip --batchSize 24 \
    --dataset_mode cylinderreal_cylinder --dataroot /ssd003/home/adilasif/adil_code/cycada/closest_corner_data \
    --which_direction AtoB \
    --verbose \
    --checkpoints_dir /h/adilasif/adil_code/cycada/cyclegan/checkpoints

