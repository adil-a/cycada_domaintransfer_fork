if [ "$1" == "all" ]; then
    how_many=100000
else
    how_many=50
fi

model=cycada_cylreal2cyl_noIdentity
epoch=100
python test.py --name ${model} \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --no_flip --batchSize 100 \
    --dataset_mode cylinderreal_cylinder --dataroot /ssd003/home/adilasif/adil_code/cycada/closest_corner_data \
    --which_direction AtoB \
    --phase train \
    --checkpoints_dir /h/adilasif/adil_code/cycada/cyclegan/checkpoints \
    --how_many ${how_many} \
    --which_epoch ${epoch} \
    --results_dir /h/adilasif/adil_code/cycada/cyclegan/results

