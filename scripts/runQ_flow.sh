#block(name=flow,   memory= 7500,  gpus=1, hours=48)
python3 main.py train /mnt/data/epic_kitchen/data/ /mnt/data/epic_kitchen/models/  --modality flow --video_feat_dim 1024 --past_sec 5 -p 5 -p 3 -p 2 --dim_curr 2 -c 2 -c 1.5 -c 1 -c 0.5 --lr 1e-4  --latent_dim 512 --linear_dim 512 --dropout_rate 0.3 --dropout_linear 0.3 --batch_size 10  --epochs 45 
