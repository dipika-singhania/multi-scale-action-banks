python3 main.py train /mnt/data/epic_kitchen/data/ /mnt/data/epic_kitchen/models/ --modality rgb --video_feat_dim 1024 --past_sec 10 -p 5 -p 3 -p 2 --dim_curr 2 -c 0 -c 1 -c 2 --lr 1e-4  --latent_dim 512 --linear_dim 512 --dropout_rate 0.3 --dropout_linear 0.3 --batch_size 10  --epochs 45 --task recognition --num_workers 4 --add_verb_loss --add_noun_loss
