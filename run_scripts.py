import argparse
import subprocess
import os
from subprocess import Popen

def ffmpeg_extraction(seconds, i):
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(i)
    ffmpeg_command = ['python3', 'main.py', 'train',
                      '/mnt/data/epic_kitchen/data/', '/mnt/data/epic_kitchen/past_sec_models/', '--modality','flow', '--video_feat_dim', '1024', '--past_sec', str(seconds),'--dim_past1', '5', '--dim_past2', '3', '--dim_past3', '2', '--dim_curr', '2', '--curr_seconds1', '2', '--curr_seconds2', '1.5', '--curr_seconds3', '1', '--curr_seconds4', '0.5', '--lr', '1e-4', '--latent_dim', '512', '--linear_dim', '512', '--dropout_rate', '0.3', '--dropout_linear', '0.3', '--batch_size', '10', '--epochs', '25'] 
    output_file = open('past_flow_' + str(seconds) + '.txt', 'w+')
    Popen(ffmpeg_command, env=my_env, stdout=output_file, stderr=output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds_start', default=2, help='start of video')
    parser.add_argument('--seconds_end', default=10, help='end of video')
    parser.add_argument('--time_incr', default=2, help='increment we need to test')
    args = parser.parse_args()

    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)

    count = 0
    for sec in range(args.seconds_start, args.seconds_end, args.time_incr):
        ffmpeg_extraction(sec, count)
        count += 1
        if count > 4:
            break
