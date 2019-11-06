import argparse
import os
import time
import lmdb
from tqdm import tqdm
import librosa
import numpy as np
from PIL import Image, ImageOps
import torchvision
import torch
import pickle

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Binding Network")
parser.add_argument('--audio_path', type=str, default="/mnt/data/epic_kitchen/data/sound_wave_whole_data.pkl")
parser.add_argument('--resampling_rate', type=int, default=24000)
parser.add_argument('--required_fps', type=int, default=60)
arguments = parser.parse_args()

def log_specgram(args, audio, window_size=5,
                 step_size=5, eps=1e-6):
    nperseg = int(round(window_size * args.resampling_rate / 1e3))
    noverlap = int(round(step_size * args.resampling_rate / 1e3))

    spec = librosa.stft(audio, n_fft=512,
                        window='hann',
                        hop_length=noverlap,
                        win_length=nperseg,
                        pad_mode='constant')

    spec = librosa.istft(spec) #np.log(np.real(spec * np.conj(spec)) + eps)
    return spec

def extract_sound_feature(args, samples, centre_sec):

    left_sec    = max(centre_sec - 0.639, 0)
    right_sec   = centre_sec + 0.639

    duration    = samples.shape[0] / float(args.resampling_rate)

    left_sample = int(round(left_sec * args.resampling_rate))
    right_sample = int(round(right_sec * args.resampling_rate))

    if left_sec < 0:
        samples = samples[:int(round(args.resampling_rate * 1.279))]

    elif right_sec > duration:
        samples = samples[-int(round(args.resampling_rate * 1.279)):]
    else:
        samples = samples[left_sample:right_sample]

    return log_specgram(args, samples)


import pandas as pd
video_csv = pd.read_csv('video_annotations.csv')
write_env = lmdb.open('/mnt/data/epic_kitchen/full_data_new/audio/', map_size=1099511627776)
img_tmpl = 'frame_{:010d}.jpg'

audio_path = pickle.load(open(arguments.audio_path, 'rb'))
for i, row in video_csv.iterrows(): #,"video feature extraction", total=len(video_csv)):

    samples = audio_path[row.video.strip()]
    spec    = librosa.stft(samples, n_fft=511)
    spec    = librosa.istft(spec) #np.log(np.real(spec * np.conj(spec)) + eps)

    #samples, sr = librosa.core.load(audio_path_full_name, sr=None, mono=True)
    interval    = 735
    start       = 0
    end         = 735
    frames_list = list()

    while (end <= spec.shape[0]):
        spec_arr = np.array(spec[start: end])
        frames_list.append(spec_arr)
        start = start + interval
        end   = end + interval


    print ("Number of frames for ,", row.video, "is ", len(frames_list))

    with write_env.begin(write=True) as txn:
        count = 0
        for data in frames_list:
            count += 1
            frame_name = row.video.strip() + "_" + img_tmpl.format(count)
            frame_name = frame_name.strip()
            frame_name = frame_name.encode('utf-8')
            txn.put(frame_name, data.tobytes())
 
 
    print("Completed lmdb entry for video ", row.video)

