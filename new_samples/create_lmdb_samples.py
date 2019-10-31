import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm

modality = 'obj'
fps = 30
required_fps = 6
img_tmpl = 'frame_{:010d}.jpg'

write_env = lmdb.open('/mnt/data/epic_kitchen/sample2/' + modality + '/', map_size=1099511627776)
read_env = lmdb.open('/mnt/data/epic_kitchen/full_data_new/' + modality + '/')

csv_file = pd.read_csv(modality + '_videos_frames_info.csv', header=None, names=['video','start','end'], dtype={'start':np.int64, 'end':np.int64})

def __get_frames(frames, video):
    """ format file names using the image template """
    frames = np.array(list(map(lambda x: video.strip() + "_" + img_tmpl.format(x), frames)))
    return frames

for i in range(len(csv_file)):
    line = csv_file.loc[i]
    frames_read = []
    frames_to_be_read = []
    frames_seg = int(fps/required_fps)
    end = 0
    for i in tqdm(range(line.start, line.end, frames_seg), "Loading each frames", total=int(line.end/frames_seg)):
        end = min(i + frames_seg, line.end - 1)
        if i >= end:
            continue
        random_numbers = np.random.randint(i, end, 1)
        frame_name = line.video.strip() + "_" + img_tmpl.format(random_numbers[0])
        frames_to_be_read.append(frame_name)
    if end < line.end:
        random_numbers = np.random.randint(end, line.end + 1, 1)
        frame_name = line.video.strip() + "_" + img_tmpl.format(random_numbers[0])
        frames_to_be_read.append(frame_name)
    
            
    with read_env.begin() as e:
        for frame_name in frames_to_be_read:
            frame_name = frame_name.strip()
            frame_name = frame_name.encode('utf-8')
            dd = e.get(frame_name)
            if dd is not None:
                data = np.frombuffer(dd, 'float32')
                frames_read.append((frame_name, data))

    with write_env.begin(write=True) as txn:
        for frame_name, data in frames_read:
            txn.put(frame_name, data.tobytes())
        

        
