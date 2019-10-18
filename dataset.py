""" Implements a dataset object which allows to read representations from LMDB datasets in a multi-modal fashion
The dataset can sample frames for both the anticipation and early recognition tasks."""
import sys
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils import data
import pandas as pd

def get_max_pooled_features(env, frame_names):
    list_data = []
    missing_kkl = []
    for kkl in range(len(frame_names)):
        with env.begin() as e:
            pool_list = []
            for name in frame_names[kkl]:
                dd = e.get(name.strip().encode('utf-8'))
                if dd is None:
                    # print("Missing frame ", name)
                    continue
                # convert to numpy array
                data = np.frombuffer(dd, 'float32')
                pool_list.append(data)
            if (len(pool_list) == 0):
                # print("Missing frames in ", kkl, "indices")
                missing_kkl.append(kkl)
                list_data.append(np.zeros(1024))
            else:
                pool_ndarr = np.array(pool_list)
                max_pool = np.max(pool_ndarr, 0)
                list_data.append(max_pool.squeeze())

    for index in missing_kkl[::-1]: # Reversing and adding next frames to previous frames to fill in indexes with many empty at start
        list_data[index] = list_data[index + 1]
    list_data  =  np.stack(list_data)
    # print(list_data.shape)
    return list_data

def read_representations(recent_frames, past_frames, env, tran=None, max_pool=False, hsplit=2):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided"""
    current = []
    past    = []

    recent_features = []
    recent_features1 = []
    recent_features2 = []
    past_features = []
    past_features1 = []
    past_features2 = []
    for e in env:
        recent_features.append(get_max_pooled_features(e, recent_frames[0]))
        recent_features1.append(get_max_pooled_features(e, recent_frames[1]))
        recent_features2.append(get_max_pooled_features(e, recent_frames[2]))

        past_features.append(get_max_pooled_features(e, past_frames[0]))
        past_features1.append(get_max_pooled_features(e, past_frames[1]))
        past_features2.append(get_max_pooled_features(e, past_frames[2]))

    recent_features = np.concatenate(recent_features, axis=-1)
    recent_features1 = np.concatenate(recent_features1, axis=-1)
    recent_features2 = np.concatenate(recent_features2, axis=-1)
    past_features = np.concatenate(past_features, axis=-1)
    past_features1 = np.concatenate(past_features1, axis=-1)
    past_features2 = np.concatenate(past_features2, axis=-1)

    if max_pool == True:
        # print("Max pooling features")
        recent_features = np.hsplit(recent_features, hsplit)
        recent_features1 = np.hsplit(recent_features1, hsplit)
        recent_features2 = np.hsplit(recent_features2, hsplit)
        past_features = np.hsplit(past_features, hsplit)
        past_features1 = np.hsplit(past_features1, hsplit)
        past_features2 = np.hsplit(past_features2, hsplit)
       
        recent_features = np.max(recent_features, axis=0)
        recent_features1 = np.max(recent_features1, axis=0)
        recent_features2 = np.max(recent_features2, axis=0)
        past_features = np.max(past_features, axis=0)
        past_features1 = np.max(past_features1, axis=0)
        past_features2 = np.max(past_features2, axis=0)
        


    # print("past features array", (past_features.shape))
    # print("present features array", (recent_features.shape))

    current.append(recent_features)
    past.append(past_features)

    current.append(recent_features)
    past.append(past_features1)

    current.append(recent_features)
    past.append(past_features2)

    current.append(recent_features1)
    past.append(past_features)

    current.append(recent_features1)
    past.append(past_features1)

    current.append(recent_features1)
    past.append(past_features2)

    current.append(recent_features2)
    past.append(past_features)

    current.append(recent_features2)
    past.append(past_features1)

    current.append(recent_features2)
    past.append(past_features2)

    return current, past


def read_data(recent_frames, past_frames, env, tran=None, max_pool=False, hsplit=2):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)"""
    # if env is a list
    if isinstance(env, list):
        # read the representations from all environments
        return read_representations(recent_frames, past_frames, env, tran, max_pool, hsplit)
    else:
        # otherwise, just read the representations
        env = [env]
        return read_representations(recent_frames, past_frames, env, tran, max_pool, hsplit)

class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb, path_to_csv, label_type = 'action',
                time_step = 0.25, sequence_length = 14, fps = 30,
                img_tmpl = "frame_{:010d}.jpg",
                transform = None,
                challenge = False,
                past_features = True,
                action_samples = None, args=None):
        """
            Inputs:
                path_to_lmdb: path to the folder containing the LMDB dataset
                path_to_csv: path to training/validation csv
                label_type: which label to return (verb, noun, or action)
                time_step: in seconds
                sequence_length: in time steps
                fps: framerate
                img_tmpl: image template to load the features
                tranform: transformation to apply to each sample
                challenge: allows to load csvs containing only time-stamp for the challenge
                past_features: if past features should be returned
                action_samples: number of frames to be evenly sampled from each action
        """

        # read the csv file
        if challenge:
            self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end'])
        else:
            self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end','verb','noun','action'])

        # print(self.annotations.head())    
        # print(self.annotations.describe())    
        self.challenge = challenge
        self.path_to_lmdb = path_to_lmdb
        self.time_step = time_step
        self.past_features = past_features
        self.action_samples = action_samples
        self.fps = fps
        self.transform = transform
        self.label_type = label_type
        self.sequence_length = sequence_length
        self.img_tmpl = img_tmpl
        self.action_samples = action_samples
        self.current_seconds  = args.current_seconds
        self.current_seconds2 = args.current_seconds2
        self.current_seconds3 = args.current_seconds3
        self.in_dim_curr      = args.in_dim_curr
        self.in_dim_past1     = args.in_dim_past1
        self.in_dim_past2     = args.in_dim_past2
        self.in_dim_past3     = args.in_dim_past3
        self.rel_sec  = args.rel_sec
        self.past_sec = args.past_sec
        self.f_max    = args.f_max
        self.hsplit   = args.hsplit

        # initialize some lists
        self.ids = [] # action ids
        self.discarded_ids = [] # list of ids discarded (e.g., if there were no enough frames before the beginning of the action
        self.past_frames = [] # names of frames sampled before each action
        self.action_frames = [] # names of frames sampled from each action
        self.labels = [] # labels of each action
        self.recent_frames = [] # recent past to taken seperately
        
        # populate them
        self.__populate_lists()

        # if a list to datasets has been provided, load all of them
        if isinstance(self.path_to_lmdb, list):
            self.env = [lmdb.open(l, readonly=True, lock=False) for l in self.path_to_lmdb]
        else:
            # otherwise, just load the single LMDB dataset
            self.env = lmdb.open(self.path_to_lmdb, readonly=True, lock=False)

    def __get_frames(self, frames, video):
        """ format file names using the image template """
        frames = np.array(list(map(lambda x: video + "_" + self.img_tmpl.format(x), frames)))
        return frames

    def __get_frames_from_indices(self, video, indices):
        list_data = []
        for kkl in range(len(indices)-1):
            cur_start = np.floor(indices[kkl]).astype('int')
            cur_end   = np.floor(indices[kkl+1]).astype('int')
            list_frames = list(range(cur_start, cur_end + 1))
            list_data.append(self.__get_frames(list_frames, video))
        return list_data

    def __get_action_bank_features(self, point, video):
        """Samples frames before the beginning of the action "point" """
        time_stamps = self.time_step #, self.time_step * (self.sequence_length + 1), self.time_step)[::-1]
        
        # compute the time stamp corresponding to the beginning of the action
        end_time_stamp = point / self.fps 


        # subtract time stamps to the timestamp of the last frame
        end_time_stamp = end_time_stamp - time_stamps


        if end_time_stamp < 2:
            return None, None

        end_past = np.floor(end_time_stamp * self.fps).astype(int)
        end_current = end_past

        instances_current = [] 
        instances_past = [] 
        # start_past =  
        start_past = max(end_past - (self.past_sec * self.rel_sec * self.fps), 0)
        # Current Features

        start_current       = max(end_past - self.current_seconds * self.rel_sec * self.fps,  0)
        start_current2      = max(end_past - self.current_seconds2 * self.rel_sec * self.fps,  0)
        start_current3      = max(end_past - self.current_seconds3 * self.rel_sec * self.fps,  0)
        sel_frames_current  = np.linspace(start_current,   end_current, self.in_dim_curr + 1,  dtype=int) 
        sel_frames_current2 = np.linspace(start_current2,  end_current, self.in_dim_curr + 1,  dtype=int) 
        sel_frames_current3 = np.linspace(start_current3,  end_current, self.in_dim_curr + 1,  dtype=int) 

        # Past Features
        sel_frames_past     = np.linspace(start_past,     end_past,    self.in_dim_past1 + 1,  dtype=int) 
        sel_frames_past2    = np.linspace(start_past,     end_past,    self.in_dim_past2 + 1,  dtype=int) 
        sel_frames_past3    = np.linspace(start_past,     end_past,    self.in_dim_past3 + 1,  dtype=int) 
        
        # start_current + sel_frames_past{,2,3}
        instances_current.append(self.__get_frames_from_indices(video, sel_frames_current))
        instances_current.append(self.__get_frames_from_indices(video, sel_frames_current2))
        instances_current.append(self.__get_frames_from_indices(video, sel_frames_current3))

        instances_past.append(self.__get_frames_from_indices(video, sel_frames_past))
        instances_past.append(self.__get_frames_from_indices(video, sel_frames_past2))
        instances_past.append(self.__get_frames_from_indices(video, sel_frames_past3))

        return instances_current, instances_past
    
    def __populate_lists(self):
        count = 0
        """ Samples a sequence for each action and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total = len(self.annotations)):
            count += 1
            """if count >10:
                break"""
            # sample frames before the beginning of the action
            current_f, past_f = self.__get_action_bank_features(a.start, a.video)
            
            # check if there were enough frames before the beginning of the action
            if past_f is not None and current_f is not None: # if the smaller frame is at least 1, the sequence is valid
                self.past_frames.append(past_f)
                self.recent_frames.append(current_f)
                self.ids.append(a.name)
                # handle whether a list of labels is required (e.g., [verb, noun]), rather than a single action
                if isinstance(self.label_type, list):
                    if self.challenge: # if sampling for the challenge, there are no labels, just add -1
                        self.labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    if self.challenge:
                        self.labels.append(-1)
                    else:
                        self.labels.append(a[self.label_type])
            else:
                #if the sequence is invalid, do nothing, but add the id to the discarded_ids list
                self.discarded_ids.append(a.name)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """ sample a given sequence """
        # get past frames
        past_frames = self.past_frames[index]
        recent_frames = self.recent_frames[index]

        # return a dictionary containing the id of the current sequence
        # this is useful to produce the jsons for the challenge
        out = {'id' : self.ids[index]}
        # print("Data asked for index ", index)

        if self.past_features:
            # read representations for past frames
            out['recent_features'], out['past_features'] = read_data(recent_frames, past_frames, self.env, self.transform, self.f_max, self.hsplit)

        # get the label of the current sequence
        label = self.labels[index]
        out['label'] = label

        return out

