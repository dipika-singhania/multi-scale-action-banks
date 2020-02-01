""" Implements a dataset object which allows to read representations from LMDB datasets in a multi-modal fashion
The dataset can sample frames for both the anticipation and early recognition tasks."""
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils import data
import pandas as pd
from torch.utils.data.dataloader import default_collate

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
            else:
                pool_ndarr = np.array(pool_list)
                max_pool = np.max(pool_ndarr, 0)
                list_data.append(max_pool.squeeze())
   
    if len(list_data) == 0:
        #print("ERROR: None of the partitions were non-zero")
        return None 
    valid_index = -1
    for _ in missing_kkl: # Reversing and adding next frames to previous frames to fill in indexes with many empty at start
       list_data.append(list_data[-1])
    list_data  =  np.stack(list_data)
    # print(list_data.shape)
    return list_data

def read_representations(recent_frames, past_frames, env, tran=None, max_pool=False, hsplit=2):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided"""
    current = []
    past    = []

    recent_features_arr = []  
    past_features_arr   = []
    # print("Shape of recent frame ", len(recent_frames)) #, "Shape of 0th frame", recent_frames[0].shape)
    # print("Shape of past frame ", len(past_frames)) # , "Shape of 0th frame", past_frames[0].shape)
    
    for e in env:
        recent_env_arr = []
        past_env_arr   = []
        for i in range(len(recent_frames)):
            recent_features = get_max_pooled_features(e, recent_frames[i])
            if recent_features is not None:
                recent_env_arr.append(recent_features)

        if len(recent_env_arr) == 0:
            print("Skipping: None of current frames is 1")
            return None, None
        elif len(recent_env_arr) < len(recent_frames):
            random_arr = np.random.randint(0, len(recent_env_arr), len(recent_frames) - len(recent_env_arr))
            for nums in random_arr:
                recent_env_arr.append(recent_env_arr[nums])       
       
        #skipped_frames = []
        for i in range(len(past_frames)):
            past_features = get_max_pooled_features(e, past_frames[i])
            if past_features is not None:
                past_env_arr.append(past_features)
            else:
                skipped_frames.append(i)

        if len(past_env_arr) == 0:
            for ele in skipped_frames:
                print("Skiiped, recent 0", recent_frames[0][0], " to ", recent_frames[0][-1])
                print("Skiiped, ", past_frames[ele][0], " to ", past_frames[ele][-1])
            print("Skipping: None of past frames is 1")
            return None, None
        elif len(past_env_arr) < len(past_frames):
            random_arr = np.random.randint(0, len(past_env_arr), len(past_frames) - len(past_env_arr))
            for nums in random_arr:
                past_env_arr.append(past_env_arr[nums])       

        recent_features_arr.append(recent_env_arr)
        past_features_arr.append(past_env_arr)

    recent_features_arr = np.concatenate(recent_features_arr, axis=-1)
    past_features_arr   = np.concatenate(past_features_arr, axis=-1)
    # print("Recent features size ", recent_features_arr.shape)
    # print("Past features size ", past_features_arr.shape)

    for i in range(len(recent_features_arr)):
        if max_pool == True:
            # print("Max pooling features")
            temp_var_split = np.hsplit(recent_features_arr[i], hsplit)
            temp_var_max = np.max(temp_var_split, axis=0)
            # print("present features array", (temp_var_max.shape))
            current.append(temp_var_max)
        else:
            # print("present features array", (recent_features_arr[i].shape))
            current.append(recent_features_arr[i])

    for i in range(len(past_features_arr)):
        if max_pool == True:
            temp_var_split = np.hsplit(past_features_arr[i], hsplit)
            temp_var_max   = np.max(temp_var_split, axis=0)
            current.append(temp_var_max)
            # print("past features array", (temp_var_max.shape))
        else:
            # print("past features array", (past_features_arr[i].shape))
            past.append(past_features_arr[i])

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

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb, path_to_csv, label_type = 'action',
                time_step = 0.25, sequence_length = 14, fps = 30,
                img_tmpl = "frame_{:010d}.jpg",
                transform = None,
                challenge = False,
                task = 'anticipation',
                action_samples = None, args=None, multi_samples=0):
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
                task: Defines which task is being used
                action_samples: number of frames to be evenly sampled from each action
                multi-sample: number of samples to pick up from
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
        self.task = task
        self.action_samples = action_samples
        self.fps = fps
        self.transform = transform
        self.label_type = label_type
        self.sequence_length = sequence_length
        self.img_tmpl = img_tmpl
        self.action_samples = action_samples
        self.curr_sec_list  = args.curr_sec_list
        self.dim_curr      = args.dim_curr
        self.dim_past_list = args.dim_past_list
        self.rel_sec  = args.rel_sec
        self.past_sec = args.past_sec
        self.f_max    = args.f_max
        self.hsplit   = args.hsplit
        self.take_future = args.take_future

        # initialize some lists
        self.ids = [] # action ids
        self.discarded_ids = [] # list of ids discarded (e.g., if there were no enough frames before the beginning of the action
        self.past_frames = [] # names of frames sampled before each action
        self.action_frames = [] # names of frames sampled from each action
        self.labels = [] # labels of each action
        self.recent_frames = [] # recent past to taken seperately
        self.multi_samples = multi_samples

        # populate them
        self.__populate_lists()

        # if a list to datasets has been provided, load all of them
        if isinstance(self.path_to_lmdb, list):
            self.env = []
            for l in self.path_to_lmdb:
                if isinstance(l, list):
                    env_sub = []
                    for m in l:
                        print("Opening lmdb environment ", m)
                        env_sub = lmdb.open(m, readonly=True, lock=False)
                    self.env.append(env_sub)
                else:
                    print("Opening lmdb environment ", l)
                    env_sub = lmdb.open(l, readonly=True, lock=False)
                    self.env.append(env_sub)
        else:
            # otherwise, just load the single LMDB dataset
            print("Opening lmdb environment ", self.path_to_lmdb)
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
            if len(list_frames) == 0:
                print("list_frames = 0. Start and end encountered =", cur_start, cur_end + 1)
            list_data.append(self.__get_frames(list_frames, video))
        if len(list_data) == 0:
            print("ERROR: no data frames found")
        return list_data

    def __get_action_bank_recognition_features(self, start_point, end_point, video):
        """Samples frames before the beginning of the action "point" """
        # compute the time stamp corresponding to the beginning of the action
        start_end_curr = []
        for ele in self.curr_sec_list:
            start_before = max(start_point - (self.fps * ele), 0)
            end_after    = end_point + (self.fps * ele)
            if start_before >= end_after:
                temp         = start_before
                start_before = end_after
                end_after    = temp + 1
            start_end_curr.append((start_before, end_after))

        # Current Features
        sel_frames_cur_list = [np.linspace(st,  end, self.dim_curr + 1, dtype=int) for (st, end) in start_end_curr]
        instances_current = []
        for sel_frames_cur in sel_frames_cur_list:
            if (len(sel_frames_cur) == 0):
                print("sel_frames_cur linespace is 0", start_end_curr)
            instances_current.append(self.__get_frames_from_indices(video, sel_frames_cur))

        # Past Fea1tures
        sel_frames_past_list = []
        if self.take_future == False:
            start_past = max(end_point - (self.fps * self.past_sec), 0)
            sel_frames_past_list = [np.linspace(start_past, end_point,    dim_p + 1,  dtype=int) for dim_p in self.dim_past_list]
        else:
            past_sec_by_2 = self.past_sec // 2
            start_past = max(end_point - (self.fps * (self.past_sec - past_sec_by_2)), 0)
            end_past   = end_point + (self.fps * past_sec_by_2)
            sel_frames_past_list = [np.linspace(start_past, end_past, dim_p + 1, dtype=int) for dim_p in self.dim_past_list]

        instances_past = []
        for sel_frames_past in sel_frames_past_list:
            if (len(sel_frames_past) == 0):
                print("sel_frames_past linespace is 0. Start past = ", start_past, "End past =", end_point)
            instances_current.append(self.__get_frames_from_indices(video, sel_frames_cur))
            instances_past.append(self.__get_frames_from_indices(video, sel_frames_past))

        return instances_current, instances_past

    def __get_action_bank_features(self, point, video, time_leap):
        """Samples frames before the beginning of the action "point" """
        if point <= (time_leap + self.curr_sec_list[0]) * self.fps:  # Need atleast 1 sec video before the actual action
            return None, None
        end_past    = point - (time_leap * self.fps)
        end_current = end_past

        # Current Features
        instances_current   = []
        start_current_list  = [max(end_past - ele * self.rel_sec * self.fps,  0) for ele in self.curr_sec_list]
        # for ele in self.curr_sec_list:
        #     if end_current < 0 or ele < 0 or ele >= end_current:
        #         import pdb
        #         pdb.set_trace()
        sel_frames_cur_list = [np.linspace(ele, end_current, self.dim_curr + 1, dtype=int) for ele in start_current_list]
        for sel_frames_cur in sel_frames_cur_list:
            instances_current.append(self.__get_frames_from_indices(video, sel_frames_cur))

        # Past Features
        instances_past       = []
        start_past           = max(end_past - (self.past_sec * self.rel_sec * self.fps), 0)
        sel_frames_past_list = [np.linspace(start_past, end_past, dim_p + 1, dtype=int) for dim_p in self.dim_past_list]
        # if start_past < 0 or end_past < 0 or start_past >= end_past:
        #     import pdb
        #     pdb.set_trace()
        for sel_frames_past in sel_frames_past_list:
            instances_past.append(self.__get_frames_from_indices(video, sel_frames_past))

        return instances_current, instances_past

    def __add_to_list(self, a, past_f, present_f, id_name):
        if past_f is not None and present_f is not None: # if the smaller frame is at least 1, the sequence is valid
            self.recent_frames.append(present_f)
            self.past_frames.append(past_f)
            self.ids.append(id_name)
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
            print("Invalid sequence ", a.name, "Video =", a.video)

    def __populate_lists(self):
        count = 0
        """ Samples a sequence for each action and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total = len(self.annotations)):
            count += 1
            # if count > 100:
            #     break
            # sample frames before the beginning of the action
            if self.task == 'anticipation':
                if isinstance(self.time_step, list):
                    for time_alpha in self.time_step:
                        id_temp = str(a.name) + "_" + str(time_alpha)
                        current_f, past_f = self.__get_action_bank_features(a.start, a.video, float(time_alpha))
                        self.__add_to_list(a, past_f, current_f, id_temp)
                else:
                    current_f, past_f = self.__get_action_bank_features(a.start, a.video, float(self.time_step))
                    self.__add_to_list(a, past_f, current_f, a.name)
                    
            elif self.task == 'recognition':
                current_f, past_f = self.__get_action_bank_recognition_features(a.start, a.end, a.video)
                # print("Count = %d", count)
                # check if there were enough frames before the beginning of the action
                if self.multi_samples == 0:
                    self.__add_to_list(a, past_f, current_f, a.name)
                else:
                    if current_f is not None and past_f is not None:
                        for i in range(self.multi_samples):
                            id_name = str(i) + "_" + str(a.name)
                            self.__add_to_list(a, past_f, current_f, id_name)
                    else:
                        self.__add_to_list(a, past_f, current_f, a.name)
                        

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

        # read representations for past frames
        if self.multi_samples > 0:
            env_num = int(self.ids[index][:self.ids[index].find('_')])
            out['recent_features'], out['past_features'] = read_data(recent_frames, past_frames, self.env[env_num], self.transform,\
                                                                     self.f_max, self.hsplit)
        else:
            out['recent_features'], out['past_features'] = read_data(recent_frames, past_frames, self.env, self.transform,\
                                                                     self.f_max, self.hsplit)

        if out['recent_features'] is None and out['past_features'] is None:
            print("Id discarded = ", self.ids[index])
            return None

        # get the label of the current sequence
        label = self.labels[index]
        out['label'] = label

        return out

