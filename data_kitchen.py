# import d4rl
import gym
import numpy as np
import itertools
import hdfdict
import tqdm
# from spirl.components.data_loader import Dataset
from general_utils import AttrDict
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import TweetTokenizer
import pandas as pd
import d4rl
from torch.utils.data import Dataset, DataLoader


class KitchenData(Dataset):
    SPLIT = AttrDict(train=0.9, val=0.1, test=0.0)

    def __init__(self, data_dir=None, data_conf=None, phase="train", resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        # self.data_dir = data_dir
        self.spec = None#data_conf.dataset_spec
        self.subseq_len = 180#self.spec.subseq_len
        self.remove_goal = False#self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = None#data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle


        # env = gym.make(self.spec.env_name)
        # filename = "/code/kitchenEnv/Datasets/kitchen_microwave_kettle_light_slider-v0.hdf5"#/code/kitchenEnv/Datasets/kitchen_microwave_kettle_light_slider-v0.hdf5 
        # self.dataset = hdfdict.load(filename)
        # print(self.dataset)
        env = gym.make('kitchen-partial-v0')
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        # if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
        #     for seq in self.seqs:
        #         seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
        #         seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # # filter demonstration sequences
        # if 'filter_indices' in self.spec:
        #     print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
        #     self.seqs = list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
        #                        for x in self.seqs[self.spec.filter_indices[0] : self.spec.filter_indices[1]+1]))
        if self.shuffle:
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)
        print("self.n_seqs", self.n_seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        # print("seq.states.shape[0]", seq.states.shape[0])
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        # if self.remove_goal:
        #     output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return self.end-self.start#int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)



class KitchenDataInorder(Dataset):
    SPLIT = AttrDict(train=0.9, val=0.1, test=0.0)

    def __init__(self, data_dir=None, data_conf=None, phase="train", resolution=None, shuffle=False, dataset_size=-1):
        self.phase = phase
        # self.data_dir = data_dir
        self.spec = None#data_conf.dataset_spec
        self.subseq_len = 180#self.spec.subseq_len
        self.remove_goal = False#self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = None#data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle


        # env = gym.make(self.spec.env_name)
        env = gym.make('kitchen-partial-v0')
        # filename = "/code/kitchenEnv/Datasets/kitchen_microwave_kettle_light_slider-v0.hdf5"#/code/kitchenEnv/Datasets/kitchen_microwave_kettle_light_slider-v0.hdf5 
        # self.dataset = hdfdict.load(filename)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        # if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
        #     for seq in self.seqs:
        #         seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
        #         seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # # filter demonstration sequences
        # if 'filter_indices' in self.spec:
        #     print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
        #     self.seqs = list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
        #                        for x in self.seqs[self.spec.filter_indices[0] : self.spec.filter_indices[1]+1]))
        if self.shuffle:
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)
        print("self.n_seqs", self.n_seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self.seqs[index]
        # print("seq.states.shape[0]", seq.states.shape[0])
        start_idx = 0#np.random.randint(0, seq.states.shape[0] - self.subseq_len)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        # if self.remove_goal:
        #     output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return self.end-self.start#598#int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)

class KitchenDataIM(Dataset):
    SPLIT = AttrDict(train=0.9, val=0.1, test=0.0)

    def __init__(self, data_dir=None, data_conf=None, phase="train", resolution=None, shuffle=False, dataset_size=-1):
        self.phase = phase
        # self.data_dir = data_dir
        self.spec = None#data_conf.dataset_spec
        self.subseq_len = 180#self.spec.subseq_len
        self.remove_goal = False#self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = None#data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle


        # env = gym.make(self.spec.env_name)
        env = gym.make('kitchen-partial-v0')
        # filename = "/code/kitchenEnv/Datasets/kitchen_microwave_kettle_light_slider-v0.hdf5"#/code/kitchenEnv/Datasets/kitchen_microwave_kettle_light_slider-v0.hdf5 
        # self.dataset = hdfdict.load(filename)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        self.pairs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            # for i in range(10)
            for i in range(1,45,5):
                for j in range(180-i):
                    start = self.dataset['observations'][j]
                    end = self.dataset['observations'][j+i]
                    seqStates = np.zeros((180,60))
                    seqActions = np.zeros((179,9))
                    seqStates[0] = start
                    seqStates[179] = end
                    seqActions[0:i] = self.dataset['actions'][j:j+i] 
                    self.seqs.append(AttrDict(
                        states=seqStates,
                        actions=seqActions,
                    ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        # if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
        #     for seq in self.seqs:
        #         seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
        #         seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # # filter demonstration sequences
        # if 'filter_indices' in self.spec:
        #     print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
        #     self.seqs = list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
        #                        for x in self.seqs[self.spec.filter_indices[0] : self.spec.filter_indices[1]+1]))
        if self.shuffle:
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)
        print("self.n_seqs", self.n_seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self.seqs[index]
        # print("seq.states.shape[0]", seq.states.shape[0])
        # start_idx = 0#np.random.randint(0, seq.states.shape[0] - self.subseq_len)
        # output = AttrDict(
        #     states=seq.states[start_idx:start_idx+self.subseq_len],
        #     actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
        #     pad_mask=np.ones((self.subseq_len,)),
        # )
        # if self.remove_goal:
        #     output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return seq

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return self.end-self.start

# train_data = KitchenDataInorder()
# OBJS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
# OBS_ELEMENT_INDICES = {
#     'bottom burner': np.array([11, 12]),
#     'top burner': np.array([15, 16]),
#     'light switch': np.array([17, 18]),
#     'slide cabinet': np.array([19]),
#     'hinge cabinet': np.array([20, 21]),
#     'microwave': np.array([22]),
#     'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
#     }
# OBS_ELEMENT_GOALS = {
#     'bottom burner': np.array([-0.88, -0.01]),
#     'top burner': np.array([-0.92, -0.01]),
#     'light switch': np.array([-0.69, -0.05]),
#     'slide cabinet': np.array([0.37]),
#     'hinge cabinet': np.array([0., 1.45]),
#     'microwave': np.array([-0.75]),
#     'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
#     }
# BONUS_THRESH = 0.3
# train_data = KitchenDataInorder(phase="val")
# dataload = DataLoader(train_data, batch_size=80, num_workers=32)
# # seqs = train_data.seqs

# # ## determine achieved subgoals + respective time steps
# # n_seqs, n_objs = len(seqs), len(OBJS)
# # subtask_steps = np.Inf * np.ones((n_seqs, n_objs))
# # for s_idx, seq in tqdm.tqdm(enumerate(seqs)):
# #     for o_idx, obj in enumerate(OBJS):
# #         for t, state in enumerate(seq.states):
# #             obj_state, obj_goal = state[OBS_ELEMENT_INDICES[obj]], OBS_ELEMENT_GOALS[obj]
# #             dist = np.linalg.norm(obj_state - obj_goal)
# #             if dist < BONUS_THRESH and subtask_steps[s_idx, o_idx] == np.Inf:
# #                 subtask_steps[s_idx, o_idx] = t

# # ## print subtask orders
# # print("\n\n")
# # for s_idx, subtasks in enumerate(subtask_steps):
# #     min_task_idxs = np.argsort(subtasks)[:4]
# #     objs = [OBJS[i] for i in min_task_idxs]
# #     print("seq {}: {}".format(s_idx, objs))

# # print(dataload)
# cnt = 0
# for entry in dataload:
#     print(cnt)
#     cnt +=1 
#     print(entry['states'].shape)
    # print(entry['actions'].shape)
    # break
# # for ind, i in enumerate(train_data):
# #     print(i['states'].shape)
# #     break
# print("cnt", cnt)