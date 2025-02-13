import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset,random_split
import random
import os

class EEGLoader(Dataset):
    def __init__(self, path, pick_session=0, settup='train', spiltratio=[0.7, 0.1, 0.2]):
        if spiltratio[0] + spiltratio[1] + spiltratio[2] != 1:
            raise ValueError('sum of spiltratio should be 1')

        # session内实验
        self.data_path = path + '/session{}'.format(pick_session)
        datanum = self.count_npz_files(self.data_path)
        # if settup == 'train':
        #     self.data_len = int(datanum * spiltratio[0])
        #     self.bias = 0
        # elif settup == 'validate':
        #     self.data_len = int(datanum * (spiltratio[0] + spiltratio[1])) - int(datanum * spiltratio[0])
        #     self.bias = int(datanum * spiltratio[0])
        # elif settup == 'test':
        #     self.data_len = datanum - int(datanum * (spiltratio[0] + spiltratio[1]))
        #     self.bias = int(datanum * (spiltratio[0] + spiltratio[1]))
        # else:
        #     raise ValueError('settup应为train/validate/test')
        if settup == 'train':
            start_idx = 0
            end_idx = int(datanum * spiltratio[0])
        elif settup == 'validate':
            start_idx = int(datanum * spiltratio[0])
            end_idx = int(datanum * (spiltratio[0] + spiltratio[1]))
        elif settup == 'test':
            start_idx = int(datanum * (spiltratio[0] + spiltratio[1]))
            end_idx = datanum
        else:
            raise ValueError('setup should be train/validate/test')
        # 预加载所有数据
        self.data = []
        self.label = []
        for i in range(start_idx, end_idx):
            data = np.load(self.data_path + '/{}.npz'.format(i))
            spike, label = data['x'], data['y']
            self.data.append(spike)
            self.label.append(label)

        # 更新数据集长度
        self.data_len = len(self.data)
    def __getitem__(self, index):
        # data = np.load(self.data_path + '/{}.npz'.format(self.bias + index))
        # spike, label = data['x'], data['y']
        return self.data[index], self.label[index] #spike, label

    def __len__(self):
        return self.data_len

    def count_npz_files(self, path):
        return len([f for f in os.listdir(path) if f.endswith('.npz')])

    def getsample(self, num):
        if num > self.data_len:
            raise ValueError('Requested sample size exceeds dataset length')
        indices = np.random.choice(self.data_len, num, replace=False)
        samples = [self.__getitem__(i)[0] for i in indices]

        # samples = torch.cat([fm for fm in samples], dim=0)
        return samples

    def filter_channels(self, channels_to_keep=
                        [0, 1, 4, 5, 7, 11, 12, 13, 16, 17, 21, 24, 25, 28, 34, 35, 39, 43, 51, 59, 60, 62, 63, 66, 72, 74, 75, 77, 78, 79]
):
        """过滤数据，只保留指定的通道"""
        for i in range(self.data_len):
            self.data[i] = self.data[i][channels_to_keep,:]

def get_samples_from_concat_dataset(dataset, num_samples):
    if num_samples > len(dataset):
        raise ValueError('Requested sample size exceeds dataset length')
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    # print(type(dataset[0]))
    samples = [dataset[i][0] for i in indices]
    # print(type(samples))
    return samples


class EEGLoader2(Dataset):
    def __init__(self, path, pick_session=(0,), all_session=range(14), settup='train', spiltratio=[0.7, 0.1, 0.2]):
        if spiltratio[0] + spiltratio[1] + spiltratio[2] != 1:
            raise ValueError('sum of spiltratio should be 1')

        train_id = list(set(all_session) - set(pick_session))
        self.data = []
        self.labels = []
        self.pick_id = []
        self.data_path = path

        if settup == 'train':
            for idx in train_id:
                session_path = path + '/session{}'.format(idx)
                datanum = self.count_npz_files(session_path)
                for i in range(int(datanum * (spiltratio[0] + spiltratio[1]))):
                    data = np.load(session_path + '/{}.npz'.format(i))
                    self.data.append(data['x'])
                    self.labels.append(data['y'])
                self.pick_id.append(idx)

        elif settup == 'validate':
            for idx in train_id:
                session_path = path + '/session{}'.format(idx)
                datanum = self.count_npz_files(session_path)
                for i in range(int(datanum * (spiltratio[0] + spiltratio[1])), datanum):
                    data = np.load(session_path + '/{}.npz'.format(i))
                    self.data.append(data['x'])
                    self.labels.append(data['y'])
                self.pick_id.append(idx)

        elif settup == 'test':
            for idx in set(pick_session):
                session_path = path + '/session{}'.format(idx)
                datanum = self.count_npz_files(session_path)
                for i in range(datanum):
                    data = np.load(session_path + '/{}.npz'.format(i))
                    self.data.append(data['x'])
                    self.labels.append(data['y'])
                self.pick_id.append(idx)
        else:
            raise ValueError('settup应为train/validate/test')

        self.data_len = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len

    def count_npz_files(self, path):
        return len([f for f in os.listdir(path) if f.endswith('.npz')])


# class EEGLoader2(Dataset):
#     def __init__(self, path, pick_session=(0,), all_session=range(14), settup='train', spiltratio=[0.7, 0.1, 0.2]):
#         if spiltratio[0] + spiltratio[1] + spiltratio[2] != 1:
#             raise ValueError('sum of spiltratio should be 1')
#         train_id = list(set(all_session) - set(pick_session))
#         self.data_len = 0
#         self.bias = []
#         self.data_lens = []
#         self.pick_id = []
#         self.data_path = path
#         if settup == 'train':
#             for idx in train_id:
#                 session_path = path + '/session{}'.format(idx)
#                 datanum = self.count_npz_files(session_path)
#                 self.data_len += int(datanum * (spiltratio[0] + spiltratio[1]))
#                 self.data_lens.append(self.data_len)
#                 self.bias.append(0)
#                 self.pick_id.append(idx)
#         elif settup == 'validate':
#             for idx in train_id:
#                 session_path = path + '/session{}'.format(idx)
#                 datanum = self.count_npz_files(session_path)
#                 self.data_len += datanum - int(datanum * (spiltratio[0] + spiltratio[1]))
#                 self.data_lens.append(self.data_len)
#                 self.bias.append(int(datanum * (spiltratio[0] + spiltratio[1])))
#                 self.pick_id.append(idx)
#         elif settup == 'test':
#             for idx in set(pick_session):
#                 session_path = path + '/session{}'.format(idx)
#                 datanum = self.count_npz_files(session_path)
#                 self.data_len += datanum
#                 self.data_lens.append(self.data_len)
#                 self.bias.append(0)
#                 self.pick_id.append(idx)
#         else:
#             raise ValueError('settup应为train/validate/test')
#
#
#
#     def __getitem__(self, index):
#         for i, index_border in enumerate(self.data_lens):
#             if index < index_border:
#                 index = index if i == 0 else (index - self.data_lens[i - 1])
#                 data = np.load(self.data_path + '/session{}'.format(self.pick_id[i]) + '/{}.npz'.format(self.bias[i] + index))
#                 spike, label = data['x'], data['y']
#                 break
#         return spike, label
#
#     def __len__(self):
#         return self.data_len
#
#     def count_npz_files(self, path):
#         return len([f for f in os.listdir(path) if f.endswith('.npz')])

class EEGLoader3(Dataset):
    def __init__(self, path, pick_session=(), setup='train', spiltratio=[0.7, 0.1, 0.2]):
        if spiltratio[0] + spiltratio[1] + spiltratio[2] != 1:
            raise ValueError('sum of spiltratio should be 1')

        train_id = list(set(pick_session))
        self.data = []
        self.labels = []
        self.data_path = path

        if setup == 'train':
            for idx in train_id:
                session_path = os.path.join(path, f'session{idx}')
                datanum = self.count_npz_files(session_path)
                # limit = int(datanum * spiltratio[0])
                self.load_session_data(session_path, 0, datanum)

        # elif setup == 'validate':
        #     for idx in train_id:
        #         session_path = os.path.join(path, f'session{idx}')
        #         datanum = self.count_npz_files(session_path)
        #         start = int(datanum * spiltratio[0])
        #         limit = start + int(datanum * spiltratio[1])
        #         self.load_session_data(session_path, start, limit)

        elif setup == 'test':
            for idx in set(pick_session):
                session_path = os.path.join(path, f'session{idx}')
                datanum = self.count_npz_files(session_path)
                self.load_session_data(session_path, 0, datanum)

        else:
            raise ValueError('setup应为train/validate/test')
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    def load_session_data(self, session_path, start, end):
        for i in range(start, end):
            data = np.load(os.path.join(session_path, f'{i}.npz'))
            spike, label = data['x'], data['y']
            self.data.append(spike)
            self.labels.append(label)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def count_npz_files(self, path):
        return len([f for f in os.listdir(path) if f.endswith('.npz')])


