import numpy as np
import torch.nn.functional
from torch.utils.data import Dataset
import random
from sklearn.mixture import GaussianMixture


def generate_gmm_mask(shape, n_components=3, random_state=0):
    """
    Generate a 2D mask based on Gaussian Mixture Model (GMM).

    Parameters:
        shape (tuple): The shape of the mask (C, T).
        n_components (int): Number of Gaussian components to use.
        random_state (int): Random seed for reproducibility.

    Returns:
        mask (ndarray): A binary mask of shape (C, T).
    """
    C, T = shape
    # Generate a grid of coordinates
    X, Y = np.meshgrid(np.linspace(0, 1, C), np.linspace(0, 1, T))
    XY = np.vstack([X.ravel(), Y.ravel()]).T

    # Fit GMM to the grid of coordinates
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(XY)

    # Predict which component each point belongs to
    labels = gmm.predict(XY)

    # Create the mask by thresholding
    mask = labels.reshape(C, T)
    # Convert the mask to a binary format
    binary_mask = (mask == mask.max()).astype(np.uint8)

    return binary_mask


def eventmix(spike_data1, spike_data2, label, label1, dataset, mask=1):
    C, T = spike_data1.shape

    mixed_spike = spike_data1 * mask + spike_data2 * (1 - mask)
    return mixed_spike

def augment_with_random_pulse_insertion(spike_data, original_dataset, insertion_prob=0.1, delay_steps=5):
    """
    在每个值为1的脉冲后，以指定概率插入一个新的脉冲。

    参数:
    - spike_data: numpy数组，形状为(C, T)，表示输入的脉冲数据
    - insertion_prob: float，插入新脉冲的概率
    - delay_steps: int，插入脉冲的延迟时间步

    返回:
    - augmented_data: numpy数组，形状为(C, T)，表示增强后的脉冲数据
    """
    # 复制原始数据，避免直接修改
    augmented_data = spike_data.copy()

    # 获取脉冲信号的形状
    C, T = spike_data.shape

    # 遍历每个通道和时间步
    for c in range(C):
        for t in range(T):
            if spike_data[c, t] == 1:  # 如果当前位置有脉冲
                # 随机决定是否在 t + delay_steps 处插入新的脉冲
                if t + delay_steps < T and np.random.rand() < insertion_prob:
                    augmented_data[c, t + delay_steps] = 1  # 插入新的脉冲

    return augmented_data


# 1. 时间偏移 (Time Shift)
def time_shift(spike_data, original_dataset, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(spike_data, shift, axis=-1)  # 沿时间维度进行偏移

# 2. 脉冲翻转 (Spike Flipping)
def spike_flipping(spike_data, original_dataset, flip_prob=0.1):
    flip_mask = np.random.rand(*spike_data.shape) < flip_prob
    flipped_data = np.where(flip_mask, 1 - spike_data, spike_data)
    return flipped_data

# 3. 局部扰动 (Local Perturbation)
def local_perturbation(spike_data, original_dataset, perturb_window=20, perturb_prob=0.2):
    perturbed_data = spike_data.copy()
    for i in range(spike_data.shape[0]):
        if np.random.rand() < perturb_prob:
            start = np.random.randint(0, spike_data.shape[-1] - perturb_window)
            end = start + perturb_window
            perturbed_data[i, start:end] = 1 - spike_data[i, start:end]
    return perturbed_data
def local_zero(spike_data, original_dataset, perturb_window=50, perturb_prob=0.2):
    perturbed_data = spike_data.copy()
    for i in range(spike_data.shape[0]):
        if np.random.rand() < perturb_prob:
            start = np.random.randint(0, spike_data.shape[-1] - perturb_window)
            end = start + perturb_window
            perturbed_data[i, start:end] = 0
    return perturbed_data
# 4. 数据混合 (Data Mixing)
def mixup(spike_data1, spike_data2, label, label2, dataset, mode='random',):
    # if mode == 'inclass':
    #     spike_data2, _ = select_inclass_sample(label, dataset)
    # else:
    #     spike_data2, _ = dataset[np.random.randint(0, len(dataset) - 1)]
    # # 从数据集中随机选择另一个样本
    # random_index = np.random.randint(0, len(dataset) - 1)
    # spike_data2, _ = dataset[random_index]

    # lam = np.random.beta(alpha, alpha)
    lam = np.random.uniform(0.6, 0.9)
    mixed_data = lam * spike_data1 + (1 - lam) * spike_data2
    # mixed_label = lam * label + (1 - lam) * label2
    return np.round(mixed_data).astype(int)


def chimeric_mixing(spike_data1, spike_data2, label, dataset, mode='random', mix_window=50):
    # if mode == 'inclass':
    #     spike_data2, _ = select_inclass_sample(label, dataset)
    # else:
    #     spike_data2, _ = dataset[np.random.randint(0, len(dataset) - 1)]
    # 从数据集中随机选择另一个样本
    # random_index = np.random.randint(0, len(dataset) - 1)
    # spike_data2, _ = dataset[random_index]

    start = np.random.randint(0, spike_data1.shape[-1] - mix_window)
    end = start + mix_window
    mixed_data = spike_data1.copy()
    mixed_data[:, start:end] = spike_data2[:, start:end]
    return mixed_data


def concatenate_spikes(spike_data, spike_data2, label, dataset, mode='random',):

    # if mode == 'inclass':
    #     spike_data2, _ = select_inclass_sample(label, dataset)
    # else:
    #     spike_data2, _ = dataset[np.random.randint(0, len(dataset) - 1)]
    # random_index = np.random.randint(0, len(dataset) - 1)
    # spike_data2, _ = dataset[random_index]

    # 计算每个脉冲数据的中间点
    mid_point = spike_data.shape[-1] // 2

    # 获取两个脉冲数据的前半段和后半段
    spike_data[..., mid_point:] = spike_data2[..., mid_point:]

    return spike_data
# 5. 噪声注入 (Noise Injection)
def noise_injection(spike_data, dataset, noise_prob=0.05):
    C,T = spike_data.shape
    noised_data = np.zeros((C,T))
    noise_mask = np.random.rand(C,T) < noise_prob
    noised_data = np.where(noise_mask, 1 - spike_data, spike_data)
    return noised_data
# def noise_injection(spike_data, noise_prob=0.05):
#     # 确保输入是一个numpy数组
#     spike_data = np.array(spike_data)
#     noised_data = spike_data.copy()
#     print(type(noise_prob))
#     for c in range(spike_data.shape[0]):
#         for t in range(spike_data.shape[1]):
#             if np.random.rand() < noise_prob:
#                 noised_data[c, t] = 1 - spike_data[c, t]
#
#     return noised_data
# 6. 时间反转 (Time Reversal)
def time_reversal(spike_data, original_dataset):
    return np.flip(spike_data.copy(), axis=-1)


# 7. 频率域变换 (Frequency Domain Transformations)
def frequency_domain_transform(spike_data, original_dataset, perturb_scale=0.1):
    freq_data = np.fft.rfft(spike_data, axis=-1)
    perturb = np.random.randn(*freq_data.shape) * perturb_scale
    freq_data += perturb
    time_data = np.fft.irfft(freq_data, n=spike_data.shape[-1], axis=-1)
    time_data = np.clip(time_data, a_min=0, a_max=1)
    return np.round(time_data).astype(int)
    # return time_data
def frequency_shift(spike_data, original_dataset, shift_scale=0.1):
    freq_data = np.fft.rfft(spike_data, axis=-1)
    shift_amount = int(freq_data.shape[-1] * shift_scale)
    freq_data = np.roll(freq_data, shift_amount, axis=-1)
    time_data = np.fft.irfft(freq_data, n=spike_data.shape[-1], axis=-1)
    time_data = np.clip(time_data, a_min=0, a_max=1)
    return np.round(time_data).astype(int)
def frequency_filter(spike_data, original_dataset, cutoff_ratio=0.1):
    freq_data = np.fft.rfft(spike_data, axis=-1)
    cutoff_index = int(freq_data.shape[-1] * cutoff_ratio)
    freq_data[cutoff_index:] = 0
    time_data = np.fft.irfft(freq_data, n=spike_data.shape[-1], axis=-1)
    time_data = np.clip(time_data, a_min=0, a_max=1)
    return np.round(time_data).astype(int)
def random_phase_perturbation(spike_data, original_dataset, perturb_scale=0.1):
    freq_data = np.fft.rfft(spike_data, axis=-1)
    magnitude = np.abs(freq_data)
    phase = np.angle(freq_data)
    phase += np.random.randn(*phase.shape) * perturb_scale
    perturbed_freq_data = magnitude * np.exp(1j * phase)
    time_data = np.fft.irfft(perturbed_freq_data, n=spike_data.shape[-1], axis=-1)
    time_data = np.clip(time_data, a_min=0, a_max=1)
    return np.round(time_data).astype(int)


def frequency_surrogate(spike, original_dataset):
    # 对 spike 数据进行傅里叶变换
    spike_fft = np.fft.fft(spike, axis=-1)

    # 随机化相位信息
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, spike.shape))
    spike_fft_randomized = np.abs(spike_fft) * random_phases

    # 对随机化相位后的频率数据进行逆傅里叶变换
    augmented_spike = np.fft.ifft(spike_fft_randomized, axis=-1)

    # 取实部并将其转换为脉冲信号（0或1）
    pulse_signal = np.real(augmented_spike)

    # 确保脉冲信号的值只为0或1
    pulse_signal = np.clip(pulse_signal, 0, 1)

    return np.round(pulse_signal).astype(int)

from scipy.signal import hilbert
def frequency_shift_hilbert(spike, original_dataset, shift_frequency=0.1):
    """
    使用希尔伯特变换对脉冲信号的频率进行平移。

    :param spike: 输入的脉冲信号数据。
    :param original_dataset: 原始数据集（未使用，但保留参数以兼容）。
    :param shift_frequency: 平移的频率范围，默认为±0.1。
    :return: 增强后的脉冲信号数据。
    """
    # 对每个通道的信号进行希尔伯特变换，得到解析信号
    analytic_signal = hilbert(spike, axis=-1)

    # 计算瞬时相位
    instantaneous_phase = np.angle(analytic_signal)

    # 生成随机频率平移量
    shift = np.random.uniform(-shift_frequency, shift_frequency, size=spike.shape)

    # 对瞬时相位进行平移
    shifted_phase = instantaneous_phase + shift

    # 构建新的解析信号
    new_analytic_signal = np.abs(analytic_signal) * np.exp(1j * shifted_phase)

    # 取实部，得到增强后的信号
    augmented_spike = np.real(new_analytic_signal)

    # 确保脉冲信号的值只为0或1
    pulse_signal = np.clip(augmented_spike, 0, 1)

    # return np.round(pulse_signal).astype(int)
    return pulse_signal
from scipy.fftpack import dct, idct
def frequency_recombination(spike, original_dataset, recombination_ratio=0.5):
    """
    使用离散余弦变换(DCT)进行频率重组。

    :param spike: 输入的脉冲信号数据。
    :param original_dataset: 原始数据集，用于选择不同样本的频率带。
    :param recombination_ratio: 选择的频率带比例，默认为0.5。
    :return: 增强后的脉冲信号数据。
    """
    # 将输入数据转换到频域
    freq_domain_spike = dct(spike, axis=-1, norm='ortho')

    # 从数据集中随机选择另一个样本
    other_spike, _ = original_dataset[np.random.randint(0, len(original_dataset))]

    # 将随机选择的样本转换到频域
    freq_domain_other_spike = dct(other_spike, axis=-1, norm='ortho')

    # 根据 recombination_ratio 将频域数据进行组合
    combined_freq_domain = (recombination_ratio * freq_domain_spike +
                            (1 - recombination_ratio) * freq_domain_other_spike)

    # 将组合后的频域数据转换回时域
    recombined_spike = idct(combined_freq_domain, axis=-1, norm='ortho')



    # 确保脉冲信号的值只为0或1
    pulse_signal = np.clip(recombined_spike, 0, 1)

    return np.round(pulse_signal).astype(int)
def random_shift_psd(spike, original_dataset, shift_range=0.1):
    """
    随机平移所有通道的功率谱密度（PSD）。

    :param spike: 输入的脉冲信号数据。
    :param original_dataset: 原始数据集（未使用，但保留参数以兼容）。
    :param shift_range: 平移范围，默认为±0.1。
    :return: 增强后的脉冲信号数据。
    """
    # 对 spike 数据进行傅里叶变换
    spike_fft = np.fft.fft(spike, axis=-1)

    # 计算功率谱密度（PSD）
    psd = np.abs(spike_fft) ** 2

    # 随机生成一个平移值，在指定范围内
    shift_value = np.random.uniform(-shift_range, shift_range, size=psd.shape)
    # 对每个通道的功率谱密度进行随机平移
    shifted_psd = psd + shift_value
    # 使用平移后的功率谱密度和原始相位构造新的频域数据
    new_spike_fft = np.sqrt(shifted_psd) * np.exp(1j * np.angle(spike_fft))
    # 对新的频域数据进行逆傅里叶变换得到增强后的脉冲信号
    augmented_spike = np.fft.ifft(new_spike_fft, axis=-1)
    # 取实部并将其转换为脉冲信号（0或1）
    pulse_signal = np.real(augmented_spike)
    # 确保脉冲信号的值只为0或1
    pulse_signal = np.clip(pulse_signal, 0, 1)

    return np.round(pulse_signal).astype(int)

import pywt
import numpy as np


def wavelet_augmentation(spike_data, original_dataset,wavelet='haar', mode='soft', alpha=0.1):
    """
    使用小波变换对脉冲数据进行增强 (适用于 C*T 数据)。

    参数:
        spike_data: 输入数据，形状为 (C, T)
        wavelet: 小波基名称
        mode: 变换模式，'soft' 或 'hard'
        alpha: 阈值

    返回:
        变换后的数据，形状为 (C, T)
    """
    augmented_data = np.zeros_like(spike_data)

    for i in range(spike_data.shape[0]):  # 对每个通道进行增强
        coeffs = pywt.wavedec(spike_data[i], wavelet)
        # 对高频系数施加阈值
        if mode == 'soft':
            coeffs[1:] = [pywt.threshold(c, alpha, mode) for c in coeffs[1:]]
        elif mode == 'hard':
            coeffs[1:] = [c * (np.abs(c) > alpha) for c in coeffs[1:]]

        # 进行逆变换
        augmented_channel = pywt.waverec(coeffs, wavelet)
        augmented_data[i] = np.clip(augmented_channel, a_min=0, a_max=1)  # 限制在0到1之间

    return np.round(augmented_data).astype(int)


def fourier_augmentation(spike_data, original_dataset, noise_scale=0.1):
    """
    使用离散傅里叶变换 (DFT) 对脉冲数据进行增强 (适用于 C*T 数据)。

    参数:
        spike_data: 输入数据，形状为 (C, T)
        noise_scale: 噪声的规模，用于在频域中添加噪声

    返回:
        变换后的数据，形状为 (C, T)
    """
    augmented_data = np.zeros_like(spike_data)

    for i in range(spike_data.shape[0]):  # 对每个通道进行增强
        # 1. 进行离散傅里叶变换
        freq_data = np.fft.fft(spike_data[i])

        # 2. 在频域中添加噪声
        noise = np.random.randn(*freq_data.shape) + 1j * np.random.randn(*freq_data.shape)
        noise *= noise_scale
        freq_data += noise

        # 3. 进行逆离散傅里叶变换
        time_data = np.fft.ifft(freq_data)

        # 4. 取实部并裁剪到0和1之间（由于输入是实数，结果的虚部理论上应该很小）
        augmented_channel = np.clip(np.real(time_data), a_min=0, a_max=1)
        augmented_data[i] = augmented_channel

    return np.round(augmented_data).astype(int)
def magnitude_scaling(spike_data, original_dataset, scaling_factor=1.1):
    freq_data = np.fft.rfft(spike_data, axis=-1)
    magnitude = np.abs(freq_data) * scaling_factor
    phase = np.angle(freq_data)
    scaled_freq_data = magnitude * np.exp(1j * phase)
    time_data = np.fft.irfft(scaled_freq_data, n=spike_data.shape[-1], axis=-1)
    time_data = np.clip(time_data, a_min=0, a_max=1)
    return np.round(time_data).astype(int)
def additive_noise_in_frequency_domain(spike_data, original_dataset, noise_scale=0.1):
    freq_data = np.fft.rfft(spike_data, axis=-1)
    noise = np.random.randn(*freq_data.shape) + 1j * np.random.randn(*freq_data.shape)
    noise *= noise_scale
    freq_data += noise
    time_data = np.fft.irfft(freq_data, n=spike_data.shape[-1], axis=-1)
    time_data = np.clip(time_data, a_min=0, a_max=1)
    return np.round(time_data).astype(int)





# 定义增强数据集类
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, augment_fn, augment_mode=None):
        self.original_dataset = original_dataset
        self.augment_fn = augment_fn
        self.augment_mode = augment_mode
        # 计算需要增强的样本数量，并随机选择相应的索引
        self.num_to_augment = int(len(original_dataset) * 1.0)
        self.indices_to_augment = random.sample(range(len(original_dataset)), self.num_to_augment)
        # 在初始化时对选中的样本进行增强
        self.augmented_data = []
        self.augmented_label = []
        for i, (spike, label) in enumerate(original_dataset):
            if i in self.indices_to_augment:
                spike0 = self.augment_fn(spike, original_dataset)
                self.augmented_data.append((spike0.copy()))
                self.augmented_label.append((label.copy()))
    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, index):
        # spike, label = self.original_dataset[index]

        # augmented_spike = self.augment_fn(spike, self.original_dataset)
        return self.augmented_data[index], self.augmented_label[index] #spike, label


class AugmentedDatasets(Dataset):
    def __init__(self, original_dataset, augment_fn=None, augment_num=0.5):
        self.original_dataset = original_dataset
        self.augment_fn = augment_fn
        self.num_to_augment = int(len(original_dataset) * augment_num)
        self.indices_to_augment = random.sample(range(len(original_dataset)), self.num_to_augment)
        self.augmented_data = []
        self.augmented_label = []
        mask = generate_gmm_mask((80, 100), n_components=3, random_state=0)
        mask_counter = 0
        for i, (spike, label) in enumerate(original_dataset):
            if i in self.indices_to_augment:
                if mask_counter % 100 == 0:
                    mask = generate_gmm_mask((80, 100), n_components=3, random_state=mask_counter)
                mask_counter += 1

                index = random.choice(range(len(original_dataset)))
                spike1, _ = self.original_dataset[index]
                if augment_fn == 'eventmix':
                    spike0 = self.augment_fn(spike,spike1,label,_,original_dataset, mask)
                else:
                    spike0 = self.augment_fn(spike, spike1, label, _, original_dataset)
                self.augmented_data.append((spike0.copy()))
                self.augmented_label.append((label.copy()))
    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, index):
        # return self.original_dataset[index][0],self.original_dataset[index][1]
        return self.augmented_data[index], self.augmented_label[index]

    def apply_augmentation(self):
        class_indices = {j: [i for i in range(len(self.original_dataset)) if self.original_dataset[i][1] == j] for j in range(3)}
        augmented_data = []
        for index in range(len(self.original_dataset)):
            spike, label = self.original_dataset[index]
            if self.augment_mode == 'inclass':
                # class_indices = [i for i, (_, lbl) in enumerate(self.original_dataset) if lbl == label]
                # class_indices = [i for i in range(len(self.original_dataset)) if self.original_dataset[i][1] == label]
                spike_data2, label2 = self.original_dataset[np.random.choice(class_indices[int(label)])]
            else:
                spike_data2, label2 = self.original_dataset[np.random.randint(0, len(self.original_dataset) - 1)]

            # label = torch.nn.functional.one_hot(torch.from_numpy(label),num_classes=3)
            # label2 = torch.nn.functional.one_hot(torch.from_numpy(label2), num_classes=3)

            augmented_spike, augmented_label = self.augment_fn(spike, spike_data2, label,label2, self.original_dataset, self.augment_mode)
            # print(augmented_spike.shape,augmented_label)
            augmented_data.append((augmented_spike.copy(), augmented_label))
        # updated_dataset = []
        #
        # for data, lbl in self.original_dataset:
        #     # One-hot encode the labels of the original dataset
        #     lbl_onehot = torch.nn.functional.one_hot(torch.tensor(lbl), num_classes=3)
        #     updated_dataset.append((data, lbl_onehot))
        #
        # # Combine original data with augmented data
        # updated_dataset.extend(augmented_data)
        # self.original_dataset = updated_dataset
        # Combine original data with augmented data
        self.original_dataset = list(self.original_dataset) + augmented_data
        return self.original_dataset

def random_mask1(spike_data,original_dataset=None, mask_prob=0.2):
    mask = np.random.rand(*spike_data.shape) < mask_prob
    augmented_spikes = np.copy(spike_data)
    augmented_spikes[mask] = 0
    return augmented_spikes

def random_add(spike_data,original_dataset=None, add_prob=0.05):
    add = np.random.rand(*spike_data.shape) < add_prob
    augmented_spikes = np.copy(spike_data)
    augmented_spikes[add] = 1
    return augmented_spikes

def random_shift(spike_data, original_dataset=None, shift_max=5, shift_prob=0.1):
    augmented_spikes = np.copy(spike_data)
    C, T = spike_data.shape
    for c in range(C):
        indices = np.where(spike_data[c] == 1)[0]  # Get indices of spikes for each channel
        for idx in indices:
            if np.random.rand() < shift_prob:  # Only shift with a certain probability
                shift = np.random.randint(shift_max, shift_max + 1)  # Random shift
                new_idx = idx + shift
                # Ensure the new index is within bounds
                if 0 <= new_idx < T:
                    augmented_spikes[c, idx] = 0  # Remove spike from old position
                    augmented_spikes[c, new_idx] = 1  # Add spike to new position
    return augmented_spikes

def SpikeAug(spike_data, original_dataset):

    choice = np.random.randint(0, 3)  # Randomly choose between 0, 1, 2
    if choice == 0:
        return random_mask1(spike_data)
    elif choice == 1:
        return random_add(spike_data)
    else:
        return random_shift(spike_data)



# def local_zero(spike_data, original_dataset, perturb_window=50, perturb_prob=0.2):
#     perturbed_data = spike_data.copy()
#     for i in range(spike_data.shape[0]):
#         if np.random.rand() < perturb_prob:
#             start = np.random.randint(0, spike_data.shape[-1] - perturb_window)
#             end = start + perturb_window
#             perturbed_data[i, start:end] = 0
#     return perturbed_data

def random_mask(spike_data,original_dataset=None, mask_prob=0.1):
    mask = (spike_data == 1) & (np.random.rand(*spike_data.shape) < mask_prob)
    augmented_spikes = np.copy(spike_data)
    augmented_spikes[mask] = 0
    return augmented_spikes

def random_time_drop(spike_data,original_dataset=None, time_drop_prob=0.05):
    augmented_spikes = np.copy(spike_data)
    C, T = spike_data.shape
    if np.random.rand() < time_drop_prob:
        # 随机选择开始和结束的时间步
        start_idx = np.random.randint(0, T)
        end_idx = np.random.randint(start_idx, T)
        augmented_spikes[:, start_idx:end_idx] = 0  # 将该时间段的所有数据置为0
    return augmented_spikes

def random_channel_drop(spike_data,original_dataset=None, channel_drop_prob=0.1):
    augmented_spikes = np.copy(spike_data)
    C, T = spike_data.shape
    # 随机选择通道进行置0
    channels_to_drop = np.random.rand(C) < channel_drop_prob
    augmented_spikes[channels_to_drop, :] = 0
    return augmented_spikes

def eventdrop(spike_data,_):
    choice = np.random.randint(0, 3)  # 随机选择增强方式
    if choice == 0:
        return random_mask(spike_data)
    elif choice == 1:
        return random_time_drop(spike_data)
    else:
        return random_channel_drop(spike_data)

def NDA(spike_data, _):
    rotate_scale = 0.1
    shear_scale = 0.1
    data = np.copy(spike_data)
    choices = ['roll', 'rotate', 'shear']
    aug = np.random.choice(choices)
    if aug == 'roll':  # 类似图像的平移
        offset = random.randint(-10, 10)  # 平移范围
        data = np.roll(data, shift=offset, axis=-1)  # 对时间维度进行平移
    elif aug == 'rotate':  # 类似图像的旋转（可理解为时域的相位平移）
        shift_amount = int(data.shape[-1] * rotate_scale)
        data = np.roll(data, shift=shift_amount, axis=-1)

    elif aug == 'shear':  # 类似图像的剪切（通过时间维度的缩放实现）
        scale_factor = 1 + random.uniform(-shear_scale, shear_scale)
        new_length = int(data.shape[-1] * scale_factor)
        for i in range(data.shape[0]):  # 遍历每个通道
            interpolated = np.interp(np.linspace(0, data.shape[-1], new_length), np.arange(data.shape[-1]), data[i])
            # 如果插值后长度超过原始长度，则进行裁剪
            if interpolated.shape[-1] > data.shape[-1]:
                interpolated = interpolated[:data.shape[-1]]
            # 如果插值后长度不足原始长度，则进行填充
            elif interpolated.shape[-1] < data.shape[-1]:
                padding_length = data.shape[-1] - interpolated.shape[-1]
                interpolated = np.pad(interpolated, (0, padding_length), mode='edge')
            # 将处理后的数据放回原数组
            data[i] = interpolated

    return data