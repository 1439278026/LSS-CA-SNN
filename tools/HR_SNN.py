import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron
import math
import torch.nn.functional as F
import time
class LSTMCellModule(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=10):
        super(LSTMCellModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x: (b, 64)  # 单个时间步输入
        h_t, c_t: (b, 10)  # LSTM 隐状态
        输出: (b,10), (b,10)  # 当前时间步的 h_t, c_t
        """
        h_t, c_t = self.lstm(x)
        return h_t, c_t

class SpatialConvModule(nn.Module):
    def __init__(self, in_channels=10, out_channels=20):
        super(SpatialConvModule, self).__init__()
        # self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(1, in_channels))  # 1×10 2D卷积
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=10)

    def forward(self, x):
        """
        x: (b,10)  # 处理单个时间步的数据
        输出: (b,20)  # 输出单个时间步的数据
        """
        time1 = time.time()
        x = x.unsqueeze(1)  # (b,10) -> (b,1,10)
        time2 = time.time()

        x = self.conv(x)  # (b, 20, 1)
        time3 = time.time()

        x = x.squeeze(-1)  # (b,20)
        time4 = time.time()
        print(time2-time1,time3-time2,time4-time3)
        # x = x.unsqueeze(1).unsqueeze(2)  # (b,10) -> (b,1,1,10)
        # x = self.conv(x)  # (b,20,1,1)
        # x = x.squeeze(-1).squeeze(-1)  # (b,20)
        return x

class HybridLIF(nn.Module):
    """ 三种不同参数的 LIF 组合 """

    def __init__(self, tau_list, theta_list):
        super().__init__()
        assert len(tau_list) == len(theta_list) == 3  # 需要 3 组 LIF

        self.lif1 = neuron.MultiStepLIFNode(tau=tau_list[0], v_threshold=theta_list[0],
                                            backend='cupy', detach_reset=True)
        self.lif2 = neuron.MultiStepLIFNode(tau=tau_list[1], v_threshold=theta_list[1],
                                            backend='cupy', detach_reset=True)
        self.lif3 = neuron.MultiStepLIFNode(tau=tau_list[2], v_threshold=theta_list[2],
                                            backend='cupy', detach_reset=True)

    def forward(self, x):
        out1 = self.lif1(x)
        out2 = self.lif2(x)
        out3 = self.lif3(x)
        return torch.cat([out1, out2, out3], dim=-1)  # 拼接三种 LIF 响应

class HRSpikingModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 三种 LIF 参数
        tau_list = [1.5, 1.0, 0.8]
        for i in range(3):
            tau_list[i] = math.exp(-tau_list[i]) + 1
        theta_list = [0.14, 0.08, 0.06]
        self.hybrid_lif = HybridLIF(tau_list, theta_list)

        # 1x1 卷积减少通道数
        self.conv = nn.Conv1d(in_features, in_features, kernel_size=3)

        # 额外的 LIF 层
        self.lif4 = neuron.MultiStepLIFNode(tau=math.exp(-1) + 1, v_threshold=0.08,
                                            backend='cupy', detach_reset=True)

        # 全连接分类层
        self.fc = nn.Linear(in_features, out_features)

        self.skip_branch = nn.Conv1d(in_features, out_features, kernel_size=1)
    def forward(self, x):
        # x  b, 20, 1
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        skip = self.skip_branch(x).squeeze(-1)
        x = self.hybrid_lif(x) # [b, 20, 1] > [b, 20, 3]
        x = self.conv(x) # [b, 20, 3] > [b, 20, 1]
        x = x.squeeze(-1)
        x = self.lif4(x)
        x = self.fc(x)

        out = x + skip
        return out  # [b, 20]


class DP_Pooling(torch.nn.Module):
    def __init__(self, L_DP, N_DP):
        super().__init__()
        self.L_DP = L_DP  # 滑动窗口大小
        self.N_DP = N_DP  # 差分窗口大小

    def forward(self, x):
        B, N, T = x.shape
        if T % self.L_DP != 0:
            raise ValueError("T should be divisible by L_DP")

        # 计算窗口划分的索引
        num_windows = T // self.L_DP  # T 轴划分为 num_windows 个窗口
        x_reshaped = x.view(B, N, num_windows, self.L_DP)  # (B, N, num_windows, L_DP)

        # 计算后半部分和
        sum_late = x_reshaped[:, :, :, -self.N_DP:].sum(dim=-1)  # (B, N, num_windows)
        # 计算前半部分和
        sum_early = x_reshaped[:, :, :, :self.N_DP].sum(dim=-1)  # (B, N, num_windows)

        # 计算 DP 池化
        dp_out = (sum_late - sum_early) / self.N_DP  # (B, N, num_windows)

        return dp_out

class DPSD(nn.Module):
    def __init__(self, L_DP=5, N_DP=2, TA_ratio=2):
        """
        :param L_DP: DP池化的窗口大小
        :param N_DP: DP-kernel 计算的窗口大小
        :param TA_ratio: 时间注意力的缩放比例
        """
        super().__init__()
        self.L_DP = L_DP
        self.N_DP = N_DP
        self.TA_ratio = TA_ratio

        # 批归一化
        self.bn = nn.BatchNorm1d(num_features=1)

    def dp_pooling(self, x):
        """实现 DP-Pooling 操作"""
        B, N, T = x.shape
        num_windows = T // self.L_DP  # 确保可以整除

        # 重新 reshape 为窗口
        x_reshaped = x.view(B, N, num_windows, self.L_DP)  # (B, N, num_windows, L_DP)

        # 计算 DP-Kernel 池化
        sum_late = x_reshaped[:, :, :, -self.N_DP:].sum(dim=-1)  # (B, N, num_windows)
        sum_early = x_reshaped[:, :, :, :self.N_DP].sum(dim=-1)  # (B, N, num_windows)
        dp_out = (sum_late - sum_early) / self.N_DP  # (B, N, num_windows)

        return dp_out

    def avg_pooling(self, x):
        """在第一维（通道维度 N）进行平均池化"""
        return x.mean(dim=1,)  # (B, T)

    def temporal_attention(self, x):
        """基于公式 (12) 计算时间注意力"""
        B, T = x.shape

        # V2 输入 (B, N, T)
        x = self.fc1(x)  # (B, N, T/L_DP)
        x = F.relu(x)  # ReLU 激活
        x = self.fc2(x)  # (B, N, T/L_DP)
        attn_weights = F.softmax(x, dim=-1)  # softmax 归一化 (B, N, T/L_DP)

        return attn_weights  # (B, N, T/L_DP)

    def forward(self, x):
        """前向传播"""
        x = x.permute(0,2,1)
        B, N, T = x.shape
        if T % self.L_DP != 0:
            raise ValueError("T should be divisible by L_DP")

        # DP池化
        dp_out = self.dp_pooling(x)  # (B, N, T/L_DP)

        # 平均池化
        avg_out = self.avg_pooling(dp_out).unsqueeze(1)  # (B, T/L_DP)
        # 时间注意力
        # ta_out = self.temporal_attention(avg_out)  # (B, N, T/L_DP)

        # print(dp_out.shape,avg_out.shape)
        # Hadamard 乘积
        fused_out = dp_out * avg_out  # (B, N, T/L_DP)

        # Flatten 并批归一化
        output = fused_out.view(B, -1)  # 展平
        output = self.bn(output.unsqueeze(1)).squeeze(1)  # 批归一化

        return output

class classifier(nn.Module):
    def __init__(self, input_dim=200, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # 全连接层 (200 → 4)
        self.log_softmax = nn.LogSoftmax(dim=1)  # LogSoftmax 激活函数

    def forward(self, x):
        x = self.fc(x)  # 线性变换
        x = self.log_softmax(x)  # LogSoftmax
        return x

class HRSNN(nn.Module):
    def __init__(self, w=None, surrogate_function=None, time_step=480,in_channels=64,out_num=3):
        super(HRSNN, self).__init__()
        self.time_step = time_step
        self.channel = in_channels
        self.num_class = out_num


        self.lstm_cell = LSTMCellModule(input_dim=self.channel, hidden_dim=10)
        self.spatial_conv = SpatialConvModule()
        self.hr1 = HRSpikingModule(20, 20)
        self.hr2 = HRSpikingModule(20, 10)

        self.lif = neuron.MultiStepLIFNode(tau=math.exp(-1) + 1, v_threshold=0.08,
                                            backend='cupy', detach_reset=True)

        self.DPSD = DPSD()
        self.classifier = classifier(input_dim=200,num_classes=self.num_class)
    # def forward(self, x):
    #     """
    #     x: (b, 480, 64)  # batch=5, 480 时间步, 每步 64 维
    #     输出: (b, 480, 20)
    #     """
    #     x = x.permute(0,2,1)
    #     B,T,C = x.shape
    #     x = x.reshape(B*T,C)
    #     h_t, c_t = self.lstm_cell(x)  # (b, 64) -> (b, 10)
    #
    #     conv_out = self.spatial_conv(h_t)  # (b, 10) -> (b, 20)
    #
    #     out = self.hr1(conv_out)
    #     out = self.hr2(out)
    #
    #     out = out.reshape(B,T,-1)
    #     out = self.lif(out)
    #
    #     out = self.DPSD(out)
    #     out = self.classifier(out)
    #
    #     return out
    def forward(self, x):
        """
        x: (b, 480, 64)  # batch=5, 480 时间步, 每步 64 维
        输出: (b, 480, 20)
        """
        start = time.time()
        x = x.permute(0, 2, 1)
        print(f"permute: {time.time() - start:.6f} s")

        B, T, C = x.shape

        start = time.time()
        x = x.reshape(B * T, C)
        print(f"reshape: {time.time() - start:.6f} s")

        start = time.time()
        h_t, c_t = self.lstm_cell(x)  # (b, 64) -> (b, 10)
        print(f"LSTM: {time.time() - start:.6f} s")

        start = time.time()
        conv_out = self.spatial_conv(h_t)  # (b, 10) -> (b, 20)
        print(f"spatial_conv: {time.time() - start:.6f} s")

        start = time.time()
        out = self.hr1(conv_out)
        print(f"hr1: {time.time() - start:.6f} s")

        start = time.time()
        out = self.hr2(out)
        print(f"hr2: {time.time() - start:.6f} s")

        start = time.time()
        out = out.reshape(B, T, -1)
        print(f"reshape: {time.time() - start:.6f} s")

        start = time.time()
        out = self.lif(out)
        print(f"LIF: {time.time() - start:.6f} s")

        start = time.time()
        out = self.DPSD(out)
        print(f"DPSD: {time.time() - start:.6f} s")

        start = time.time()
        out = self.classifier(out)
        print(f"classifier: {time.time() - start:.6f} s")

        return out





if __name__ == '__main__':
    # 测试
    torch.cuda.set_device('cuda:0')

    # batch_size, in_channels, seq_len = 5, 20, 1
    # x = torch.randn(batch_size, in_channels).cuda()
    # # 初始化模型
    # model = HRSpikingModule(in_channels, 20).cuda()  # 10 个输出类别
    # output = model(x)
    #
    # print(output.shape)  # 预期输出: [16, 10]

    x = torch.randn(5, 80, 100).cuda()  # batch=5, 480 时间步, 每步 64 维
    model = HRSNN(time_step=100,in_channels=80).cuda()
    output = model(x)
    print(output.shape)  # 预期输出: torch.Size([5, 480, 20])

    # 示例
    # x = torch.randn(5, 480, 10)  # Batch=2, Channels=3, Time=12
    # dp_module = DPSD()
    # output = dp_module(x)
    # print(output.shape)  # (B, N*T/L_DP)
