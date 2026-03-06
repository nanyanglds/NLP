# -*- coding: utf-8 -*-
"""

@author: Dream Boy
"""
import torch
import numpy as np
import NeuronlikeProcess
from SynapticWeight import Syn_Weight


torch.set_grad_enabled(False)
torch.set_printoptions(profile="full")  # 让张量以完整形式在控制台显示，不受默认的简略显示规则限制


input_signal = torch.tensor([[[0.4]]])  # 输入信号

R = torch.tensor([[[[2e7]]], [[[5e7]]]])  # 神经元膜电阻
C = torch.tensor([[[[2e-9]]], [[[3e-9]]]])  # 神经元膜电容
q0 = 2e-9  # 初始输入信号和神经网络权重基础传递电荷量
lamda_q = 3  # 突触前递质与突触后电荷传递系数
t_input = 0.1  # 输入信号间隔，即输入信号频率表征
n = 100  # 循环次数
k = 0.2  # 初始突触权重
s = 1  # 区域激活函数

L = 2  # 神经网络层数
batch_size = input_signal.size(0)
M = input_signal.size(1)
N = input_signal.size(2)


# 重置训练权重
weit_dict = {
    'weit_ELTPD':torch.full((L, batch_size, M, N), 0),  # 早期LTP和LTD综合结果
    'weit_ELTPD_keep':torch.full((L, batch_size, M, N), 0),  # 保持ELTPD存储
    'weit_LLTPD':torch.full((L, batch_size, M, N), 0),  # 晚期LTP和LTD综合结果
    'weit_LLTPD_keep':torch.full((L, batch_size, M, N), k),  # 保持LLTPD存储
    'weit':torch.full((L, batch_size, M, N), k),  # 突触权重存储
    'weit_k':torch.full((L, batch_size, M, N), 0.3)  # 突触在整个递质分配中的比重，对于多个突触后神经元，每个神经元的递质分配比重
    }

MemP_dict = {
    'MemP':torch.zeros(L, batch_size, M, N),  # 膜电位
    'MemP_keep':torch.zeros(L, batch_size, M, N)  # 保持膜电位存储
    }

Ca_dict = {
    'Ca_keep':torch.zeros(L, batch_size, M, N),  # 保持Ca浓度
    'C_star_0':torch.zeros(L, batch_size, M, N),  # 保持[C**]积累量
    'N_star_0':torch.zeros(L, batch_size, M, N)  # 保持[N*]积累量
    }

t_dict = { 
    't_Leakage':torch.zeros(L, batch_size, M, N),  # 膜电位改变时间，计算膜电位漏电
    't_Spiking':torch.zeros(L, batch_size, M, N),  # 神经元激发时间
    't_keep':torch.zeros(L, batch_size, M, N)  # 记录时间点
    }

dict = {'weit_dict': weit_dict, 'MemP_dict': MemP_dict, 'Ca_dict': Ca_dict, 't_dict': t_dict}

W_ELTPD = dict['weit_dict']['weit_ELTPD']
W_ELTPD_keep = dict['weit_dict']['weit_ELTPD_keep']
W_LLTPD = dict['weit_dict']['weit_LLTPD']
W_LLTPD_keep = dict['weit_dict']['weit_LLTPD_keep']
W_dict = dict['weit_dict']['weit']
weit_k = dict['weit_dict']['weit_k']

MemPotential = dict['MemP_dict']['MemP']
MemP_keep = dict['MemP_dict']['MemP_keep']

Ca_keep = dict['Ca_dict']['Ca_keep']
C_star_0 = dict['Ca_dict']['C_star_0']
N_star_0 = dict['Ca_dict']['N_star_0']

T_Leakage = dict['t_dict']['t_Leakage']
T_Spiking = dict['t_dict']['t_Spiking']
t = dict['t_dict']['t_keep']


# excel输出转换
MemPotential_flaten = []
OutSpikes_flaten = []
w_dict_flaten_0 = []
w_ELTPD_flaten = []
w_LLTPD_flaten = []
delta_E_LTP_flaten = []
delta_E_LTD_flaten = []
delta_L_LTP_flaten = []
delta_L_LTD_flaten = []
C_star_flaten = []
N_star_flaten = []
Ca_flaten = []



# 过程变量存储
Signal = torch.zeros(L, batch_size, M, N).float()  # 各神经元输入信号
OutSpikes = torch.zeros(L, batch_size, M, N).float()  # Spiking状态
Q_transmitter = torch.zeros(L, batch_size, M, N).float()  # 生成递质量
delta_T_Spiking = torch.zeros(L, batch_size, M, N).float()  # 临近两次Spiking间隔时间
T_to_Spiking = torch.zeros(L, batch_size, M, N).float()  # 距离上次Spiking的时间

for i in range(n):
    Signal_allto0layer = input_signal * q0
    Signal_allto1layer = Q_transmitter[0] * weit_k[0] * (1 - torch.exp(-W_dict[0])) * s * lamda_q
    
    Signal[0] = Signal_allto0layer
    Signal[1] = Signal_allto1layer
    
    ### 膜电位MemPotential和时间记录表T_Leakage更新
    t = t + t_input
    
    ### 激发
    neuron_excit_encape = NeuronlikeProcess.MLN(L, batch_size, M, N)  # 封装为.pyc文件
    output_spikes, memP_keep, membrane_potential, q_trans, ca_keep, ca, t_MemP_leakage, t_spiking, delta_t_spiking, t_to_spiking = neuron_excit_encape(MemP_keep, Signal, R, C, Ca_keep, T_Leakage, T_Spiking, t)
    OutSpikes = output_spikes  # 输出Spike
    MemP_keep = memP_keep  # 保持膜电位
    MemPotential = membrane_potential  # 膜电位
    Q_transmitter = q_trans  # 生成递质量
    Ca_keep = ca_keep  # 保持Ca浓度
    Ca_gener = ca  # 本次生成Ca浓度
    T_Leakage = t_MemP_leakage  # 膜电位改变时间
    T_Spiking = t_spiking  # 神经元激发时间
    delta_T_Spiking = delta_t_spiking  # 临近两次Spiking间隔时间
    T_to_Spiking = t_to_spiking  # 距离上次Spiking的时间
    
    
    Spiking_mask = OutSpikes.clone()
    
    ###突触权重更新
    w_dict, w_ELTPD_keep, w_LLTPD_keep, w_ELTPD, w_LLTPD, delta_e_ltp, delta_l_ltp, delta_e_ltd, delta_l_ltd, c_star_0, n_star_0, c_star, n_star = Syn_Weight(L, batch_size, M, N).forward(W_ELTPD_keep, W_LLTPD_keep, Ca_keep, delta_T_Spiking, T_to_Spiking, C_star_0, N_star_0, Spiking_mask)
    W_dict = w_dict  # 突触权重
    W_ELTPD_keep = w_ELTPD_keep  # 保持ELTPD
    W_LLTPD_keep = w_LLTPD_keep  # 保持LLTPD
    W_ELTPD = w_ELTPD  # 早期LTP和LTD综合结果
    W_LLTPD = w_LLTPD  # 晚期LTP和LTD综合结果
    delta_E_LTP = delta_e_ltp  # 当次激发ELTP增量
    delta_L_LTP = delta_l_ltp  # 当次激发LLTP增量
    delta_E_LTD = delta_e_ltd  # 当次激发ELTD增量
    delta_L_LTD = delta_l_ltd  # 当次激发LLTD增量
    C_star_0 = c_star_0  # 保持[C**]积累量
    N_star_0 = n_star_0  # 保持[N*]积累量
    C_star = c_star  # [C**]积累量
    N_star = n_star  # [N*]积累量
    
    print("W_dict[0]",": ", i, W_dict[0])
    
    
    
    MemPotential_list = MemPotential.clone()
    MemPotential_flaten.append(MemPotential_list.numpy())
    
    OutSpikes_list = OutSpikes.clone()
    OutSpikes_flaten.append(OutSpikes_list.numpy())
    
    w_dict_list_0 = W_dict[0].clone()
    w_dict_flaten_0.append(w_dict_list_0.numpy())
    
    w_ELTPD_list = W_ELTPD[1].clone()
    w_ELTPD_flaten.append(w_ELTPD_list.numpy())
    w_LLTPD_list = W_LLTPD[1].clone()
    w_LLTPD_flaten.append(w_LLTPD_list.numpy())
    
    delta_E_LTP_list = delta_E_LTP[1].clone()
    delta_E_LTP_flaten.append(delta_E_LTP_list.numpy())
    delta_E_LTD_list = - delta_E_LTD[1].clone()
    delta_E_LTD_flaten.append(delta_E_LTD_list.numpy())
    delta_L_LTP_list = delta_L_LTP[1].clone()
    delta_L_LTP_flaten.append(delta_L_LTP_list.numpy())
    C_star_list = C_star[1].clone()
    C_star_flaten.append(C_star_list.numpy())
    delta_L_LTD_list = - delta_L_LTD[1].clone()
    delta_L_LTD_flaten.append(delta_L_LTD_list.numpy())
    N_star_list = N_star[1].clone()
    N_star_flaten.append(N_star_list.numpy())
    
    Ca_list = Ca_gener[1].clone()
    Ca_flaten.append(Ca_list.numpy())
    

# 保存训练权重字典
dict['weit_dict']['weit_ELTPD'] = W_ELTPD
dict['weit_dict']['weit_ELTPD_keep'] = W_ELTPD_keep
dict['weit_dict']['weit_LLTPD'] = W_LLTPD
dict['weit_dict']['weit_LLTPD_keep'] = W_LLTPD_keep
dict['weit_dict']['weit'] = W_dict
dict['weit_dict']['weit_k'] = weit_k

dict['MemP_dict']['MemP'] = MemPotential
dict['MemP_dict']['MemP_keep'] = MemP_keep

dict['Ca_dict']['Ca_keep'] = Ca_keep
dict['Ca_dict']['C_star_0'] = C_star_0
dict['Ca_dict']['N_star_0'] = N_star_0

dict['t_dict']['T_Leakage'] = T_Leakage
dict['t_dict']['T_Spiking'] = T_Spiking  # 时刻记录表
dict['t_dict']['t_keep'] = t


torch.save(dict, 'model_dict模拟.pth')  # 保存训练权重字典



MemPotential_data = np.vstack(MemPotential_flaten).reshape(-1, 2)
np.savetxt('1.MemPotential_data.csv', MemPotential_data, delimiter=',')

OutSpikes_data = np.vstack(OutSpikes_flaten).reshape(-1, 2)
np.savetxt('2.OutSpikes_data.csv', OutSpikes_data, delimiter=',')

w_dict_data_0 = np.vstack(w_dict_flaten_0).reshape(-1, 1)
np.savetxt('3.w_dict_data_0.csv', w_dict_data_0, delimiter=',')

w_ELTPD_data = np.vstack(w_ELTPD_flaten).reshape(-1, 1)
np.savetxt('4.w_ELTPD_data.csv', w_ELTPD_data, delimiter=',')
w_LLTPD_data = np.vstack(w_LLTPD_flaten).reshape(-1, 1)
np.savetxt('5.w_LLTPD_data.csv', w_LLTPD_data, delimiter=',')

delta_E_LTP_data = np.vstack(delta_E_LTP_flaten).reshape(-1, 1)
np.savetxt('6.delta_E_LTP_data.csv', delta_E_LTP_data, delimiter=',')
delta_E_LTD_data = np.vstack(delta_E_LTD_flaten).reshape(-1, 1)
np.savetxt('7.delta_E_LTD_data.csv', delta_E_LTD_data, delimiter=',')
delta_L_LTP_data = np.vstack(delta_L_LTP_flaten).reshape(-1, 1)
np.savetxt('8.delta_L_LTP_data.csv', delta_L_LTP_data, delimiter=',')
C_star_data = np.vstack(C_star_flaten).reshape(-1, 1)
np.savetxt('9.C_star_data.csv', C_star_data, delimiter=',')
delta_L_LTD_data = np.vstack(delta_L_LTD_flaten).reshape(-1, 1)
np.savetxt('10.delta_L_LTD_data.csv', delta_L_LTD_data, delimiter=',')
N_star_data = np.vstack(N_star_flaten).reshape(-1, 1)
np.savetxt('11.N_star_data.csv', N_star_data, delimiter=',')

Ca_data = np.vstack(Ca_flaten).reshape(-1, 1)
np.savetxt('12.Ca_data.csv', Ca_data, delimiter=',')



