import numpy as np
import gym
import gc
import os
import sys
import gym_environments
import random
import mpnn as gnn
import tensorflow as tf
from collections import deque
import multiprocessing
import time as tt
import glob
from read_tles import read_tles
from distance_tools import create_basic_ground_station_for_satellite_shadow,distance_m_between_satellites
import torch
import datetime
import time
import math
from mpnn import GCN
from sklearn.metrics import mean_squared_error
import torch.nn as nn

high_model = GCN()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ENV_NAME = 'GraphEnv-v1'
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
ITERATIONS = 10000
# TRAINING_EPISODES = 20
TRAINING_EPISODES = 1
# EVALUATION_EPISODES = 40
EVALUATION_EPISODES = 1
# FIRST_WORK_TRAIN_EPISODE = 60
FIRST_WORK_TRAIN_EPISODE = 1

MULTI_FACTOR_BATCH = 6 # Number of batches used in training
TAU = 0.08 # Only used in soft weights copy

differentiation_str = "sample_DQN_agent"
checkpoint_dir = "./models"+differentiation_str
store_loss = 3 # Store the loss every store_loss batches

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)



tf.random.set_seed(1)

train_dir = "./TensorBoard/"+differentiation_str
# summary_writer = tf.summary.create_file_writer(train_dir)
listofDemands = [8* (10**6/2000), 32* (10**6/2000), 64* (10**6/2000)]
# listofDemands = [0, 0, 0]
# copy_weights_interval = 50
copy_weights_interval = 5
evaluation_interval = 1
# epsilon_start_decay = 70
epsilon_start_decay = 5


hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.0001,
    # 'learning_rate': 0.01,
    'batch_size': 30,
# 'batch_size': 10,
    'T': 3,
    'num_demands': len(listofDemands)
}

MAX_QUEUE_SIZE = 4000

NORM = 10000000

N =  2 * 5
sat_per_orbit = 5
orbit_num = 2
# N = 72*22
# sat_per_orbit = 22
# orbit_num = 72
# N = 36*20
# sat_per_orbit = 20
# orbit_num = 36

MY_FILE = "sat0.txt"
PERIOD = 100
# GAMMA = 0.1
GAMMA = 1000
NORM_REWARD = 1000000
my_tle = read_tles(MY_FILE)

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes



criterion = torch.nn.MSELoss()

from torch_geometric.data import Data

from torch.utils.data import DataLoader, Dataset

def time_transfer(time_slot):
    start = '2000/01/01 00:00:00'
    # 先将字符串转化为时间格式
    a = datetime.datetime.strptime(start, "%Y/%m/%d %H:%M:%S")
    c = a + datetime.timedelta(seconds=time_slot)
    timeArray = time.strptime(str(c), "%Y-%m-%d %H:%M:%S")
    ts = time.strftime("%Y/%m/%d %H:%M:%S", timeArray)
    return str(ts)



def index_sat_bei_nan(idx):
    curr_plane = int(idx/sat_per_orbit)
    offset1 = ((idx % sat_per_orbit) + 1) % sat_per_orbit
    offset2 = ((idx % sat_per_orbit) - 1) % sat_per_orbit
    return curr_plane*sat_per_orbit + offset1, curr_plane*sat_per_orbit + offset2

def sw_angle(sat, sat1, sat2, curr_time):

    c = distance_m_between_satellites(my_tle['satellites'][sat1], my_tle['satellites'][sat2],
                                                   time_transfer(curr_time),
                                                   time_transfer(curr_time))
    a = distance_m_between_satellites(my_tle['satellites'][sat1], my_tle['satellites'][sat],
                                                   time_transfer(curr_time),
                                                   time_transfer(curr_time))
    b = distance_m_between_satellites(my_tle['satellites'][sat2], my_tle['satellites'][sat],
                                  time_transfer(curr_time),
                                  time_transfer(curr_time))

    # c^2 = a^2 + b^2 - 2 * a * b * cos(C)
    result = 0

    if (a**2+b**2-c**2)/(2*a*b) >= 1:
        result = 0
    elif (a**2+b**2-c**2)/(2*a*b) <= -1:
        result = math.pi
    else:
        result = math.acos((a**2+b**2-c**2)/(2*a*b))
    print(result)

    return result/NORM_REWARD



def ternary(n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))


def normalization(data):
    """
    归一化函数
    把所有数据归一化到[0，1]区间内，数据列表中的最大值和最小值分别映射到1和0，所以该方法一定会出现端点值0和1。
    此映射是线性映射，实质上是数据在数轴上等比缩放。

    :param data: 数据列表，数据取值范围：全体实数
    :return:
    """
    # print(data)
    min_value = min(data)
    max_value = max(data)
    new_list = []
    for i in data:
        new_list.append((i - min_value) / (max_value - min_value+ 0.001))
    return new_list




def node_traffic_intensity_list(sor_list, dst_list, demand_list, time_list):
    ## return the traffic intensity list for all the nodes in the next D time slot

    # print(sor_list)
    # print(dst_list)
    # print(demand_list)
    # print(time_list)

    my_list = [0] * N
    list_cnt = 0
    for i in sor_list:
        curr_locality_index = [index_sat_bei_nan((i - sat_per_orbit) % N)[0], (i - sat_per_orbit) % N,
                           index_sat_bei_nan((i - sat_per_orbit) % N)[1], index_sat_bei_nan(i)[0], i,
                           index_sat_bei_nan(i)[1],
                           index_sat_bei_nan((i + sat_per_orbit) % N)[0], (i + sat_per_orbit) % N,
                           index_sat_bei_nan((i + sat_per_orbit) % N)[1]]
        my_list[curr_locality_index[4]] += 2 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[0]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[1]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[2]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[3]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[5]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[6]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[7]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[8]] += 1 * demand_list[list_cnt] * time_list[list_cnt]

        list_cnt += 1

    list_cnt = 0
    for i in dst_list:
        curr_locality_index = [index_sat_bei_nan((i - sat_per_orbit) % N)[0], (i - sat_per_orbit) % N,
                               index_sat_bei_nan((i - sat_per_orbit) % N)[1], index_sat_bei_nan(i)[0], i,
                               index_sat_bei_nan(i)[1],
                               index_sat_bei_nan((i + sat_per_orbit) % N)[0], (i + sat_per_orbit) % N,
                               index_sat_bei_nan((i + sat_per_orbit) % N)[1]]
        my_list[curr_locality_index[4]] += 2 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[0]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[1]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[2]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[3]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[5]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[6]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[7]] += 1 * demand_list[list_cnt] * time_list[list_cnt]
        my_list[curr_locality_index[8]] += 1 * demand_list[list_cnt] * time_list[list_cnt]

        list_cnt += 1

    return normalization(my_list)



def ISL_link_possibility(upper_limit_sat_per_orbit = 5):
    ## return a right linkage index list
    ## return [[],[],...]
    # ...
    # 3
    # 2
    # 1
    # 0
    my_list = []

    for i in range(3**upper_limit_sat_per_orbit):
        should_add = True
        my_ternary = ternary(i)
        if len(my_ternary) < upper_limit_sat_per_orbit:
            diff = upper_limit_sat_per_orbit - len(my_ternary)
            my_ternary = '0'*diff + my_ternary

        curr_list = []
        for n in range(upper_limit_sat_per_orbit):
            if (not n == 0) and (not n == upper_limit_sat_per_orbit-1):
                if my_ternary[n] == '0':
                    curr_list.append((n-1)%upper_limit_sat_per_orbit)
                elif my_ternary[n] == '1':
                    curr_list.append(n % upper_limit_sat_per_orbit)
                elif my_ternary[n] == '2':
                    curr_list.append((n+1) % upper_limit_sat_per_orbit)
            elif n == 0:
                if my_ternary[n] == '1' or my_ternary[n] == '0':
                    curr_list.append(n % upper_limit_sat_per_orbit)
                elif my_ternary[n] == '2':
                    curr_list.append((n + 1) % upper_limit_sat_per_orbit)
            else:
                if my_ternary[n] == '0':
                    curr_list.append((n-1)%upper_limit_sat_per_orbit)
                elif my_ternary[n] == '1' or my_ternary[n] == '2':
                    curr_list.append(n % upper_limit_sat_per_orbit)

            if curr_list[-1] in curr_list[:-1]:
                should_add = False
                break
        if should_add:
            my_list.append(curr_list)

    return my_list








## inferring
class MyDataset(Dataset):
    def __init__(self, N, curr_time, env_eval):


        ## sat_num * 9 (连接方式)个
        self.data = []

        ## 先iterate all agents, then iterate all linkage possibilities
        ## node feature (intensity)

        ## policy network: concanate with 1) ISL sw cost  2). 中间node左右连的两个ISL的距离和

        ### locality_index_list
        ## 0 3 6
        ## 1 4 7
        ## 2 5 8
        # for i in range(10):
        for i in range(N):
            # print("load dataset for: ")
            # print(i)

            # for j in range(9):
            curr_locality_index = [index_sat_bei_nan((i-sat_per_orbit)%N)[0],(i-sat_per_orbit)%N,
                                   index_sat_bei_nan((i-sat_per_orbit)%N)[1],index_sat_bei_nan(i)[0],i,index_sat_bei_nan(i)[1],
                                   index_sat_bei_nan((i+sat_per_orbit)%N)[0],(i+sat_per_orbit)%N,index_sat_bei_nan((i+sat_per_orbit)%N)[1]]
            # print(curr_locality_index)
            x = []
            for m in range(9):
                x.append([node_traffic_intensity_list(env_eval.sor_list, env_eval.dst_list,
                                                      env_eval.demand_list, env_eval.time_list)[curr_locality_index[m]]])
            x = torch.Tensor(x)

            ## 得到当前左右连的index
            curr_link_list = []

            for n in range(3):
                if env_eval.graph.has_edge(curr_locality_index[n],curr_locality_index[4]):
                    curr_link_list.append(curr_locality_index[n])
                    break

            for n in range(6,9):
                if env_eval.graph.has_edge(curr_locality_index[n],curr_locality_index[4]):
                    curr_link_list.append(curr_locality_index[n])
                    break




            # x = torch.Tensor([[1, 2], [2, 4], [i, i + 1]])
            edge_index = torch.LongTensor([[0, 1, 1,2,3,4,4,5,6,7,7,8,0,4,4,6], [1, 0, 2,1,4,3,5,4,7,6,8,7,4,0,6,4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[0]], my_tle['satellites'][curr_locality_index[4]],
                                                   time_transfer(curr_time),
                                                   time_transfer(curr_time))+
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]], my_tle['satellites'][curr_locality_index[6]],
                                                   time_transfer(curr_time),
                                                   time_transfer(curr_time)))/NORM
            sw = sw_angle(curr_locality_index[4],curr_link_list[0],curr_locality_index[0],curr_time) +\
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[6], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 0, 4, 4, 7], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 0, 7, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[0]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[7]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[0], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[7], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 0, 4, 4, 8], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 0, 8, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[0]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[8]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[0], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[8], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 1, 4, 4, 6], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 1, 6, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[1]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[6]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[1], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[6], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 1, 4, 4, 7], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 1, 7, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[1]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[7]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[1], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[7], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 1, 4, 4, 8], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 1, 8, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[1]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[8]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[1], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[8], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 2, 4, 4, 6], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 2, 6, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[2]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[6]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[2], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[6], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 2, 4, 4, 7], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 2, 7, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[2]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[7]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM

            # print(distance_sum)

            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[2], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[7], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)

            edge_index = torch.LongTensor(
                [[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 2, 4, 4, 8], [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 4, 2, 8, 4]])
            # data = Data(x=x, edge_index=edge_index, batch = torch.LongTensor([i]*10), y=i)
            distance_sum = (distance_m_between_satellites(my_tle['satellites'][curr_locality_index[2]],
                                                          my_tle['satellites'][curr_locality_index[4]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time)) +
                            distance_m_between_satellites(my_tle['satellites'][curr_locality_index[4]],
                                                          my_tle['satellites'][curr_locality_index[8]],
                                                          time_transfer(curr_time),
                                                          time_transfer(curr_time))) / NORM
            sw = sw_angle(curr_locality_index[4], curr_link_list[0], curr_locality_index[2], curr_time) + \
                 sw_angle(curr_locality_index[4], curr_link_list[1], curr_locality_index[8], curr_time)

            data = Data(x=x, edge_index=edge_index, distance_sum=torch.FloatTensor([distance_sum]), sw=torch.FloatTensor([sw]))
            self.data.append(data)



    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # node, edge, batch = self.data[idx]
        # return node, edge, batch
        # return self.data[idx].x, self.data[idx].edge_index, self.data[idx].batch, self.data[idx].y
        return self.data[idx].x, self.data[idx].edge_index, self.data[idx].distance_sum, self.data[idx].sw






class High_agent:
    def __init__(self, batch_size):
        self.memory = deque(maxlen=MAX_QUEUE_SIZE)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.epsilon_decay = 0.985
        self.primary_network = high_model
        self.target_network = high_model
        self.optimizer_primary = torch.optim.Adam(self.primary_network.parameters(), lr=0.001)
        self.optimizer_target = torch.optim.Adam(self.target_network.parameters(), lr=0.001)
        self.numbersamples = batch_size
        self.listQValues = None
        self.numbersamples_high = 32
        # criterion = torch.nn.MSELoss()

    def train(self, curr_time, env_training):
        model.train()

        dataset = MyDataset(N=N, curr_time=curr_time, env_eval=env_training)
        data_loader = DataLoader(dataset, batch_size=batch_number)

        for a, b, c, d in data_loader:
            a = a.reshape((batch_number * node_number, -1))

            b = b.reshape((2, -1))

            c2 = []
            for i in range(batch_number):
                for _ in range(node_number):
                    c2.append(i)
            c2 = torch.LongTensor(c2)
            d = torch.LongTensor(d)

            out = model(a, b, c2)

            out = out.to(torch.float32)
            d = d.to(torch.float32)

            loss = criterion(out, d)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()



    def my_forward(self, curr_time, env_eval, is_test, epi_it):

        ## N rows, each contains a concatted vector of next states [a, b,c,d] that induces a max softmax
        next_state_max = []
        edge_score_dict = dict()
        next_next_state = []

        self.primary_network.eval()
        dataset = MyDataset(N=N, curr_time=curr_time, env_eval=env_eval)
        data_loader = DataLoader(dataset, batch_size=9)
        loss = 0
        out_list = []
        sw_switching = 0
        agent_number = 0
        for a, b, c, d in data_loader:

            ## 循环内每次只为一个node选，一共9种连接情况
            # a = a.reshape((9 * N, -1))
            # out = []
            a = a.reshape((9 * 9, -1))

            b_original = []
            for i in range(9):
                b_original.append(b[i])

            b = b.reshape((2, -1))

            c2 = []
            for i in range(9):
                for _ in range(9):
                    c2.append(i)
            c2 = torch.LongTensor(c2)
            c = torch.FloatTensor(c)
            d = torch.FloatTensor(d)
            # out = model(a, b, c2)
            random.seed(int(25 + curr_time+agent_number+epi_it*2))
            my_draw = random.random()



            next_next_state.append([a,b,c,d])

            if is_test or my_draw > self.epsilon:
                ## 取最大
                out = self.primary_network(a, b, c2, c, d)
                out = out.to(torch.float32)
            # d = d.to(torch.float32)
            # loss += criterion(out, d)
                out_list.append(out)

                # print("Not random!")
            else:
                ## leizhui
                out = self.primary_network(a, b, c2, c, d)
                random.seed(int(24+curr_time+agent_number+epi_it))
                # for aa in range(9):
                #     out[aa]=random.random()
                out[0] = random.random()
                out[1] = random.random()
                out[2] = random.random()
                out[3] = random.random()
                out[4] = random.random()
                out[5] = random.random()
                out[6] = random.random()
                out[7] = random.random()
                out[8] = random.random()
                out = out.to(torch.float32)
                out_list.append(out)
                # print("Random!")
            #
            # print("Out:")
            # print(out)
            individual_max_choice_index = torch.argmax(out)
            # print("Choice index:")
            # print(individual_max_choice_index)

            next_state_max.append([a[9*individual_max_choice_index: 9*(individual_max_choice_index+1)],
                                   b_original[individual_max_choice_index],torch.unsqueeze(c[individual_max_choice_index],0),
                                   torch.unsqueeze(d[individual_max_choice_index],0)])


            sw_switching += d[individual_max_choice_index][0]

            agent_number += 1


        ## edge score accumulation

        for i in range(N):
            # for j in range(9):
            curr_locality_index = [index_sat_bei_nan((i - sat_per_orbit) % N)[0], (i - sat_per_orbit) % N,
                                   index_sat_bei_nan((i - sat_per_orbit) % N)[1], index_sat_bei_nan(i)[0], i,
                                   index_sat_bei_nan(i)[1],
                                   index_sat_bei_nan((i + sat_per_orbit) % N)[0], (i + sat_per_orbit) % N,
                                   index_sat_bei_nan((i + sat_per_orbit) % N)[1]]

            ## the softmax output of choices of a node
            curr_edge_score = out_list[i]

            # print(len(curr_edge_score))

            for n in [0, 1, 2, 6, 7, 8]:
                if str(curr_locality_index[4]) + ':' + str(curr_locality_index[n]) not in edge_score_dict:
                    edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[n])] = 0

            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[0])] += curr_edge_score[0]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[6])] += curr_edge_score[0]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[0])] += curr_edge_score[1]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[7])] += curr_edge_score[1]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[0])] += curr_edge_score[2]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[8])] += curr_edge_score[2]

            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[1])] += curr_edge_score[3]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[6])] += curr_edge_score[3]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[1])] += curr_edge_score[4]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[7])] += curr_edge_score[4]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[1])] += curr_edge_score[5]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[8])] += curr_edge_score[5]

            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[2])] += curr_edge_score[6]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[6])] += curr_edge_score[6]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[2])] += curr_edge_score[7]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[7])] += curr_edge_score[7]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[2])] += curr_edge_score[8]
            edge_score_dict[str(curr_locality_index[4]) + ':' + str(curr_locality_index[8])] += curr_edge_score[8]

        all_choice = ISL_link_possibility(5)
        ## right true index
        final_linkage = []
        for i in range(orbit_num):
            for j in range(int(sat_per_orbit / 5)):
                max_val = -1000000
                max_choice = 0
                for curr_choice in all_choice:
                    curr_val = 0
                    s = 0
                    for t in curr_choice:
                        my_left = i * sat_per_orbit + s + 5 * j
                        my_right = ((i + 1) * sat_per_orbit + t + 5 * j) % N
                        curr_val += edge_score_dict[str(my_left) + ':' + str(my_right)]
                        curr_val += edge_score_dict[str(my_right) + ':' + str(my_left)]
                        s += 1
                    if curr_val > max_val:
                        max_val = curr_val
                        max_choice = curr_choice

                ## fix test
                # max_choice = all_choice[2]

                s2 = 0
                for t2 in max_choice:
                    # my_left2 = i * sat_per_orbit + s2
                    my_right2 = ((i + 1) * sat_per_orbit + t2 + 5 * j) % N
                    s2 += 1
                    final_linkage.append(my_right2)

        return next_state_max, final_linkage, sw_switching, next_next_state




    def replay(self, epi):
        batch = random.sample(self.memory, self.numbersamples_high)

        left_list = []
        right_list = []

        for x in batch:
            self.primary_network.train()
            c2 = torch.LongTensor([0]*9)
            qsa = self.primary_network(x[2][0], x[2][1], c2, x[2][2], x[2][3])
            r = x[0] - GAMMA * x[1]
            self.target_network.eval()
            c3 = []
            for i in range(9):
                for _ in range(9):
                    c3.append(i)

            c3 = torch.LongTensor(c3)


            # print(self.target_network(x[3][0], x[3][1], c3, x[3][2], x[3][3]))


            q_next_max = torch.max(self.target_network(x[3][0], x[3][1], c3, x[3][2], x[3][3])).detach()
            left_list.append(qsa)
            right_list.append(r+self.gamma*q_next_max)


        left_list = torch.tensor(left_list,dtype=torch.float32, requires_grad=True)
        right_list = torch.tensor(right_list, dtype=torch.float32)

        loss = criterion(left_list, right_list)
        print("loss: ")
        print(loss)
        loss.backward()
        self.optimizer_primary.step()
        self.optimizer_primary.zero_grad()

        if epi % copy_weights_interval == 0:
            self.target_network.load_state_dict(self.primary_network.state_dict())





class DQNAgent:
    def __init__(self, batch_size):
        self.memory = deque(maxlen=MAX_QUEUE_SIZE)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.writer = None
        # self.K = 10 # K-paths
        self.K = 4  # K-paths
        self.listQValues = None
        self.numbersamples = batch_size
        self.action = None
        self.capacity_feature = None
        self.bw_allocated_feature = np.zeros((env_training.numEdges,len(env_training.listofDemands)))

        self.global_step = 0
        self.primary_network = gnn.myModel(hparams)
        self.primary_network.build()
        self.target_network = gnn.myModel(hparams)
        self.target_network.build()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'],momentum=0.9,nesterov=True)

    def act(self, env, state, demand, source, destination, flagEvaluation):
        """
        Given a demand stored in the environment it allocates the K=4 shortest paths on the current 'state'
        and predicts the q_values of the K=4 different new graph states by using the GNN model.
        Picks the state according to epsilon-greedy approach. The flag=TRUE indicates that we are testing
        the model and thus, it won't activate the drop layers.
        """
        # Set to True if we need to compute K=4 q-values and take the maxium
        takeMax_epsilon = False
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # Initialize action
        action = 0

        # We get the K-paths between source-destination
        # pathList = env.allPaths[str(source) +':'+ str(destination)]
        pathList = env.num_shortest_path( source,destination, demand)




        path = 0

        # 1. Implement epsilon-greedy to pick allocation
        # If flagEvaluation==TRUE we are EVALUATING => take always the action that the agent is saying has higher q-value
        # Otherwise, we are training with normal epsilon-greedy strategy
        if flagEvaluation:
            # If evaluation, compute K=4 q-values and take the maxium value
            takeMax_epsilon = True
        else:
            # If training, compute epsilon-greedy
            z = np.random.random()
            if z > self.epsilon:
                # Compute K=4 q-values and pick the one with highest value
                # In case of multiple same max values, return the first one
                takeMax_epsilon = True
            else:
                # Pick a random path and compute only one q-value
                path = np.random.randint(0, len(pathList))
                action = path

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path]
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate demand to bw_allocated
            while (j < len(currentPath)):
                state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute their q-values using cummax
            listGraphs.append(state_copy)
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)

            if not takeMax_epsilon:
                # If we don't need to compute the K=4 q-values we exit
                break

            path = path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        # Predict qvalues for all graphs within tensors
        self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                        tensors['second'], tensors['num_edges'], training=False).numpy()

        if takeMax_epsilon:
            # We take the path with highest q-value
            action = np.argmax(self.listQValues)
        else:
            return pathList, path, list_k_features[0]

        ## action, 做了这个action后 图的特征
        return pathList, action, list_k_features[action]
    
    def get_graph_features(self, env, copyGraph):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.bw_allocated_feature.fill(0.0)
        # Normalize capacity feature
        # self.capacity_feature = (copyGraph[:,0] - 100.00000001) / 200.0
        self.capacity_feature = (copyGraph[:, 0] - 10**6/2/50+0.01) / (10**6/50)

        iter = 0
        for i in copyGraph[:, 1]:
            if i == listofDemands[0]:
                self.bw_allocated_feature[iter][0] = 1
            elif i == listofDemands[1]:
                self.bw_allocated_feature[iter][1] = 1
            elif i == listofDemands[2]:
                self.bw_allocated_feature[iter][2] = 1
            iter = iter + 1
        
        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        # print("bet")
        # print(env.between_feature)

        sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2 - hparams['num_demands']]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs
    
    def _write_tf_summary(self, gradients, loss):
        with summary_writer.as_default():
            tf.summary.scalar(name="loss", data=loss[0], step=self.global_step)
            tf.summary.histogram(name='gradients_5', data=gradients[5], step=self.global_step)
            tf.summary.histogram(name='gradients_7', data=gradients[7], step=self.global_step)
            tf.summary.histogram(name='gradients_9', data=gradients[9], step=self.global_step)
            tf.summary.histogram(name='FirstLayer/kernel:0', data=self.primary_network.variables[0], step=self.global_step)
            tf.summary.histogram(name='FirstLayer/bias:0', data=self.primary_network.variables[1], step=self.global_step)
            tf.summary.histogram(name='kernel:0', data=self.primary_network.variables[2], step=self.global_step)
            tf.summary.histogram(name='recurrent_kernel:0', data=self.primary_network.variables[3], step=self.global_step)
            tf.summary.histogram(name='bias:0', data=self.primary_network.variables[4], step=self.global_step)
            tf.summary.histogram(name='Readout1/kernel:0', data=self.primary_network.variables[5], step=self.global_step)
            tf.summary.histogram(name='Readout1/bias:0', data=self.primary_network.variables[6], step=self.global_step)
            tf.summary.histogram(name='Readout2/kernel:0', data=self.primary_network.variables[7], step=self.global_step)
            tf.summary.histogram(name='Readout2/bias:0', data=self.primary_network.variables[8], step=self.global_step)
            tf.summary.histogram(name='Readout3/kernel:0', data=self.primary_network.variables[9], step=self.global_step)
            tf.summary.histogram(name='Readout3/bias:0', data=self.primary_network.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1

    @tf.function
    def _forward_pass(self, x):
        prediction_state = self.primary_network(x[0], x[1], x[2], x[3], x[4], training=True)
        preds_next_target = tf.stop_gradient(self.target_network(x[6], x[7], x[9], x[10], x[11], training=True))
        return prediction_state, preds_next_target

    def _train_step(self, batch):
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            preds_state = []
            target = []
            for x in batch:
                prediction_state, preds_next_target = self._forward_pass(x)
                # Take q-value of the action performed
                preds_state.append(prediction_state[0])
                # We multiple by 0 if done==TRUE to cancel the second term
                target.append(tf.stop_gradient([x[5] + self.gamma*tf.math.reduce_max(preds_next_target)*(1-x[8])]))

            loss = tf.keras.losses.MSE(tf.stack(target, axis=1), tf.stack(preds_state, axis=1))
            # Loss function using L2 Regularization
            regularization_loss = sum(self.primary_network.losses)
            loss = loss + regularization_loss

        # Computes the gradient using operations recorded in context of this tape
        grad = tape.gradient(loss, self.primary_network.variables)
        #gradients, _ = tf.clip_by_global_norm(grad, 5.0)
        gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        self.optimizer.apply_gradients(zip(gradients, self.primary_network.variables))
        del tape
        return grad, loss
    
    def replay(self, episode):
        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(self.memory, self.numbersamples)
            
            grad, loss = self._train_step(batch)
            # if i%store_loss==0:
            #     fileLogs.write(".," + '%.9f' % loss.numpy() + ",\n")
        
        # Soft weights update
        # for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
        #     t.assign(t * (1 - TAU) + e * TAU)

        # Hard weights update
        if episode % copy_weights_interval == 0:
            self.target_network.set_weights(self.primary_network.get_weights()) 
        # if episode % evaluation_interval == 0:
        #     self._write_tf_summary(grad, loss)
        gc.collect()
    
    def add_sample(self, env_training, state_action, action, reward, done, new_state, new_demand, new_source, new_destination):
        self.bw_allocated_feature.fill(0.0)
        new_state_copy = np.copy(new_state)

        state_action['graph_id'] = tf.fill([tf.shape(state_action['link_state'])[0]], 0)
    
        # We get the K-paths between new_source-new_destination
        # pathList = env_training.allPaths[str(new_source) +':'+ str(new_destination)]
        pathList = env_training.num_shortest_path(new_source, new_destination, new_demand)

        # print(pathList)

        path = 0
        list_k_features = list()

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths

        # print(len(pathList))

        while path < len(pathList):
            currentPath = pathList[path]
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate new_demand to bw_allocated
            while (j < len(currentPath)):
                new_state_copy[env_training.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = new_demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute it's qvalues using cummax
            features = agent.get_graph_features(env_training, new_state_copy)

            list_k_features.append(features)
            path = path + 1
            new_state_copy[:,1] = 0
        
        vs = [v for v in list_k_features]



        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
                'graph_id': tf.concat([v for v in graph_ids], axis=0),
                'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
                'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
                'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
                'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )    
        
        # We store the state with the action marked, the graph ids, first, second, num_edges, the reward, 
        # new_state(-1 because we don't need it in this case), the graph ids, done, first, second, number of edges
        self.memory.append((state_action['link_state'], state_action['graph_id'], state_action['first'], # 2
                        state_action['second'], tf.convert_to_tensor(state_action['num_edges']), # 4
                        tf.convert_to_tensor(reward, dtype=tf.float32), tensors['link_state'], tensors['graph_id'], # 7
                        tf.convert_to_tensor(int(done==True), dtype=tf.float32), tensors['first'], tensors['second'], # 10 
                        tf.convert_to_tensor(tensors['num_edges']))) # 12




if __name__ == "__main__":

    batch_number = 2
    node_number = 3

    from mpnn import GCN
    from sklearn.metrics import mean_squared_error
    import torch.nn as nn

    model = GCN()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    criterion = torch.nn.MSELoss()


    # Get the environment and extract the number of actions.
    env_training = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env_training.seed(SEED)

    final_linkage = [0] * N
    for i in range(N):
        final_linkage[i] = (i + sat_per_orbit)%N


    print("a:")
    env_training.generate_environment(graph_topology, listofDemands, 0, final_linkage)

    env_eval = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env_eval.seed(SEED)

    print("b:")
    env_eval.generate_environment(graph_topology, listofDemands, 0,  final_linkage)

    batch_size = hparams['batch_size']
    agent = DQNAgent(batch_size)
    high_agent = High_agent(batch_size)


    eval_ep = 0
    train_ep = 0
    max_reward = 0
    reward_id = 0

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    # We store all the information in a Log file and later we parse this file 
    # to extract all the relevant information
    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint = tf.train.Checkpoint(model=agent.primary_network, optimizer=agent.optimizer)

    rewards_test = np.zeros(200)

    demand_per_energy = np.zeros(EVALUATION_EPISODES)
    demand_full_ratio = np.zeros(EVALUATION_EPISODES)
    tot_demand = np.zeros(EVALUATION_EPISODES)
    tot_energy = np.zeros(EVALUATION_EPISODES)


    counter_store_model = 1

    # for ep_it in range(ITERATIONS):
    for ep_it in range(300):
        overall_reward = 0
        overall_switching_reward = 0

        ### 总的train + test --- 1 episode
        if ep_it > epsilon_start_decay and high_agent.epsilon > high_agent.epsilon_min:
            high_agent.epsilon *= high_agent.epsilon_decay
            high_agent.epsilon *= high_agent.epsilon_decay


        if ep_it%1==0:
            print("Training iteration: ", ep_it)

        if ep_it==0:
            # At the beginning we don't have any experiences in the buffer. Thus, we force to
            # perform more training episodes than usually
            train_episodes = FIRST_WORK_TRAIN_EPISODE
        else:
            train_episodes = TRAINING_EPISODES


        # We only evaluate the model every evaluation_interval steps
        if ep_it % evaluation_interval == 0:

            ## 要重复两遍 总train+总test
            final_linkage = [0] * N
            for i in range(N):
                final_linkage[i] = (i + sat_per_orbit) % N


            time_slot = 0
            # for eps in range(EVALUATION_EPISODES):
            max_low_reward_list = []
            sw_switching_list = []
            next_state_max_list = []
            next_next_list = []

            ## 每个 episode 有几个 period
            for eps in range(5):
                fileLogs.write(">>> current period " + str(eps) + ",\n")

                max_low_reward = -float("inf")


                ## inner iteration number
                for inner_iter in range(1):
                # for inner_iter in range(200):

                    if inner_iter > epsilon_start_decay and agent.epsilon > agent.epsilon_min:
                        agent.epsilon *= agent.epsilon_decay
                        agent.epsilon *= agent.epsilon_decay



                    epi_length = 0
                    # if time_slot != 0:
                    env_training.generate_environment(graph_topology, listofDemands, time_slot,final_linkage)
                    state, demand, source, destination = env_training.reset(listofDemands,time_slot)
                    rewardAddTest = 0


                    while 1:
                        # We execute evaluation over current state

                        path_list,action, state_action = agent.act(env_training, state, demand, source, destination, False)

                        env_training.quant_change(time_slot)

                        if inner_iter == 0:
                            action = 0

                        new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action, demand, source, destination, time_slot,
                                                                                                  path_list)

                        agent.add_sample(env_training, state_action, action, reward, done, new_state, new_demand,
                                         new_source, new_destination)

                        state = new_state
                        demand = new_demand
                        source = new_source
                        destination = new_destination
                        if done or epi_length == PERIOD:
                            print("current train eps length: "+str(epi_length))
                            break
                        time_slot += 1
                        epi_length += 1

                    agent.replay(inner_iter)
                    checkpoint.save(checkpoint_prefix)
                    time_slot -= PERIOD


                    ## low-level test starts
                    epi_length = 0

                    # if time_slot != 0:
                    env_eval.generate_environment(graph_topology, listofDemands, time_slot, final_linkage)
                    state, demand, source, destination = env_eval.reset(listofDemands, time_slot)
                    rewardAddTest = 0

                    while 1:
                        # We execute evaluation over current state
                        path_list, action, state_action = agent.act(env_eval, state, demand, source, destination, True)

                        env_eval.quant_change(time_slot)

                        if inner_iter == 0:
                            action = 0

                        new_state, reward, done, new_demand, new_source, new_destination = env_eval.make_step(state,
                                                                                                              action,
                                                                                                              demand,
                                                                                                              source,
                                                                                                              destination,
                                                                                                              time_slot,
                                                                                                              path_list)

                        reward = reward/NORM_REWARD

                        # agent.add_sample(env_eval, state_action, action, reward, done, new_state, new_demand,
                        #                  new_source, new_destination)
                        rewardAddTest = rewardAddTest + reward
                        state = new_state
                        demand = new_demand
                        source = new_source
                        destination = new_destination
                        if done or epi_length == PERIOD:
                            print("current test eps length: " + str(epi_length))
                            break
                        time_slot += 1
                        epi_length += 1

                    time_slot -= PERIOD
                    if rewardAddTest > max_low_reward:
                        max_low_reward = rewardAddTest

                    rewards_test[inner_iter] = rewardAddTest
                    fileLogs.write(">," + str(rewards_test[inner_iter]) + ",\n")
                    fileLogs.flush()

                    agent.epsilon = 1


                max_low_reward_list.append(max_low_reward)
                # overall_reward += max_low_reward
                overall_reward += 60 * max_low_reward
                agent.memory = []
                gc.collect()

                time_slot += PERIOD

                ## 进行到下一个period
                next_state_max, final_linkage, sw_switching, next_next = high_agent.my_forward(time_slot, env_training, False,ep_it)
                overall_switching_reward -= GAMMA * sw_switching
                overall_reward -= GAMMA * sw_switching

                sw_switching_list.append(sw_switching)

                ## !!! EPISODE NUMBER * (9 nodes)
                next_state_max_list.append(next_state_max)
                next_next_list.append(next_next)

                counter_store_model = counter_store_model + 1



            for bb in range(len(max_low_reward_list)-1):
                for agent_id in range(N):
                    high_agent.memory.append((max_low_reward_list[bb+1], sw_switching_list[bb], next_state_max_list[bb][agent_id], next_next_list[bb+1][agent_id]))

            high_agent.replay(ep_it)

            fileLogs.write("episode " + str(ep_it) + str(": ") + ",\n")
            fileLogs.write("overall reward " + str(overall_reward) + ",\n")
            fileLogs.write("overall switching reward " + str(overall_switching_reward) + ",\n")

            gc.collect()

    exit(0)