##train2.py改版
####改动
####运行在HBP平台上
##theta值####
##如何控制spikes数量
##论文原文：由于每个神经元只对输入数字的一个非常小的子集作出响应，因此响应非常稀疏，
##每个例子只触发很少的尖峰。即使在最大的有6400个兴奋性神经元的神经网络中，只有≈17个尖峰在响应一位数的表现时被激发。
#细胞类型
##delay改为random
###input_groups_Xe
##矩阵重排列貌似只是为了画图
##加了正则化 


##label 分类 还没解决

import pyNN.spiNNaker as sim
from pyNN.utility import get_simulator, init_logging, normalized_filename
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import time
import os.path
import scipy
import pickle
from struct import unpack
from pyNN.random import RandomDistribution
from pyNN.parameters import Sequence
from pyNN.utility.plotting import DataTable
from quantities import Hz, s, ms, mV
from struct import unpack
import copy
import math


# specify the location of the MNIST data
MNIST_data_path = ''
#sim, options = get_simulator()

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
#读取数据
def get_labeled_data(picklename):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('{}.pickle'.format(picklename),'rb'))
    else:
        # Open the images with gzip in read binary mode
        images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
        labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number #跳过了MNIST数据集的魔法数字的四个字节
        number_of_images = unpack('>I', images.read(4))[0] #num = 60000
        rows = unpack('>I', images.read(4))[0] #rows = 28
        cols = unpack('>I', images.read(4))[0] #cols = 28
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0] #N = 60000

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array

        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i) #提示数据读了多少

            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))

    return data


#保存连接
def save_connections(ending = ''):
    print('save connections')
    conn = connections_XeAe
    # connListSparse = zip(conn.i, conn.j, conn.w)####
    connListSparse = conn.get('weight', format='array')
    np.save('XeAe', connListSparse)

def save_theta(ending = ''):
    print('save theta')
    np.save('theta', theta)
            

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

def normalize_weights():
    len_source = n_input
    len_target = n_e
    # connection = np.zeros((len_source, len_target))
    # print(connection)
    temp_conn = np.copy(connections_XeAe.get('weight', format='array'))
    #print(temp_conn)
    colSums = np.sum(temp_conn, axis = 0) ##axis=1表示按行相加 , axis=0表示按列相加
    print(colSums)
    colFactors = weight['ee_input']/colSums
    print(colFactors)
    for j in range(n_e):#
        temp_conn[:,j] *= colFactors[j]
    connections_XeAe.set(weight = temp_conn)
    #print(connections_XeAe.get('weight', format='array'))

#theta
def thetaFunction(theta,spikes,sim_time,theta_plus_e=0.001,tc_theta=1e7):
    ret_theta=[]
    t=copy.deepcopy(theta)
    for j in range(len(spikes)):
        t[j]+=theta_plus_e*spikes[j] # 然后，这个位置的theta += theta_plus_e*n
        t[j]=t[j]*math.exp(-sim_time/tc_theta) #theta随着时间衰减
        # ret_theta.append(t)
        # t=copy.deepcopy(t)
    ret_theta = copy.deepcopy(t)
    return ret_theta

sim.setup()

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print('time needed to load training set:', end - start)

#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------

#np.random.seed(0) #使得后续生产的随机数可预测
data_path = './'

weight_path = data_path + 'random/'
num_examples = 200 #使用训练例子的数量

ending = ''
n_input = 784 #输入层，即28*28
n_e = 100 #兴奋层
n_i = n_e #抑制层

#运行时间
single_example_time = 350 #ms
resting_time = 150
theta_time = single_example_time + resting_time
runtime = num_examples * (single_example_time + resting_time)

weight_update_interval = 20
save_connections_interval = 100
update_interval = 50

#细胞参数

e_params = {
    'v_rest':   -65.0,  # Resting membrane potential in mV.
    'cm':         1.0,  # Capacity of the membrane in nF
    'tau_m':     100.0,  # Membrane time constant in ms. ##这里是100，降低了噪声的影响
    'tau_refrac': 5.0,  # Duration of refractory period in ms.
    'tau_syn_E':  1.0,  # Decay time of the excitatory synaptic conductance in ms.
    'tau_syn_I':  2.0,  # Decay time of the inhibitory synaptic conductance in ms.
    'e_rev_E':    0.0,  # Reversal potential for excitatory input in mV
    'e_rev_I':  -100.0,  # Reversal potential for inhibitory input in mV
    'v_thresh': -52.0,  # Spike threshold in mV.
    'v_reset':  -65.0,  # Reset potential after a spike in mV.
    'i_offset':   0.0,  # Offset current in nA
}

i_params = {
    'v_rest':   -60.0,  # Resting membrane potential in mV.
    'cm':         1.0,  # Capacity of the membrane in nF
    'tau_m':     10.0,  # Membrane time constant in ms.
    'tau_refrac': 2.0,  # Duration of refractory period in ms.
    'tau_syn_E':  1.0,  # Decay time of the excitatory synaptic conductance in ms.
    'tau_syn_I':  2.0,  # Decay time of the inhibitory synaptic conductance in ms.
    'e_rev_E':    0.0,  # Reversal potential for excitatory input in mV
    'e_rev_I':  -85.0,  # Reversal potential for inhibitory input in mV
    'v_thresh': -40.0,  # Spike threshold in mV.
    'v_reset':  -45.0,  # Reset potential after a spike in mV.
    'i_offset':   0.0,  # Offset current in nA
}

V_INIT_E = -25.0
V_INIT_I = -20.0

weightMatrix = np.random.random((n_input, n_e)) + 0.01
weightMatrix *= 0.3
#weight_XeAe = [(i, j, weightMatrix[i,j]) for j in range(n_e) for i in range(n_input)]
weight_XeAe = weightMatrix
print(weightMatrix)



#使用STDP学习从输入神经元到兴奋性神经元的所有突触
timing_rule = sim.SpikePairRule(tau_plus=8.0,tau_minus=2.0, #8,1
                             A_plus=0.0625,A_minus=0.0625) # 80,20
weight_rule = sim.AdditiveWeightDependence(w_max=1.0,w_min=0)
stdp = sim.STDPMechanism(timing_dependence=timing_rule,
                            weight_dependence=weight_rule,
                            #weight=RandomDistribution(distribution='normal_clipped',low=0,high=1, mu=0.5, sigma=0.3),
                            weight=weight_XeAe,
                            delay=RandomDistribution(distribution='uniform',low=1,high=10)
                            )


input_intensity = 2.
start_input_intensity = input_intensity

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A'] ###
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0,10)
delay['ei_input'] = (0,5)


fig_num = 1 #图片的序号
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------

print('create neuron group A')

neuron_groups_Ae = sim.Population(n_e, 
                                sim.IF_cond_exp, 
                                cellparams=e_params, 
                                initial_values={'v': V_INIT_E}, 
                                label = 'Ae'
                                )
neuron_groups_Ai = sim.Population(n_i, 
                                sim.IF_cond_exp, 
                                cellparams=i_params, 
                                initial_values={'v': V_INIT_I}, 
                                label = 'Ai'
                                )



print('create recurrent connections')

#Ai -> Ae 的连接
#connect_AiAe = 17.0*(np.ones([n_e,n_e]) - np.identity(n_e))
connect_AiAe = []
for i in range(n_e):
    for j in range(n_i):
        if not i==j:
            connect_AiAe.append((i,j))


connections_AeAi = sim.Projection(neuron_groups_Ae, 
                                neuron_groups_Ai,
                                sim.OneToOneConnector(), 
                                synapse_type = sim.StaticSynapse(weight=10.4, delay=1.0),
                                receptor_type = 'excitatory'
                                )
connections_AiAe = sim.Projection(neuron_groups_Ai, 
                                neuron_groups_Ae,
                                connector = sim.FromListConnector(connect_AiAe),
                                synapse_type = sim.StaticSynapse(weight=17, delay=1.0),
                                receptor_type = 'inhibitory'
                                )

print('create monitors for A')
#峰值计数 'Ae' & 'Ai'
neuron_groups_Ae.record("spikes")
neuron_groups_Ai.record("spikes")

#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------


input_groups_Xe = sim.Population(n_input, 
                                sim.SpikeSourcePoisson(rate = 0), 
                                label = 'Xe')


print ('create connections between X and A ')

connections_XeAe = sim.Projection(input_groups_Xe, 
                                neuron_groups_Ae,
                                sim.AllToAllConnector(allow_self_connections=False), 
                                stdp,
                                receptor_type = 'excitatory'
                                )

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

#connections_XeAe.set(weight = weight_XeAe)



#previous_spike_count = np.zeros(n_e)
previous_spike_count = 0
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

#保存初始权重
##正则化
# sim.run(50)
# normalize_weights()
# initWeight = connections_XeAe.get('weight', format='array')
# np.save('initWeight', initWeight)

sim.run(0)
print(connections_XeAe.get('weight',format = 'array'))
j = 0
# print(neuron_groups_Ae)
# print(neuron_groups_Ai)
# print(input_groups_Xe)

####theta
theta = np.zeros(n_e)
print(theta.shape)

old_list = []
while j < (int(num_examples)):
    print(j)
    normalize_weights()
    spike_rates = training['x'][j%60000,:,:].reshape((n_input)) /64. *  input_intensity

    input_groups_Xe.set(rate = spike_rates) ##输入神经元的激发率
    #input_groups_Xe.set(rate = 0)
    #print(input_groups_Xe.get('rate'))

    sim.run(single_example_time) ##运行              

    #更新assignment
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
        print(outputNumbers)
        np.save('initWeight', outputNumbers)
    # if j % weight_update_interval == 0 ## 20
        #更新2d权重图
    #保存结果
    if j % save_connections_interval == 0 and j > 0:
        save_connections(str(j))
        save_theta(str(j)) ##

    list1 = []
    list2 = []
    spike_counters['Ae'] = neuron_groups_Ae.get_data().segments[0].spiketrains
    count = sum(len(a) for a in spike_counters['Ae'])
    current_spike_count =  count - previous_spike_count
    previous_spike_count = count

    for a in spike_counters['Ae']:
        list1.append(len(a))
    
    if j!=0:
        list2=[list1[i] - old_list[i] for i in range(0,len(list1))]
        result_monitor[j%update_interval,:] = list2
        spikes = list2
        print(list2)
    else:
        result_monitor[j%update_interval,:] = list1
        spikes = list1
        print(list1)

    old_list=list1

    # if current_spike_count%100 != 0:
    #     print(list2)
    #     print(current_spike_count)

    print(current_spike_count)

    ###每一个细胞发生峰值的次数是一样的，并且很多，它们不该（？）是一样的
    
    result_monitor[j%update_interval,:] = current_spike_count
    input_numbers[j] = training['y'][j%60000][0] ###
    outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
    if j % 100 == 0 and j > 0:#每完成训练100个给出提示
        print ('runs done:', j, 'of', int(num_examples))

    
    #####假设峰值不到5，重置强度
    input_groups_Xe.set(rate = 0) ##
    sim.run(resting_time)
    input_intensity = start_input_intensity#重置强度
    #print(input_groups_Xe.get('rate'))


    print(connections_XeAe.get('weight', format='array'))

    # print('a')
    # print(neuron_groups_Ae.get('a'))
    # print('b')
    # print(neuron_groups_Ae.get('b'))


    #####更新阈值
    print('update v_thresh')
    # if j != 0:
    #     initV = neuron_groups_Ae.get('v_thresh')
    # else:
    #     initV = np.zeros(n_e) + neuron_groups_Ae.get('v_thresh')
    initV = np.zeros(n_e) - 52
    theta = thetaFunction(theta,spikes,theta_time)
    print(theta)
    v_change = [initV[i] - theta[i] for i in range(n_e)]

    neuron_groups_Ae.set(v_thresh = v_change)
    print(neuron_groups_Ae.get('v_thresh'))


    j += 1

    
    

    

#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------

print ('save results')

save_theta()
save_connections()

print(connections_XeAe.get('weight', format='array'))

print(outputNumbers)
