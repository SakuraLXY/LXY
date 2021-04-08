#修改了细胞类型
#from pyNN.nest import *
#import pyNN.nest as sim
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
import gzip

# specify the location of the MNIST data
MNIST_data_path = ''
sim, options = get_simulator()

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
#读取数据
def get_labeled_data(picklename):
    # Read input-vector (image) and target class (label, 0-9) and return
    # it as list of tuples.
    
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('{}.pickle'.format(picklename),'rb'))
    else:
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

#读取权重
def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr

#保存连接
def save_connections(ending = ''):
    print('save connections')
    for connName in save_conns: #save_conns = 'XeAe'
        conn = connections[connName]
        # connListSparse = zip(conn.i, conn.j, conn.w)####
        connListSparse = conn.get('weight', format='array')
        np.save(data_path + connName + ending, connListSparse)
              

#存储theta值
def save_theta(ending = ''):
    print('save theta')
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

#权重矩阵重排列
def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e)) #784*100
    n_e_sqrt = int(np.sqrt(n_e)) #n_e的平方根
    n_in_sqrt = int(np.sqrt(n_input)) #n_input的平方根
    num_values_col = n_e_sqrt*n_in_sqrt #平方根相乘
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row)) #重排列的权重矩阵
    connMatrix = np.zeros((n_input, n_e)) #连接矩阵
    connMatrix = connections[name].get('weight', format='array')
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    # #test
    # print(weight_matrix)
    # print(rearranged_weights)
    return rearranged_weights

#画出输入权重的图##有点复杂
# def plot_2d_input_weights():
#     name = 'XeAe'
#     weights = get_2d_input_weights()
#     plt.figure(fig_num, figsize=(18, 18))
#     im2 = 
#     return im2, fig

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

np.random.seed(0) #使得后续生产的随机数可预测
data_path = './'

weight_path = data_path + 'random/'
num_examples = 1000 #使用训练例子的数量

ending = ''
n_input = 784 #输入层，即28*28
n_e = 100 #兴奋层
n_i = n_e #抑制层

#运行时间
single_example_time = 350 #ms
resting_time = 150
runtime = num_examples * (single_example_time + resting_time)
#时间间隔，之后改
# if num_examples <= 10000:
#     update_interval = num_examples
#     weight_update_interval = 20
# else:
#     update_interval = 10000
#     weight_update_interval = 100
# if num_examples <= 60000:
#     save_connections_interval = 10000
# else:
#     save_connections_interval = 10000
#     update_interval = 10000
weight_update_interval = 20
save_connections_interval = 1000
update_interval = 200

# e_params = {'v_rest'     : -65.0,
#             'tau_refrac' : 5.0,      #不应期
#             'v_thresh'   : -52.0, 
#             'v_reset'    : -65.0, 
#             'tau_syn_E'  : 1.0,
#             'tau_syn_I'  : 2.0,
#             'e_rev_I'    : -100.0,
#             'a'          : 0.001e3,
#             'b'          :0.005,
# }

# i_params = {'v_rest'     : -60.0,
#             'tau_refrac' : 2.0, 
#             'v_thresh'   : -40.0, 
#             'v_reset'    : -45.0, 
#             'tau_syn_E'  : 1.0,
#             'tau_syn_I'  : 2.0,
#             'e_rev_I'    : -85.0,
#             'a'          : 0.001e3,
#             'b'          :0.005,
# }

e_params = {'v_rest'     : -65.0,
            'tau_refrac' : 5.0,      #不应期
            'v_thresh'   : -52.0, 
            'v_reset'    : -65.0, 
            'e_rev_I'    : -100.0,

}

i_params = {'v_rest'     : -60.0,
            'tau_refrac' : 2.0, 
            'v_thresh'   : -40.0, 
            'v_reset'    : -45.0, 
            'e_rev_I'    : -85.0,
}
V_INIT_E = -25.0
V_INIT_I = -20.0

 #STDP，需要修改
#使用STDP学习从输入神经元到兴奋性神经元的所有突触
stdp = sim.STDPMechanism(
            weight=0.1,  # this is the initial value of the weight
            delay="0.2 + 0.01*d",
            timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                            A_plus=0.01, A_minus=0.012),
            weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=1.0))

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
input_intensity = 2.
start_input_intensity = input_intensity

fig_num = 1 #图片的序号
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))



print('1')

#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names):
    print('create neuron group', name)

    neuron_groups[name+'e'] = sim.Population(n_e, sim.IF_cond_exp, cellparams=e_params, initial_values={'v': V_INIT_E}, label = 'Ae')
    neuron_groups[name+'i'] = sim.Population(n_i, sim.IF_cond_exp, cellparams=i_params, initial_values={'v': V_INIT_I}, label = 'Ai')

    #这里缺失给神经元设立theta值

    print('create recurrent connections')
    #connector_AeAi = get_matrix_from_file(weight_path + '../random/' + 'AeAi' + ending + '.npy')
    #connector_AiAe = get_matrix_from_file(weight_path + '../random/' + 'AiAe' + ending + '.npy')

    connect_AiAe = 17*(np.ones([n_e,n_e]) - np.identity(n_e))
    #connect_AiAe.dtype = bool
    #print(connect_AiAe)

    connections['AeAi'] = sim.Projection(neuron_groups[name+'e'], neuron_groups[name+'i'],
                                    sim.OneToOneConnector(), synapse_type = sim.StaticSynapse(weight=10.4, delay=0.5))
    connections['AiAe'] = sim.Projection(neuron_groups[name+'i'], neuron_groups[name+'e'],
                                    sim.AllToAllConnector(allow_self_connections=False), synapse_type = sim.StaticSynapse(weight=connect_AiAe, delay=0.5))

    #connections['AeAi'].set(weight = 10.4)
    #connections['AiAe'].set(weight = connect_AiAe)
    #print(connections['AiAe'].get('weight',format = 'array'))
    print('create monitors for', name)
    #峰值计数 'Ae' & 'Ai'
    neuron_groups['Ae'].record('spikes')
    neuron_groups['Ai'].record('spikes')
    

  
#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
pop_values = [0,0,0]


for i,name in enumerate(input_population_names):#['X']
    #参数可调整
    input_groups[name+'e'] = sim.Population(n_input, sim.SpikeSourcePoisson, 
                                            cellparams={'start':0.0, 'rate':0., 'duration':1000.0}, label = 'Xe')

for name in input_connection_names:
    print ('create connections between', name[0], 'and', name[1])

    connections['XeAe'] = sim.Projection(input_groups['Xe'], neuron_groups['Ae'], sim.AllToAllConnector(allow_self_connections=False), stdp)

    #随机初始权重和延迟
    #matrix_XeAe = np.load('random/../random1/XeAe.npy')
    #print('loading', matrix_XeAe.shape)
    #connections['XeAe'].set(weight = RandomDistribution('uniform', (0, 0.3)))
    #connections['XeAe'].set(delay = RandomDistribution('uniform', (1, 10)))


#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

#previous_spike_count = np.zeros(n_e)
previous_spike_count = 0
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

sim.run(0)
j = 0
while j < (int(num_examples)):
#while j < 10:
    ##这里有一行把权重正则化
    print(j)
    spike_rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity

    input_groups['Xe'].rate = spike_rates ##输入神经元的激发率
    sim.run(single_example_time) ##运行
    
    #更新assignment
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    # if j % weight_update_interval == 0 ## 20
        #更新2d权重图
    #保存结果
    if j % save_connections_interval == 0 and j > 0:
        save_connections(str(j))
        ##save_theta(str(j)) ##


    #当前峰值计数,如果峰值小于5，增大强度， 把rate设为0，然后重新展示图片
    spike_counters['Ae'] = neuron_groups['Ae'].get_data().segments[0].spiketrains
    count = sum(len(a) for a in spike_counters['Ae'])#len(a)的意思是在运行时间内发生了多少次峰值
    current_spike_count =  count - previous_spike_count
    previous_spike_count = count
    print(count)
    if current_spike_count < 5:
        input_intensity += 1
        for i,name in enumerate(input_population_names):#'X'
            input_groups['Xe'].rate = 0 ##
        sim.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        input_numbers[j] = training['y'][j%60000][0] ###
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 100 == 0 and j > 0:#每完成训练100个给出提示
            print ('runs done:', j, 'of', int(num_examples))
        # if j % update_interval == 0 and j > 0:
        #     if do_plot_performance:
        #         unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
        #         print ('Classification performance', performance[:(j/float(update_interval))+1])
        for i,name in enumerate(input_population_names):#'X'
            input_groups['Xe'].rate = 0 ##
            input_intensity = start_input_intensity#重置强度
            j += 1
   

#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print ('save results')

#save_theta()
save_connections()

print(outputNumbers)
