import numpy as np 
import time
import h5py
from prop_fairness import *
from max_rate import * 


channel_file = h5py.File('DU_8_26_64_100_52.hdf5','r')
H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_all = np.array(H_r + 1j*H_i)

# Define Configurations
Num_tti = 100
Num_RBG = 52
Num_BS = 64
corr_th = 0.5
total_tti = 10

config = 80

if config == 80:
    Num_UE = 80 #
    Num_slice = 8

    Num_UE_ps = int(Num_UE/Num_slice)

    # SLAs = np.array([135, 120, 130, 140, 45, 110, 50, 125])
    SLAs = np.array([235, 220, 230, 240, 145, 210, 150, 225])

    SEL_UE = 16
    

    H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')

    ue_num_list = [3,3,2,2]
    for i in range (Num_slice):
        H[i*Num_UE_ps:i*Num_UE_ps+ue_num_list[0], :,:,:] = H_all[0,i*ue_num_list[0]:(i+1)*ue_num_list[0],:,:,:]
        H[i*Num_UE_ps+ue_num_list[0]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1], :,:,:] = H_all[1,i*ue_num_list[1]:(i+1)*ue_num_list[1],:,:,:]
        H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2], :,:,:] = H_all[4,i*ue_num_list[2]:(i+1)*ue_num_list[2],:,:,:] *0.1
        H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]+ue_num_list[3], :,:,:] = H_all[5,i*ue_num_list[3]:(i+1)*ue_num_list[3],:,:,:] *0.1

def sort_vector(vector):
    sorted_vector = sorted(vector, reverse=True)
    positions = [i for i in range(len(vector))]
    positions_sorted = [pos for _, pos in sorted(zip(vector, positions), reverse=True)]
    new_matrix = np.array([sorted_vector, positions_sorted])
    return new_matrix

def sel_ue(user_set, action, SEL_UE):
    sum_before = 0
    for i in range (1,(SEL_UE+1)):
        sum_before += len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= len(list(combinations(user_set, i)))
            ue_select = list(combinations(user_set, i))[action-sum_before]
            break
    return ue_select,idx

assigned_RBG = 0
rt_history = np.ones((Num_UE,)) * 0.01

cap_rbg_slice = np.zeros((Num_tti, Num_RBG, Num_slice))
cap_rbg_slice_ue = np.zeros((Num_tti, Num_RBG, Num_slice, Num_UE))
action_rbg_slice = np.zeros((Num_tti, Num_RBG, Num_slice), dtype=int)


total_slice_tp = np.zeros((Num_slice,))
UE_list = [num for num in range(Num_UE)]

total_time = 0
total_inter = 0

for tti in range (0, 10):
    print("tti",tti)
    start = time.time()
    rbgs = [[],[],[],[],[],[],[],[]]

    delta_data = SLAs * (tti+1) - total_slice_tp

    for rbg in range (0, Num_RBG):
        H_t = H[:,:, tti, rbg]
        # print("H shape:", H_t.shape)
        for sl in range (0, Num_slice):
            action_rbg_slice[tti,rbg,sl], cap_rbg_slice_ue[tti,rbg,sl,:], cap_rbg_slice[tti,rbg,sl] = choose_action(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps], H_t, SEL_UE, Num_UE)

    cap_rb_sl = cap_rbg_slice[tti,:,:]
    inter_slice = time.time()
    while any(element > 0 for element in delta_data):
        
        order = np.argmax(delta_data)
        rbg_num = np.argmax(cap_rb_sl[:,order])
        rbgs[order].append(rbg_num)

        delta_data[order] -= cap_rb_sl[rbg_num,order]
        total_slice_tp[order] += cap_rb_sl[rbg_num,order]

        cap_rb_sl= np.delete(cap_rb_sl, rbg_num, axis=0)

        ue_select, idx = sel_ue(UE_list[order*Num_UE_ps:(order+1)*Num_UE_ps], action_rbg_slice[tti,rbg_num,order],SEL_UE)
        for count, ue in enumerate(ue_select):
            rt_history[ue] += cap_rbg_slice_ue[tti,rbg_num,order,count]
            

    end = time.time()
    total_time += end - start
    total_inter += end - inter_slice

    assigned_RBG += len(rbgs[0]) + len(rbgs[1]) + len(rbgs[2]) + len(rbgs[3]) + len(rbgs[4]) + len(rbgs[5]) + len(rbgs[6]) + len(rbgs[7])

jfi = np.zeros((Num_slice,))
for i in range (Num_slice):
    jfi[i] = np.square((np.sum(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps]))) / (Num_UE_ps * np.sum(np.square(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps])))
print('avg jfi is', np.mean(jfi))
avg_slice_tp = total_slice_tp / total_tti
print("Average TP is:", avg_slice_tp)
print("Total Assgined RBG:", assigned_RBG/total_tti)


        
        
        
        
        
        
        
        
        
        
        
