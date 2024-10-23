import numpy as np 
import time
import h5py
from prop_fairness import * 

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
        H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]+ue_num_list[3], :,:,:] = H_all[5,i*ue_num_list[3]:(i+1)*ue_num_list[3],:,:,:] * 0.1



# Sort algorithm
def sort(matrix):

    flattened_with_indices = [(value, (i, j)) for i, row in enumerate(matrix) for j, value in enumerate(row)]

    # Sort the flattened list in descending order based on values
    sorted_indices = sorted(flattened_with_indices, key=lambda x: x[0], reverse=True)

    # Extract the 2-D indices from the sorted list
    sorted_indices_2d = [index[1] for index in sorted_indices]

    return sorted_indices_2d

def sel_ue(user_set, action):
    sum_before = 0
    for i in range (1,(P_N_UE+1)):
        sum_before += len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= len(list(combinations(user_set, i)))
            ue_select = list(combinations(user_set, i))[action-sum_before]
            break
    return ue_select,idx

rt_history = np.ones((Num_UE,)) * 0.01

assigned_RBG = 0

cap_rbg_slice = np.zeros((Num_tti, Num_RBG, Num_slice))
action_rbg_slice = np.zeros((Num_tti, Num_RBG, Num_slice), dtype=int)
cap_rbg_slice_ue = np.zeros((Num_tti, Num_RBG, Num_slice, Num_UE_ps))

total_slice_tp = np.zeros((Num_slice,))
UE_list = [num for num in range(Num_UE)]


for tti in range (0, 10):
    print("tti",tti)
    rbgs = [[],[],[],[],[],[],[],[]]

    achieved_data = np.zeros((Num_slice,))

    
    for rbg in range (0, Num_RBG):
        H_t = H[:,:, tti, rbg]
        for sl in range (0, Num_slice):
            action_rbg_slice[tti,rbg,sl], cap_rbg_slice[tti,rbg,sl], cap_rbg_slice_ue[tti,rbg,sl,:] = choose_action(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps], H_t, rt_history)

    start = time.time()
    sorted_indices_2d = sort(cap_rbg_slice[tti,:,:].squeeze())

    
    # Inter-Slice Scheduler
    while len(sorted_indices_2d) > 0:
        rbg_sl = sorted_indices_2d[0]
        rbg = rbg_sl[0]
        sl = rbg_sl[1]
        if achieved_data[sl] < SLAs[sl]:
            rbgs[sl].append(rbg)
            achieved_data[sl] += cap_rbg_slice[tti,rbg,sl]
            total_slice_tp[sl] += cap_rbg_slice[tti,rbg,sl]

            # update history
            ue_select, idx = sel_ue(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps], action_rbg_slice[tti,rbg,sl])
            for count, ue in enumerate(ue_select):
                rt_history[ue] += cap_rbg_slice_ue[tti,rbg,sl,count]


            sorted_indices_2d = [(i, j) for i, j in sorted_indices_2d if i != rbg]
        
        else:
            sorted_indices_2d = [(i, j) for i, j in sorted_indices_2d if j != sl]
    
    end = time.time()
    # print("Running time is:", end - start, '\n')

    # assigned_RBG += len(rbgs[0]) + len(rbgs[1]) + len(rbgs[2]) + len(rbgs[3])
    assigned_RBG += len(rbgs[0]) + len(rbgs[1]) + len(rbgs[2]) + len(rbgs[3]) + len(rbgs[4]) + len(rbgs[5]) + len(rbgs[6]) + len(rbgs[7])

jfi = np.zeros((Num_slice,))
for i in range (Num_slice):
    jfi[i] = np.square((np.sum(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps]))) / (Num_UE_ps * np.sum(np.square(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps])))
print('avg jfi is', np.mean(jfi))

avg_slice_tp = total_slice_tp / total_tti
print("Average TP is:", avg_slice_tp)
print("Total Assgined RBG:", assigned_RBG/total_tti)

        
        
        
        
        
        
        
        
        
        
        
        
