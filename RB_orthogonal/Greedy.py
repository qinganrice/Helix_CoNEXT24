import numpy as np 
import time
import h5py
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

config = 16 # 16 is small size, 80 is medium size
uncorr = 1 # correlated case and uncorrelated case
loose_sla = 0 # Loose SLA or tight SLA

if config == 80:
    Num_UE = 80 #
    Num_slice = 8

    Num_UE_ps = int(Num_UE/Num_slice)

    if loose_sla == 1:
        SLAs = np.array([135, 120, 130, 140, 45, 110, 50, 125])
    else:
        SLAs = np.array([235, 220, 230, 240, 145, 210, 150, 225])

    SEL_UE = 8
    

    H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')

    
    if uncorr == 0:
        for i in range (Num_slice):
            H[i*Num_UE_ps:(i+1)*Num_UE_ps, :,:,:] = H_all[int(i//2),int(i%2)*Num_UE_ps:(int(i%2)+1)*Num_UE_ps,:,:,:]
    else:
        ue_num_list = [3,3,2,2]
        for i in range (Num_slice):
            H[i*Num_UE_ps:i*Num_UE_ps+ue_num_list[0], :,:,:] = H_all[0,i*ue_num_list[0]:(i+1)*ue_num_list[0],:,:,:]
            H[i*Num_UE_ps+ue_num_list[0]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1], :,:,:] = H_all[1,i*ue_num_list[1]:(i+1)*ue_num_list[1],:,:,:]
            H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2], :,:,:] = H_all[2,i*ue_num_list[2]:(i+1)*ue_num_list[2],:,:,:]
            H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]+ue_num_list[3], :,:,:] = H_all[3,i*ue_num_list[3]:(i+1)*ue_num_list[3],:,:,:]

elif config == 16:
    Num_UE = 16 #
    Num_slice = 4

    Num_UE_ps = int(Num_UE/Num_slice)

    if loose_sla == 1:
        SLAs = np.array([135, 120, 130, 140])
    else:
        SLAs = np.array([235, 220, 230, 240])

    SEL_UE = 8
    
    H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')
    
    if uncorr == 0:
        for i in range (Num_slice):
            H[i*Num_UE_ps:(i+1)*Num_UE_ps, :,:,:] = H_all[i,0:Num_UE_ps,:,:,:]
    else:
        for i in range (Num_slice):
            H[i*Num_UE_ps, :,:,:] = H_all[0,i,:,:,:]
            H[i*Num_UE_ps+1, :,:,:] = H_all[1,i,:,:,:]
            H[i*Num_UE_ps+2, :,:,:] = H_all[2,i,:,:,:]
            H[i*Num_UE_ps+3, :,:,:] = H_all[3,i,:,:,:]


def sel_ue(user_set, action):
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

# Sort algorithm
def sort(matrix):

    flattened_with_indices = [(value, (i, j)) for i, row in enumerate(matrix) for j, value in enumerate(row)]

    # Sort the flattened list in descending order based on values
    sorted_indices = sorted(flattened_with_indices, key=lambda x: x[0], reverse=True)

    # Extract the 2-D indices from the sorted list
    sorted_indices_2d = [index[1] for index in sorted_indices]

    return sorted_indices_2d

total_time = 0
total_inter = 0

for tti in range (0, 10):
    print("tti",tti)
    
    start = time.time()

    rbgs = [[],[],[],[],[],[],[],[]]

    delta_data = np.zeros((Num_slice,))
    
    for rbg in range (0, Num_RBG):
        H_t = H[:,:, tti, rbg]
        for sl in range (0, Num_slice):
            action_rbg_slice[tti,rbg,sl], cap_rbg_slice_ue[tti,rbg,sl,:], cap_rbg_slice[tti,rbg,sl] = choose_action(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps], H_t, SEL_UE, Num_UE)
    
    # start = time.time()
    # sorted_indices_2d = sort(cap_rbg_slice[tti,:,:].squeeze())
    cap_rb_sl = cap_rbg_slice[tti,:,:]
    inter_slice = time.time()
    # Inter-Slice Scheduler
    sl_list = [num for num in range (Num_slice)]
    rbg_list = [num for num in range (Num_RBG)]
    while (len(rbg_list)) and (len(sl_list)):
        # print(cap_rb_sl.shape)
        rbg_sl = np.unravel_index(cap_rb_sl.argmax(), cap_rb_sl.shape)
        rbg = rbg_list[rbg_sl[0]]
        sl = sl_list[rbg_sl[1]]
        # print(rbg_list, sl_list, delta_data)
        if (delta_data[sl] - SLAs[sl] < 0):
            rbgs[sl].append(rbg)
            delta_data[sl] += cap_rb_sl[rbg_sl[0],rbg_sl[1]]
            total_slice_tp[sl] += cap_rb_sl[rbg_sl[0],rbg_sl[1]]
            cap_rb_sl= np.delete(cap_rb_sl, rbg_sl[0], axis=0)
            rbg_list.remove(rbg_list[rbg_sl[0]])
        else:
            cap_rb_sl= np.delete(cap_rb_sl, rbg_sl[1], axis=1)
            sl_list.remove(sl_list[rbg_sl[1]])

        ue_select, idx = sel_ue(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps], action_rbg_slice[tti,rbg,sl])
        for count, ue in enumerate(ue_select):
            rt_history[ue] += cap_rbg_slice_ue[tti,rbg,sl,count]

    end = time.time()

    total_time += end - start
    total_inter += end - inter_slice
    # print("Running time is:", end - start, '\n')

    # assigned_RBG += len(rbgs[0]) + len(rbgs[1]) + len(rbgs[2]) + len(rbgs[3])
    assigned_RBG += len(rbgs[0]) + len(rbgs[1]) + len(rbgs[2]) + len(rbgs[3]) + len(rbgs[4]) + len(rbgs[5]) + len(rbgs[6]) + len(rbgs[7])

# jfi = np.zeros((Num_slice,))
# for i in range (Num_slice):
#     jfi[i] = np.square((np.sum(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps]))) / (Num_UE_ps * np.sum(np.square(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps])))
# print('jfi is', jfi)

avg_slice_tp = total_slice_tp / total_tti
print("Average TP is:", avg_slice_tp)
print("Total Assgined RBG:", assigned_RBG/total_tti)
# print("Total time:", total_time/total_tti)
# print("Total inter-slice time:", total_inter/total_tti)

        
        
        
        
        
        
        
        
        
        
        
        
