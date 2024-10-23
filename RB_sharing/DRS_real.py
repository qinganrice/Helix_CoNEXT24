import numpy as np 
import time
import h5py

# Define Configurations
Num_tti = 100
Num_RBG = 52
Num_BS = 64
corr_th = 0.5
total_tti = 10

config = 200 # Real-world size 
uncorr = 0 # correlated case and uncorrelated case (Mobility case set to 0)
loose_sla = 0 # Loose SLA or tight SLA
static = 2 # Static (=1) or mobility case (slow = 0, fast = 2)

if static == 0:
    channel_file = h5py.File('DU_8_25_64_100_52_slow_mob.hdf5','r')
    print("Slow Mobility")
elif static == 1:
    channel_file = h5py.File('DU_8_26_64_100_52.hdf5','r')
    print("Static Case")
elif static == 2:
    channel_file = h5py.File('DU_8_25_64_100_52_fast_mob.hdf5','r')
    print("Fast Mobility")


H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_all = np.array(H_r + 1j*H_i)

if config == 200:
    Num_UE = 200 #
    Num_slice = 8

    Num_UE_ps = [10, 12, 18, 20, 25, 33, 45, 37] # average 25

    if loose_sla == 1:
        SLAs = np.array([135, 120, 130, 140, 45, 110, 50, 125])
    else:
        SLAs = np.array([235, 220, 230, 240, 145, 210, 150, 225])

    SEL_UE = 16
    

    H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')

    #---------------------------- 8 slices 25 UEs Cluster Deployment ----------------------------
    if uncorr ==0:
        H[0:10, :,:,:] = H_all[0,0:10,:,:,:]

        H[10:22, :,:,:] = H_all[0,10:22,:,:,:]

        H[22:25, :,:,:] = H_all[0,22:25,:,:,:]
        H[25:40, :,:,:] = H_all[1,0:15,:,:,:]

        H[40:50, :,:,:] = H_all[1,15:25,:,:,:]
        H[50:60, :,:,:] = H_all[2,0:10,:,:,:]

        H[60:75, :,:,:] = H_all[2,10:25,:,:,:]
        H[75:85, :,:,:] = H_all[3,0:10,:,:,:]

        H[85:100, :,:,:] = H_all[3,10:25,:,:,:]
        H[100:118, :,:,:] = H_all[4,0:18,:,:,:]

        H[118:125, :,:,:] = H_all[4,18:25,:,:,:]
        H[125:150, :,:,:] = H_all[5,0:25,:,:,:]
        H[150:163, :,:,:] = H_all[6,0:13,:,:,:]

        H[163:175, :,:,:] = H_all[6,13:25,:,:,:]
        H[175:200, :,:,:] = H_all[7,0:25,:,:,:]

    else:
    # [10, 12, 18, 20, 25, 33, 45, 37]
        H[0:2, :,:,:] = H_all[0,0:2,:,:,:]
        H[2:4, :,:,:] = H_all[1,0:2,:,:,:]
        H[4:6, :,:,:] = H_all[2,0:2,:,:,:]
        H[6:8, :,:,:] = H_all[3,0:2,:,:,:]
        H[8:10, :,:,:] = H_all[4,0:2,:,:,:]

        H[10:12, :,:,:] = H_all[5,0:2,:,:,:]
        H[12:14, :,:,:] = H_all[6,0:2,:,:,:]
        H[14:16, :,:,:] = H_all[7,0:2,:,:,:]
        H[16:18, :,:,:] = H_all[0,2:4,:,:,:]
        H[18:20, :,:,:] = H_all[1,2:4,:,:,:]
        H[20:22, :,:,:] = H_all[2,2:4,:,:,:]

        H[22:24, :,:,:] = H_all[3,2:4,:,:,:]
        H[24:26, :,:,:] = H_all[4,2:4,:,:,:]
        H[26:28, :,:,:] = H_all[5,2:4,:,:,:]
        H[28:30, :,:,:] = H_all[6,2:4,:,:,:]
        H[30:32, :,:,:] = H_all[7,2:4,:,:,:]
        H[32:34, :,:,:] = H_all[0,4:6,:,:,:]
        H[34:36, :,:,:] = H_all[1,4:6,:,:,:]
        H[36:38, :,:,:] = H_all[2,4:6,:,:,:]
        H[38:40, :,:,:] = H_all[3,4:6,:,:,:]

        H[40:42, :,:,:] = H_all[4,4:6,:,:,:]
        H[42:44, :,:,:] = H_all[5,4:6,:,:,:]
        H[44:46, :,:,:] = H_all[6,4:6,:,:,:]
        H[46:48, :,:,:] = H_all[7,4:6,:,:,:]
        H[48:50, :,:,:] = H_all[0,6:8,:,:,:]
        H[50:52, :,:,:] = H_all[1,6:8,:,:,:]
        H[52:54, :,:,:] = H_all[2,6:8,:,:,:]
        H[54:56, :,:,:] = H_all[3,6:8,:,:,:]
        H[56:58, :,:,:] = H_all[4,6:8,:,:,:]
        H[58:60, :,:,:] = H_all[5,6:8,:,:,:]

        H[60:62, :,:,:] = H_all[6,6:8,:,:,:]
        H[62:64, :,:,:] = H_all[7,6:8,:,:,:]
        H[64:66, :,:,:] = H_all[0,8:10,:,:,:]
        H[66:68, :,:,:] = H_all[1,8:10,:,:,:]
        H[68:70, :,:,:] = H_all[2,8:10,:,:,:]
        H[70:72, :,:,:] = H_all[3,8:10,:,:,:]
        H[72:74, :,:,:] = H_all[4,8:10,:,:,:]
        H[74:76, :,:,:] = H_all[5,8:10,:,:,:]
        H[76:78, :,:,:] = H_all[6,8:10,:,:,:]
        H[78:80, :,:,:] = H_all[7,8:10,:,:,:]
        H[80:82, :,:,:] = H_all[0,10:12,:,:,:]
        H[82:84, :,:,:] = H_all[1,10:12,:,:,:]
        H[84:85, :,:,:] = H_all[2,10:11,:,:,:]

        # [33, 45, 37]
        H[85:90, :,:,:] = H_all[3,10:15,:,:,:]
        H[90:96, :,:,:] = H_all[4,10:16,:,:,:]
        H[96:100, :,:,:] = H_all[5,10:14,:,:,:]
        H[100:110, :,:,:] = H_all[6,10:20,:,:,:]
        H[110:118, :,:,:] = H_all[7,10:18,:,:,:]

        H[118:131, :,:,:] = H_all[0,12:25,:,:,:]
        H[131:144, :,:,:] = H_all[1,12:25,:,:,:]
        H[144:158, :,:,:] = H_all[2,11:25,:,:,:]
        H[158:163, :,:,:] = H_all[3,15:20,:,:,:]

        H[163:168, :,:,:] = H_all[3,20:25,:,:,:]
        H[168:177, :,:,:] = H_all[4,16:25,:,:,:]
        H[177:188, :,:,:] = H_all[5,14:25,:,:,:]
        H[188:193, :,:,:] = H_all[6,20:25,:,:,:]
        H[193:200, :,:,:] = H_all[7,18:25,:,:,:]
    

def sort_vector(vector):
    sorted_vector = sorted(vector, reverse=True)
    positions = [i for i in range(len(vector))]
    positions_sorted = [pos for _, pos in sorted(zip(vector, positions), reverse=True)]
    new_matrix = np.array([sorted_vector, positions_sorted])
    return new_matrix


def sort(matrix):

    flattened_with_indices = [(value, (i, j)) for i, row in enumerate(matrix) for j, value in enumerate(row)]

    # Sort the flattened list in descending order based on values
    sorted_indices = sorted(flattened_with_indices, key=lambda x: x[0], reverse=True)

    # Extract the 2-D indices from the sorted list
    sorted_indices_2d = [index[1] for index in sorted_indices]

    return sorted_indices_2d


def check_corr(h,k):
    corr = abs(np.dot(np.matrix(h).getH(),k))/(np.linalg.norm(h)*np.linalg.norm(k))
    return corr

# assigned_RBG = 0
total_slice_tp = np.zeros((Num_slice,))
total_rb_allocated = 0
total_time = 0


for tti in range (0, total_tti):
    print("Now TTI:",tti)
    delta_slice = SLAs * (tti+1) - total_slice_tp
    # print("Delta:", delta_slice)
    # sel_result = [[] for _ in range(Num_RBG)]
    # Num_RBG_remain = 16
    RBG_remain = [num for num in range(Num_RBG)]
    

    start = time.time()

    # Channel Gain of all UEs on each RB
    cg_rb_ue = np.zeros((Num_RBG,Num_UE))

    # Initialize avg channel gain
    for rbg in RBG_remain:
        for ue in range (0, Num_UE): # UE Fixed
            cg_rb_ue[rbg,ue] = np.linalg.norm(H[ue, :, tti, rbg])
           # # [80,64,100,52]


    # print("Len RB",len(RBG_remain))
    tx_total_time = 0

    while any(element > 1 for element in delta_slice) and len(RBG_remain):
    # while (sum(1 for x in delta_slice if x > 1)>1) and len(RBG_remain):
    # New RB 
        # print("New RB:,", delta_slice, len(RBG_remain))
        # large_list = []
        UE_list = [num for num in range(Num_UE)]
        large_ue_list = []
        small_ue_list = []
        # Norm
        non_zero = [x for x in delta_slice if x > 0]
        if len(non_zero):
            mean_delat = sum(non_zero) / len(non_zero)
        else:
            mean_delat = np.mean(delta_slice)
        # mean_delat = np.mean(delta_slice)
        for sl in range (0, Num_slice):
            if delta_slice[sl] >= mean_delat:
                large_ue_list.extend(UE_list[sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])])
                # large_list.append(sl)
            else:
                small_ue_list.extend(UE_list[sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])])    
                # small_list.append(sl)
        # print("Large List:",large_ue_list)
        cg_large = cg_rb_ue[:,np.array(large_ue_list)]
        cg_small = cg_rb_ue[:,np.array(small_ue_list)]
        
        # Sort all indices, indices in slice and user channel gain 
        sorted_large = sort(cg_large) # (i,j)

        # cg_large_mean = np.mean(cg_large,axis = 1)
        rbg_index = sorted_large[0][0]
        # sorted_small = sort(cg_small)
        
        # Find RB
        sel_rbg = RBG_remain[int(rbg_index)] # relative number
        
        # ------------ Correlation Matrix ---------------------
        corr_matrix = []
        corr_time = time.time()
        for ue1 in range (0,Num_UE):
            corr_ue = []
            
            for ue2 in range (0,Num_UE):                
                corr_ue.append(check_corr(np.reshape(H[ue1, :, tti, sel_rbg],(Num_BS,1)),np.reshape(H[ue2, :, tti, sel_rbg],(Num_BS,1))))
                
            corr_matrix.append(corr_ue)
        corr_end = time.time()
        # print("Corr time:", corr_end - corr_time)
        # ------------ Correlation Matrix ---------------------

        sel_UE_list = []

        sorted_rb_large = sort_vector(cg_large[int(rbg_index),:])[1,:]
        sorted_rb_small = sort_vector(cg_small[int(rbg_index),:])[1,:]


        large_ue_remain = [large_ue_list[int(i)] for i in sorted_rb_large] # absolute
        small_ue_remain = [small_ue_list[int(i)] for i in sorted_rb_small]

        for count,dlt in enumerate (delta_slice):
            if dlt ==0:
                for i in range (sum(Num_UE_ps[:count]),sum(Num_UE_ps[:(count+1)])):
                    if i in small_ue_remain:
                        small_ue_remain.remove(i)

        # ---------------------------------- Start Allocation on one RB ---------------------------------------
        # print("Entering UE selection")
        while (len(sel_UE_list) < SEL_UE) and ((large_ue_remain) or (small_ue_remain)):
            # print("Sel UE length:",len(sel_UE_list))
            # print(large_ue_remain, small_ue_remain)
            UE_remain = large_ue_remain + small_ue_remain
            sorted_large_ue = large_ue_remain.copy()
            sorted_small_ue = small_ue_remain.copy()
            while sorted_large_ue:
                sel_UE_list.append(int(sorted_large_ue[0]))
                if len(sel_UE_list) >= SEL_UE:
                    break
                else:
                    for ue in UE_remain:
                        if (ue != sorted_large_ue[0]) and (corr_matrix[int(sorted_large_ue[0])][int(ue)] > corr_th):
                            UE_remain.remove(ue)
                            if ue in sorted_large_ue:
                                sorted_large_ue.remove(ue)
                            elif ue in sorted_small_ue:
                                sorted_small_ue.remove(ue)
                    UE_remain.remove(int(sorted_large_ue[0]))
                    large_ue_remain.remove(int(sorted_large_ue[0]))
                    sorted_large_ue.remove(int(sorted_large_ue[0]))
                    

            if len(sel_UE_list) >= SEL_UE:
                break    
            
            while sorted_small_ue:
                sel_UE_list.append(int(sorted_small_ue[0]))
                if len(sel_UE_list) >= SEL_UE:
                    break
                else:
                    for ue in UE_remain:
                        if (ue != sorted_small_ue[0]) and (corr_matrix[int(sorted_small_ue[0])][int(ue)] > corr_th):
                            UE_remain.remove(ue)
                            sorted_small_ue.remove(ue)
                    UE_remain.remove(int(sorted_small_ue[0]))
                    small_ue_remain.remove(int(sorted_small_ue[0]))
                    sorted_small_ue.remove(int(sorted_small_ue[0]))
                    


        tx_time = time.time()
        
        # ---------------------- Calculate Data Rate ----------------------------
        # print("UE sele list:", sel_UE_list)
        H_s = H[np.array(sel_UE_list), :, tti, int(sel_rbg)]
        sig_user = np.real(np.diagonal(np.linalg.inv(np.dot(H_s, H_s.conj().T))))
        SINR = 10 / sig_user
        cap_per_ue = np.log2(1+SINR)
        sum_rate = np.sum(cap_per_ue)

        # print("Sum rate:",sum_rate, sel_UE_list)
        # -------------------------------------------------------------------------
        # [10, 12, 18, 20, 25, 33, 45, 37]
        for i, ue in enumerate (sel_UE_list):
            if ue < 10:
                total_slice_tp[0] += cap_per_ue[i]
            elif ue < 22:
                total_slice_tp[1] += cap_per_ue[i]
            elif ue < 40:
                total_slice_tp[2] += cap_per_ue[i]
            elif ue < 60:
                total_slice_tp[3] += cap_per_ue[i]
            elif ue < 85:
                total_slice_tp[4] += cap_per_ue[i]
            elif ue < 118:
                total_slice_tp[5] += cap_per_ue[i]
            elif ue < 163:
                total_slice_tp[6] += cap_per_ue[i]
            elif ue < 200:
                total_slice_tp[7] += cap_per_ue[i]

        # for i, ue in enumerate (sel_UE_list):
        #     total_slice_tp[int(ue//num_ps)] += cap_per_ue[i]

        cg_rb_ue = np.delete(cg_rb_ue, int(rbg_index), axis=0) 
        RBG_remain.remove(RBG_remain[int(rbg_index)])
        

        # Updata Delta
        delta_slice = SLAs * (tti+1) - total_slice_tp
        for count,dlt in enumerate (delta_slice):
            if dlt <=0:
                cg_rb_ue[:,UE_list[sum(Num_UE_ps[:count]):sum(Num_UE_ps[:(count+1)])]] = 0
        delta_slice[delta_slice < 0] = 0
        # print("Delta:", delta_slice)

        end_in = time.time()
        tx_total_time += end_in - tx_time
    
    total_rb_allocated += Num_RBG - len(RBG_remain)

    end = time.time()
    total_time += end - start - tx_total_time

    

avg_slice_tp = total_slice_tp / total_tti
avg_time = total_time / total_tti
print("Average TP is:", avg_slice_tp)
print("Total Assgined RBG:", total_rb_allocated/total_tti)
# print("Average time is:", avg_time)
        
        
        
        
        
        
        
        
        
        
        
