import numpy as np 
import time
import h5py


channel_file = h5py.File('DU_8_26_64_100_52.hdf5','r')
H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_all = np.array(H_r + 1j*H_i)
# (4, 26, 64, 100, 52)
# print(H_all.shape)

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
        H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2], :,:,:] = H_all[4,i*ue_num_list[2]:(i+1)*ue_num_list[2],:,:,:] * 0.1
        H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]+ue_num_list[3], :,:,:] = H_all[5,i*ue_num_list[3]:(i+1)*ue_num_list[3],:,:,:] * 0.1

rt_history = np.ones((Num_UE,)) * 0.01

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
                large_ue_list.extend(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps])
                # large_list.append(sl)
            else:
                small_ue_list.extend(UE_list[sl*Num_UE_ps:(sl+1)*Num_UE_ps])    
                # small_list.append(sl)
        # print("Large List:",large_list)
        cg_large = cg_rb_ue[:,np.array(large_ue_list)]
        cg_small = cg_rb_ue[:,np.array(small_ue_list)]
        
        # Sort all indices, indices in slice and user channel gain 
        sorted_large = sort(cg_large) # (i,j)

        # cg_large_mean = np.mean(cg_large,axis = 1)
        rbg_index = sorted_large[0][0]
        # sorted_small = sort(cg_small)
        
        # Find RB
        sel_rbg = RBG_remain[int(rbg_index)] # absolute number
        
        cg_large_pf = np.zeros((len(large_ue_list),))
        cg_small_pf = np.zeros((len(small_ue_list),))
        for count, ue in enumerate (large_ue_list):
            sl = ue // Num_UE_ps
            cg_large_pf[count] = cg_large[int(rbg_index),count] / (rt_history[ue]/np.max(rt_history[sl*Num_UE_ps:(sl+1)*Num_UE_ps]))
            # cg_large_pf[count] = cg_large[int(rbg_index),count] / rt_history[ue]
        for count, ue in enumerate (small_ue_list):
            sl = ue // Num_UE_ps
            cg_small_pf[count] = cg_small[int(rbg_index),count] / (rt_history[ue]/np.max(rt_history[sl*Num_UE_ps:(sl+1)*Num_UE_ps]))
            # cg_small_pf[count] = cg_small[int(rbg_index),count] / rt_history[ue]
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

        sorted_rb_large = sort_vector(cg_large_pf)[1,:]
        sorted_rb_small = sort_vector(cg_small_pf)[1,:]


        large_ue_remain = [large_ue_list[int(i)] for i in sorted_rb_large] # absolute
        small_ue_remain = [small_ue_list[int(i)] for i in sorted_rb_small]

        for count,dlt in enumerate (delta_slice):
            if dlt ==0:
                for i in range (count*Num_UE_ps,(count+1)*Num_UE_ps):
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

        H_s = H[np.array(sel_UE_list), :, tti, int(sel_rbg)]
        sig_user = np.real(np.diagonal(np.linalg.inv(np.dot(H_s, H_s.conj().T))))
        SINR = 10 / sig_user
        cap_per_ue = np.log2(1+SINR)
        sum_rate = np.sum(cap_per_ue)

        # print("Sum rate:",sum_rate)
        # -------------------------------------------------------------------------

        for i, ue in enumerate (sel_UE_list):
            total_slice_tp[int(ue//Num_UE_ps)] += cap_per_ue[i]
        
        for count, ue in enumerate (sel_UE_list):
            rt_history[ue] += cap_per_ue[count]

        cg_rb_ue = np.delete(cg_rb_ue, int(rbg_index), axis=0) 
        RBG_remain.remove(RBG_remain[int(rbg_index)])
        

        # Updata Delta
        delta_slice = SLAs * (tti+1) - total_slice_tp
        # for count,dlt in enumerate (delta_slice):
        #     if dlt <=0:
        #         cg_rb_ue[:,count*Num_UE_ps:(count+1)*Num_UE_ps] = 0
        delta_slice[delta_slice < 0] = 0
        # print("Delta:", delta_slice)

        end_in = time.time()
        tx_total_time += end_in - tx_time
    
    total_rb_allocated += Num_RBG - len(RBG_remain)

    end = time.time()
    total_time += end - start - tx_total_time

jfi = np.zeros((Num_slice,))
for i in range (Num_slice):
    jfi[i] = np.square((np.sum(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps]))) / (Num_UE_ps * np.sum(np.square(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps])))
print('avg jfi is', np.mean(jfi))    

avg_slice_tp = total_slice_tp / total_tti
avg_time = total_time / total_tti
print("Average TP is:", avg_slice_tp)
print("Total Assgined RBG:", total_rb_allocated/total_tti)
# print("Average time is:", avg_time)
        
        
        
        
        
        
        
        
        
        
        
