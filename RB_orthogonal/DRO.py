import numpy as np 
import time
import h5py


channel_file = h5py.File('DU_8_26_64_100_52.hdf5','r')
H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_all = np.array(H_r + 1j*H_i)
# # (4, 26, 64, 100, 52)
# print(H_all.shape)

# Define Configurations
Num_tti = 100
Num_RBG = 52
Num_BS = 64
corr_th = 0.5
total_tti = 10

config = 80 # 16 is small size, 80 is medium size
uncorr = 0 # correlated case and uncorrelated case
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

    # ue_num_list = [3,3,2,2]
    # for i in range (Num_slice):
    #     H[i*Num_UE_ps:i*Num_UE_ps+ue_num_list[0], :,:,:] = H_all[0,i*ue_num_list[0]:(i+1)*ue_num_list[0],:,:,:]
    #     H[i*Num_UE_ps+ue_num_list[0]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1], :,:,:] = H_all[1,i*ue_num_list[1]:(i+1)*ue_num_list[1],:,:,:]
    #     H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2], :,:,:] = H_all[4,i*ue_num_list[2]:(i+1)*ue_num_list[2],:,:,:] *0.1
    #     H[i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]:i*Num_UE_ps+ue_num_list[0]+ue_num_list[1]+ue_num_list[2]+ue_num_list[3], :,:,:] = H_all[5,i*ue_num_list[3]:(i+1)*ue_num_list[3],:,:,:]*0.1

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

# elif config == 200:
#     Num_UE = 200 #
#     Num_slice = 8

#     Num_UE_ps = int(Num_UE/Num_slice)

#     if loose_sla == 1:
#         SLAs = np.array([135, 120, 130, 140, 45, 110, 50, 125])
#     else:
#         SLAs = np.array([235, 220, 230, 240, 145, 210, 150, 225])

#     SEL_UE = 16
    

#     H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')

#     #---------------------------- 8 slices 25 UEs Cluster Deployment ----------------------------
#     if uncorr ==0:
#         for i in range (Num_slice):
#             H[i*Num_UE_ps:(i+1)*Num_UE_ps, :,:,:] = H_all[i,0:Num_UE_ps,:,:,:]

#     #---------------------------- 8 slices 25 UEs Evenly Distributed Deployment ----------------------------
#     else:
#         ue_num_list = [4,3,3,3,3,3,3,3]
#         for i in range (Num_slice):
#             for j in range (8):
#                 if i == j:
#                     H[(i*Num_UE_ps+j*3):(i*Num_UE_ps+j*3+4), :,:,:] = H_all[j,i*3:((i+1)*3+1),:,:,:]
#                 elif j>i:
#                     H[(i*Num_UE_ps+j*3+1):(i*Num_UE_ps+(j+1)*3+1), :,:,:] = H_all[j,i*3:((i+1)*3),:,:,:]
#                 else:
#                     H[(i*Num_UE_ps+j*3):(i*Num_UE_ps+(j+1)*3), :,:,:] = H_all[j,(i*3+1):((i+1)*3+1),:,:,:]

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
rt_history = np.ones((Num_UE,)) * 0.01

for tti in range (0, total_tti):
    print("Now TTI:",tti)
    delta_slice = SLAs * (tti+1) - total_slice_tp
    # print("Delta:", delta_slice)
    # sel_result = [[] for _ in range(Num_RBG)]
    # Num_RBG_remain = 16
    RBG_remain = [num for num in range(Num_RBG)]
    UE_list = [num for num in range(Num_UE)]

    start = time.time()

    # Channel Gain of all UEs on each RB
    cg_rb_ue = np.zeros((Num_RBG,Num_UE))
    avg_cb_rb_slice = np.zeros((Num_RBG,Num_slice))

    # cg_time = time.time()

    # Initialize avg channel gain
    for rbg in RBG_remain:
        for ue in range (0, Num_UE): # UE Fixed
            cg_rb_ue[rbg,ue] = np.linalg.norm(H[ue, :, tti, rbg])
        for sl in range (0, Num_slice):
            avg_cb_rb_slice[rbg,sl] = np.mean(cg_rb_ue[rbg,sl*Num_UE_ps:(sl+1)*Num_UE_ps])

    # print("CG time:", cg_time - start)


    tx_total_time = 0
    while any(element > 0 for element in delta_slice) and len(RBG_remain):
    # New RB 
        # start = time.time()
        
        large_sl_list = []
        # Norm
        non_zero = [x for x in delta_slice if x > 0]
        if len(non_zero):
            mean_delta = sum(non_zero) / len(non_zero)
        else:
            mean_delta = np.mean(delta_slice)
        
        for sl in range (0, Num_slice):
            if delta_slice[sl] >= mean_delta:
                large_sl_list.append(int(sl))
        # print(mean_delta, delta_slice,large_sl_list)
        avg_cg_large = avg_cb_rb_slice[:,np.array(large_sl_list)]
        # print(avg_cg_large.shape)

        sorted_large = sort(avg_cg_large)

        # ----------------Find RB and slice---------------------
        sel_rbg = RBG_remain[int(sorted_large[0][0])] # absolute number (rbg,slice)
        sel_sl = large_sl_list[int(sorted_large[0][1])]
        # ----------------Find RB and slice---------------------


        cg_user = cg_rb_ue[sel_rbg,sel_sl*Num_UE_ps:(sel_sl+1)*Num_UE_ps]
        # print(cg_rb_ue)
        ue_remain = list(sort_vector(cg_user)[1,:])
        sel_UE_list = []

        while (len(sel_UE_list) < SEL_UE) and (len(sel_UE_list) < Num_UE_ps):

            sorted_ue = ue_remain.copy()
            sel_UE_list.append(sel_sl*Num_UE_ps + int(sorted_ue[0]))
            ue_remain.remove(ue_remain[0])
            for idx in sorted_ue[1:]:
                vio_th = 0
                for ue in sel_UE_list:
                    # corr_s = time.time()
                    corr = check_corr(np.reshape(H[int(sel_sl*Num_UE_ps + idx), :, tti, sel_rbg],(Num_BS,1)),np.reshape(H[int(ue), :, tti, sel_rbg],(Num_BS,1)))[0,0]
                    # corr_e = time.time()
                    # print("corr time:", corr_e - corr_s)
                    if corr > corr_th:
                        vio_th = 1
                if vio_th == 0:
                    sel_UE_list.append(int(sel_sl*Num_UE_ps + idx))
                    ue_remain.remove(idx)
                if (len(sel_UE_list) >= SEL_UE) or (len(sel_UE_list) == Num_UE_ps):
                    break


        tx_time = time.time()


        # ---------------------- Calculate Data Rate ----------------------------
        H_s = H[np.array(sel_UE_list), :, tti, int(sel_rbg)]
        
        # print("H_s shape:", H_s)
        # zf_s = np.linalg.pinv(H_s)

        sig_user = np.real(np.diagonal(np.linalg.inv(np.dot(H_s, H_s.conj().T))))
        SINR = 10 / sig_user
        cap_per_ue = np.log2(1+SINR)
        # print("Cap per UE:", cap_per_ue)
        sum_rate = np.sum(cap_per_ue)

        # print("Sum rate:",sum_rate)
        # -------------------------------------------------------------------------

        total_slice_tp[sel_sl] += sum_rate
        for count, ue in enumerate (sel_UE_list):
            rt_history[ue] += cap_per_ue[count]

        avg_cb_rb_slice = np.delete(avg_cb_rb_slice, int(sorted_large[0][0]), axis=0) 
        RBG_remain.remove(RBG_remain[int(sorted_large[0][0])])
           

        # Updata Delta
        delta_slice = SLAs * (tti+1) - total_slice_tp
        delta_slice[delta_slice < 0] = 0
        # print("Delta:", delta_slice)
        
        
        end_in = time.time()
        tx_total_time += end_in - tx_time
        # print("Running time is:", end - start, '\n')
    
    total_rb_allocated += Num_RBG - len(RBG_remain)

    end = time.time()
    total_time += end - start - tx_total_time
    # print("tx total", tx_total_time)

jfi = np.zeros((Num_slice,))
# for i in range (Num_slice):
#     jfi[i] = np.square((np.sum(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps]))) / (Num_UE_ps * np.sum(np.square(rt_history[i*Num_UE_ps:(i+1)*Num_UE_ps])))
# print('jfi is', jfi)


avg_slice_tp = total_slice_tp / total_tti
# avg_time = total_time / total_tti
print("Average TP is:", avg_slice_tp)
print("Avg Assgined RBG:", total_rb_allocated / total_tti)
# print("Average time is:", avg_time)



        
        
        
        
        
        
        
        
        
        
        
