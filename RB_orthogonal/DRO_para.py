import numpy as np 
import time
import h5py
import math

print("Data Loading and preprocessing")
channel_file = h5py.File('DU_8_26_64_100_52.hdf5','r')
H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_all = np.array(H_r + 1j*H_i)
# (4, 26, 64, 100, 52)
print(H_all.shape)

# Define Configurations
Num_tti = 100
Num_RBG = 52
Num_BS = 64
corr_th = 0.5
total_tti = 10
new_rb_para = 1

config = 200

if config == 200:
    Num_UE = 200 #
    Num_slice = 8

    Num_UE_ps = [10, 12, 18, 20, 25, 33, 45, 37] # average 25

    # SLAs = np.array([135, 120, 130, 140, 45, 110, 50, 125])
    SLAs = np.array([235, 220, 230, 240, 145, 210, 150, 225])

    SEL_UE = 16
    

    H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')

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

# ----------------- Functions ---------------------- #
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


def group_alg(H):
    corr_matrix = np.empty((H.shape[0],H.shape[0]),dtype = 'float32')
    # print(corr_matrix.shape, H.shape)
    user_set = [num for num in range (Num_UE)]
    user_set_remain = [num for num in range (Num_UE)]
    group_set = []
    low_set = []
    cg_ue = np.zeros((H.shape[0],))
    
    for i in range (H.shape[0]):
        cg_ue[i] = np.linalg.norm(H[i, :])
        for j in range (H.shape[0]):
            if i != j:
                corr_matrix[i,j] = check_corr(H[i,:].reshape(Num_BS,1),H[j,:].reshape(Num_BS,1))[0,0]
                # print(corr_matrix[i,j])
            else:
                corr_matrix[i,j] = 0
    for i in range (corr_matrix.shape[0]):
        if all(x < corr_th for x in corr_matrix[i,:]):
            # print("Low User")
            low_set.append(i)
            # print(low_set)
            user_set_remain.remove(i)
            # print(user_set_remain)
    if len(low_set)>1:
        cg_group = cg_ue[np.array(low_set)]
        cg_group_sort = sort_vector(cg_group)[1,:]
        low_set = [low_set[int(i)] for i in cg_group_sort]


    if user_set_remain:
        while user_set_remain:
            small_group = []
            check_corr_user = user_set_remain.copy()
            while check_corr_user:
                small_group.append(check_corr_user[0])
                user_set_remain.remove(check_corr_user[0])
                if len(check_corr_user) > 1:
                    for ue in check_corr_user[1:]:
                        if corr_matrix[check_corr_user[0],ue] > corr_th:
                            check_corr_user.remove(ue)
                
                check_corr_user.remove(check_corr_user[0])
            small_group += low_set

            # -- Reorder list by cg -- #
            if len(small_group) > 1:
                cg_group = cg_ue[np.array(small_group)] 
                cg_group_sort = sort_vector(cg_group)[1,:]
                small_group = [small_group[int(i)] for i in cg_group_sort]

            group_set.append(small_group)
    else:
        group_set.append(low_set)

    return group_set, corr_matrix, cg_ue

def alloc_rb(sel_rbg, sel_sl, rbg_index, tti, H, sl_ue_list):
    sel_UE_list = []
            
    if Num_UE_ps[sel_sl] <= SEL_UE:
        sel_UE_list = UE_list[sum(Num_UE_ps[:sel_sl]):sum(Num_UE_ps[:(sel_sl+1)])]
    else:
        group_list = group_set[tti][sel_rbg]
        while len(sel_UE_list) < SEL_UE:
            ue_index = sl_ue_list[np.argmax(cg_rb_ue[int(rbg_index),np.array(sl_ue_list)])]
            for group in group_list:
                if ue_index in group:
                    for ele in group:
                        if (ele not in sel_UE_list) and (ele in sl_ue_list):
                            sel_UE_list.append(ele)
                            sl_ue_list.remove(ele)
                        if len(sel_UE_list) >= SEL_UE:
                            break
                    group_list.remove(group)
                    break

    # ---------------------- Calculate Data Rate ----------------------------
    H_s = H[np.array(sel_UE_list), :, tti, int(sel_rbg)]

    sig_user = np.real(np.diagonal(np.linalg.inv(np.dot(H_s, H_s.conj().T))))
    SINR = 10 / sig_user
    cap_per_ue = np.log2(1+SINR)
    sum_rate = np.sum(cap_per_ue)
    # print("Sel UE list and data rate:", sel_UE_list, sum_rate)
    
    return sum_rate
    # -------------------------------------------------------------------------


# ----------------- Functions ---------------------- #

corr_mat_set = np.empty((total_tti, Num_RBG, Num_UE, Num_UE),dtype = 'float32')
cg_tti_rb_ue = np.empty((total_tti, Num_RBG, Num_UE),dtype = 'float32')
group_set = []

for tti in range (total_tti):
    group_rbg = []
    
    for rbg in range (Num_RBG):  
        group_ue, corr_mat, cg_ue = group_alg(H[:, :, tti, rbg])
        corr_mat_set[tti,rbg,:,:] = corr_mat
        group_rbg.append(group_ue)
        cg_tti_rb_ue[tti,rbg,:] = cg_ue
    group_set.append(group_rbg)

total_slice_tp = SLAs * 100
total_rb_allocated = 0
total_time = 0
avg_inter = 0
parallel = 1


for tti in range (0, total_tti):
    print("Now TTI:",tti)
    delta_slice = SLAs * (100+tti+1) - total_slice_tp
    # print("Delta:", delta_slice)
    # sel_result = [[] for _ in range(Num_RBG)]
    # Num_RBG_remain = 16
    RBG_remain = [num for num in range(Num_RBG)]
    UE_list = [num for num in range(Num_UE)]

    start = time.time()

    # Channel Gain of all UEs on each RB
    cg_rb_ue = cg_tti_rb_ue[tti,:,:]
    avg_cb_rb_slice = np.zeros((Num_RBG,Num_slice))

    if new_rb_para: 
        if tti > 1:
            parallel = int(math.sqrt(current_alloc_rb * 2))+1
        else:
            parallel = 1
    # 

    # Initialize avg channel gain
    # for rbg in RBG_remain:
    #     for sl in range (0, Num_slice):
    #         avg_cb_rb_slice[rbg,sl] = np.mean(cg_rb_ue[rbg,sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])])

    cg_time = time.time()
    # print("cg time:",cg_time - start)

    rb_alloc_time = 0
    change_flag = 0
    while any(element > 1 for element in delta_slice) and len(RBG_remain):
        print("Current parallel:",parallel)
    # New RB 
        # start = time.time()
        loop_time = time.time()
        large_ue_list = []
        small_ue_list = []
        # Norm
        non_zero = [x for x in delta_slice if x > 0]
        if len(non_zero):
            mean_delta = sum(non_zero) / len(non_zero)
        else:
            mean_delta = np.mean(delta_slice)
        
        for sl in range (0, Num_slice):
            if delta_slice[sl] >= mean_delta:
                large_ue_list += UE_list[sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])]
            elif (delta_slice[sl] < mean_delta) and (delta_slice[sl]>0):
                small_ue_list += UE_list[sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])] 

        cg_large = cg_rb_ue[:,np.array(large_ue_list)]

        if parallel == 1:
            max_index = np.unravel_index(cg_large.argmax(), cg_large.shape)
            rbg_index = max_index[0]
            sl_index = large_ue_list[int(max_index[1])]


            # ----------------Find RB and slice---------------------
            sel_rbg = RBG_remain[int(rbg_index)] # absolute number (rbg,slice)
            # sel_sl = int(sl_index//num_ps) 
            if sl_index < 10:
                sel_sl = 0
            elif sl_index < 22:
                sel_sl = 1
            elif sl_index < 40:
                sel_sl = 2
            elif sl_index < 60:
                sel_sl = 3
            elif sl_index < 85:
                sel_sl = 4
            elif sl_index < 118:
                sel_sl = 5
            elif sl_index < 163:
                sel_sl = 6
            elif sl_index < 200:
                sel_sl = 7

            sl_ue_list = UE_list[sum(Num_UE_ps[:sel_sl]):sum(Num_UE_ps[:(sel_sl+1)])]
            # ----------------Find RB and slice---------------------

            # prep_time = time.time()
            # print("Remove finished slice Time:", prep_time - delta_time)
            # print("Select slice:", sel_sl)
            sum_rate = alloc_rb(sel_rbg, sel_sl, rbg_index, tti, H, sl_ue_list)
            # print("Sum rate is:", sum_rate)
            total_slice_tp[sel_sl] += sum_rate

            cg_rb_ue = np.delete(cg_rb_ue, int(rbg_index), axis=0) 
            RBG_remain.remove(RBG_remain[int(rbg_index)])
        else:
            # print("Multiple RBs")
            large_rb_list = []
            # ue_index_list = []
            # cg_find_rb = cg_large.copy()
            iter_time = time.time()
            for count in range (parallel):
                max_index = np.unravel_index(cg_large.argmax(), cg_large.shape)
                rbg_index = max_index[0]
                sl_index = large_ue_list[int(max_index[1])]

                sel_rbg = RBG_remain[int(rbg_index)] # absolute value
                # sel_sl = int(sl_index//num_ps)# absolute value\

                if sl_index < 10:
                    sel_sl = 0
                elif sl_index < 22:
                    sel_sl = 1
                elif sl_index < 40:
                    sel_sl = 2
                elif sl_index < 60:
                    sel_sl = 3
                elif sl_index < 85:
                    sel_sl = 4
                elif sl_index < 118:
                    sel_sl = 5
                elif sl_index < 163:
                    sel_sl = 6
                elif sl_index < 200:
                    sel_sl = 7

                sl_ue_list = UE_list[sum(Num_UE_ps[:sel_sl]):sum(Num_UE_ps[:(sel_sl+1)])]
                
                cg_large = np.delete(cg_large, int(rbg_index), axis=0)
                sum_rate = alloc_rb(sel_rbg, sel_sl, rbg_index, tti, H, sl_ue_list)

                total_slice_tp[sel_sl] += sum_rate

                cg_rb_ue = np.delete(cg_rb_ue, int(rbg_index), axis=0) 
                RBG_remain.remove(RBG_remain[int(rbg_index)])
            iter_over = time.time()
            

        # Updata Delta
        delta_slice = SLAs * (100+tti+1) - total_slice_tp
        # delta_slice[delta_slice < 0] = 0

        end_in = time.time()

        if parallel >1:
            rb_time = end_in - iter_over + (iter_over-iter_time)/parallel + iter_time - loop_time
        else: 
            rb_time = end_in - loop_time
        print("One RB time:", rb_time, '\n')
        rb_alloc_time += rb_time

        # Double Parallel
        if (parallel > 1) and (change_flag == 1):
            parallel -= 1
            change_flag = 0
        else:
            change_flag = 1

        # if (parallel > 1) :
        #     parallel -= 1
    
    rb_finish = time.time()

    current_alloc_rb = Num_RBG - len(RBG_remain)
    total_rb_allocated += current_alloc_rb
    end = time.time()
    print("end time:",end - rb_finish)

    if tti>1:
        total_time += (end - rb_finish + rb_alloc_time + cg_time - start)
    print("One TTI time:", (end - rb_finish + rb_alloc_time + cg_time - start), "\n")

    
# avg_slice_tp = total_slice_tp / total_tti
# avg_time = total_time / (total_tti-1)
# print("Average TP is:", avg_slice_tp)
# print("Avg Assgined RBG:", total_rb_allocated / total_tti)
# print("Average time is:", avg_time)
# print("Average inter time is:", avg_inter / (total_tti-1))

avg_slice_tp = total_slice_tp / total_tti
avg_time = total_time / (total_tti-2)
print("Average TP is:", avg_slice_tp/11)
print("Total Assgined RBG:", total_rb_allocated/(total_tti))
print("Average time is:", avg_time)


        
        
        
        
        
        
        
        
        
        
        
