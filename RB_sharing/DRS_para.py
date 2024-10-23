import numpy as np 
import time
import h5py
import math

print("Data Loading and preprocessing")
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
new_rb_para = 1

# ----------------- Network Configurations ---------------------- #

config = 200

if config == 200:
    Num_UE = 200 #
    Num_slice = 8

    Num_UE_ps = [10, 12, 18, 20, 25, 33, 45, 37] # average 25

    # SLAs = np.array([135, 120, 130, 140, 45, 110, 50, 125])
    SLAs = np.array([235, 220, 230, 240, 145, 210, 150, 225])

    SEL_UE = 16
    

    H = np.empty((Num_UE,Num_BS,Num_tti,Num_RBG), dtype = 'complex64')

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
    
# ----------------- Functions ---------------------- #


def sort_vector(vector):
    sorted_vector = sorted(vector, reverse=True)
    positions = [i for i in range(len(vector))]
    positions_sorted = [int(pos) for _, pos in sorted(zip(vector, positions), reverse=True)]
    new_matrix = np.array([sorted_vector, positions_sorted])
    return new_matrix


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


corr_mat_set = np.empty((total_tti, Num_RBG, Num_UE, Num_UE),dtype = 'float32')
cg_tti_rb_ue = np.empty((total_tti, Num_RBG, Num_UE),dtype = 'float32')
group_all = []
group_len = []

for tti in range (total_tti):
    group_rbg = []
    for rbg in range (Num_RBG):  
        group_ue, corr_mat, cg_ue = group_alg(H[:, :, tti, rbg])
        corr_mat_set[tti,rbg,:,:] = corr_mat
        group_rbg.append(group_ue)
        cg_tti_rb_ue[tti,rbg,:] = cg_ue
    group_all.append(group_rbg)


def alloc_rb (num_rb, tti, H, ue_index, large_ue_list, small_ue_list):

    group_list= group_all[tti][num_rb]
    len_remain = SEL_UE
    sel_UE_list = []
    while len_remain > 0:
        # print(group_list, len_remain, sel_UE_list, ue_index)
        for group in group_list:
            if ue_index in group:
                if len(group) <= len_remain:
                    for ele in group:
                        if ele not in sel_UE_list:
                            if ele in large_ue_list: 
                                sel_UE_list.append(ele)
                                large_ue_list.remove(ele)
                                len_remain -= 1 
                            elif ele in small_ue_list:
                                sel_UE_list.append(ele)
                                small_ue_list.remove(ele) 
                                len_remain -= 1
                    group_list.remove(group)
                    break
                else: # Finish allocation in this loop
                    for ele in group:
                        # print("ELE:", ele)
                        if ele not in sel_UE_list:
                            if ele in large_ue_list:
                                sel_UE_list.append(ele)
                                large_ue_list.remove(ele)
                                len_remain -= 1 
                                # group.remove(ele)
                        if len_remain <= 0:
                            break
                    
                    if len_remain >0:
                        for ele in group:
                            if ele not in sel_UE_list:
                                if ele in small_ue_list:
                                    sel_UE_list.append(ele)
                                    small_ue_list.remove(ele) 
                                    len_remain -= 1 
                            if len_remain <= 0:
                                break 

                    break

        if len_remain > 0:
            # large_ue_list = [int(x) for x in large_ue_list]
            # print("Large UE List:", large_ue_list)
            # print("Small UE List:", small_ue_list, delta_slice)
            # print(group_list, len_remain, sel_UE_list)
            if large_ue_list:
                ue_index = large_ue_list[np.argmax(cg_rb_ue[int(rbg_index),np.array(large_ue_list)])]  
            elif small_ue_list:
                ue_index = small_ue_list[np.argmax(cg_rb_ue[int(rbg_index),np.array(small_ue_list)])] 
            else:
                break

    H_s = H[np.array(sel_UE_list), :, tti, int(num_rb)]
    sig_user = np.real(np.diag(np.linalg.inv(np.dot(H_s, H_s.conj().T))))
    SINR = 10 / sig_user
    cap_per_ue = np.log2(1+SINR)
    return sel_UE_list, cap_per_ue



# total_slice_tp = np.zeros((Num_slice,))
total_slice_tp = SLAs * 100
total_rb_allocated = 0
total_time = 0
show_time = 0
period = 1

parallel = 1


for tti in range (0, total_tti):
    print("Now TTI:",tti)
    # if tti % period and (tti>1):
    #     alloc_reuse_time, num_alloc_rb = alloc_reuse(last_alloc_result, H, total_slice_tp)
    # elif (tti % period == 0) or (tti == 1):
    delta_slice = SLAs * (100+tti+1) - total_slice_tp
    last_slice_tp = total_slice_tp.copy()
    RBG_remain = [num for num in range(Num_RBG)]
    
    # last_alloc_result = []
    start = time.time()

    # Channel Gain of all UEs on each RB
    cg_rb_ue = cg_tti_rb_ue[tti,:,:]

    # if not new_rb_para:
    #     if tti > 1:
    #         # parallel = 1
    #         parallel = int(math.sqrt(current_alloc_rb * 2))+1
    #     else:
    #         parallel = 1

    # parallel = 1
    cg_time = time.time()

    rb_alloc_time = 0
    change_flag = 0
    while any(element > 1 for element in delta_slice) and len(RBG_remain):
        
        loop_time = time.time()
        UE_list = [num for num in range(Num_UE)]
        large_ue_list = []
        small_ue_list = []
        
        non_zero = [x for x in delta_slice if x > 0]
        if len(non_zero):
            mean_delta = sum(non_zero) / len(non_zero)
        else:
            mean_delta = np.mean(delta_slice)

        if new_rb_para: 
            if tti > 1:
                parallel = int(np.max(non_zero) * len(non_zero) / avg_rb)+1
                # parallel = int(np.mean(non_zero) * len(non_zero) / avg_rb)+1

            else:
                parallel = 1


        # Adaptive Parallel
        # if tti > 1:
        #     parallel = int(min(non_zero) * len(non_zero) / avg_rb)+1
        # else:
        #     parallel = 1
        print("Current parallel:",parallel)

        for sl in range (0, Num_slice):
            if delta_slice[sl] >= mean_delta:
                large_ue_list += UE_list[sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])]
            elif (delta_slice[sl] < mean_delta) and (delta_slice[sl]>0):
                small_ue_list += UE_list[sum(Num_UE_ps[:sl]):sum(Num_UE_ps[:(sl+1)])] 

        cg_large = cg_rb_ue[:,np.array(large_ue_list)]
        # if show_time:
        #     delta_time = time.time()
        #     print("Delta classify time:", delta_time - loop_time)

        if parallel == 1:
            max_index = np.unravel_index(cg_large.argmax(), cg_large.shape)
            rbg_index = max_index[0]
            sel_rbg = RBG_remain[int(rbg_index)]
            ue_index = large_ue_list[int(max_index[1])]
            sel_UE_list, cap_per_ue = alloc_rb(sel_rbg, tti, H, ue_index, large_ue_list, small_ue_list)
            # for i, ue in enumerate (sel_UE_list):
            #     total_slice_tp[int(ue//num_ps)] += cap_per_ue[i]

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

            cg_rb_ue = np.delete(cg_rb_ue, int(rbg_index), axis=0) 
            RBG_remain.remove(RBG_remain[int(rbg_index)])
            # avg_rb = np.sum(cap_per_ue) / parallel
        else:    
            print("Multiple RBs")
            large_rb_list = []
            ue_index_list = []
            # cg_find_rb = cg_large.copy()
            iter_time = time.time()
            total_rate = 0
            for count in range (parallel):
                max_index = np.unravel_index(cg_rb_ue[:,np.array(large_ue_list)].argmax(), cg_rb_ue[:,np.array(large_ue_list)].shape)
                rbg_index = max_index[0]
                sel_rbg = RBG_remain[int(rbg_index)] # absolute value
                # print("Current RBG:", sel_rbg)
                # ue_index = large_ue_list[np.argmax(cg_large[int(rbg_index),:])]
                ue_index = large_ue_list[int(max_index[1])]
                # cg_large = np.delete(cg_large, int(rbg_index), axis=0) 
                
                large_ue = large_ue_list.copy()
                small_ue = small_ue_list.copy()

                # ---------------------------------- Start Allocation on one RB ---------------------------------------
                sel_UE_list, cap_per_ue = alloc_rb(sel_rbg, tti, H, ue_index, large_ue, small_ue)

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
                total_rate += np.sum(cap_per_ue)
            
            # avg_rb = total_rate / parallel
            iter_over = time.time()
            
                # if show_time:
                #     delete_time = time.time()
                #     print("delete time:", delete_time - sel_time)


        # Updata Delta
        delta_slice = SLAs * (100+tti+1) - total_slice_tp

        end_in = time.time()


        if parallel >1:
            rb_time = end_in - iter_over + (iter_over-iter_time)/parallel + iter_time - loop_time
        else: 
            rb_time = end_in - loop_time
        print("One RB time:", rb_time, '\n')
        rb_alloc_time += rb_time

        #Double Parallel
        if not new_rb_para:
            if (parallel > 1) and (change_flag == 1):
                parallel -= 1
                change_flag = 0
            else:
                change_flag = 1

        # if (parallel > 1) :
        #     parallel -= 1

        

        # last_alloc_result.append((sel_rbg, sel_UE_list))

    rb_finish = time.time()
    tp_tti = np.sum(total_slice_tp - last_slice_tp)
    current_alloc_rb = Num_RBG - len(RBG_remain)
    total_rb_allocated += current_alloc_rb
    avg_rb = tp_tti / current_alloc_rb
    end = time.time()
    if tti>1:
        total_time += (end - rb_finish + rb_alloc_time + cg_time - start)
    print("One TTI time:", (end - rb_finish + rb_alloc_time + cg_time - start), "\n")


    

avg_slice_tp = total_slice_tp / total_tti
avg_time = total_time / (total_tti-2)
print("Average TP is:", avg_slice_tp/11)
print("Total Assgined RBG:", total_rb_allocated/(total_tti))
print("Average time is:", avg_time)
        
        
        
        
        
        
        
        
        
        
        

