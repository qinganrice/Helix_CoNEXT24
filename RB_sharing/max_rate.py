import numpy as np 
import time
from itertools import combinations
import h5py
# from mimo_sim_ul import *
### Configuration Setting ###
# P_N_UE = 10 # How many UEs are scheduled each frame
BS_ANT = 64


### Given Scheduling Decision, decode scheduled UEs ###
def sel_ue(user_set, action, P_N_UE):
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

### Make Scheduling Decisions ###
def choose_action(user_set, H, P_N_UE, Num_UE): # t is Frame #
    # print("Entering......")
    actions_num = 0 # number of actions
    for i in range (1,P_N_UE+1):
        actions_num += len(list(combinations(user_set,i)))
    # PF = np.zeros(actions_num)
    cap_total = np.zeros(actions_num)
    cap_per_ue = np.zeros((actions_num,Num_UE))
    # print('Current Frame is:',t)
    for action in range (0, actions_num):
        
        ue_select, idx = sel_ue(user_set, action, P_N_UE)
        
        
        H_s = H[ue_select,:]
        sig_user = np.real(np.diagonal(np.linalg.inv(np.dot(H_s, H_s.conj().T))))
        SINR = 10 / sig_user
        data_rate = np.log2(1+SINR)
        cap_per_ue[action,0:idx] = data_rate
        # print("data rate:",cap_per_ue[action,0:idx])
        cap_total[action] = np.sum(data_rate)

        # cap_total[action], _, cap_per_ue[action,0:idx] = data_process(np.reshape(H_s,(BS_ANT,-1)),idx,mod_select)

        
    
    return np.argmax(cap_total), cap_per_ue[np.argmax(cap_total),:], cap_total[np.argmax(cap_total)]

