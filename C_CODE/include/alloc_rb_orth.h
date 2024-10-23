#include <iostream>
#include <armadillo>
// Function declaration
std::pair<std::vector<int>, double> alloc_rb(std::vector<std::vector<int>>& group_list, const int SEL_UE, const std::vector<int>& Num_UE_ps, int sel_sl, int num_rb, int tti, const arma::cx_mat& H, int ue_index, std::vector<int>& sl_ue_list, arma::vec& cg_ue_vec);
