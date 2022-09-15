#import numpy as np
#print('Amplitude Network Results')
#amp_network = np.zeros((4,5))
#SEt train
#amp_network[0,0] =  
#amp_network[0,1] = 
#amp_network[0,2] =
#amp_network[0,3] = 
#amp_network[0,4] =
#SEt train before
#amp_network[1,0] =
#amp_network[1,1] = 
#amp_network[1,2] = 
#amp_network[1,3] =
#amp_network[1,4] =
#SEt val
# amp_network[2,0] =
# amp_network[2,1] =
# amp_network[2,2] =
# amp_network[2,3] =
# amp_network[2,4] =
# # SEt val before
# amp_network[3,0] =
# amp_network[3,1] =
# amp_network[3,2] =
# amp_network[3,3] =
# amp_network[3,4] =
# print(amp_network)
amp_network_mean = amp_network.mean(axis=1)
amp_network_std = amp_network.std(axis=1)
print('{:3.2e}±{:3.2e}'.format(amp_network_mean[0], amp_network_std[0]))
print('{:3.2e}±{:3.2e}'.format(amp_network_mean[1], amp_network_std[1]))
# print('{:3.2e}+/-{:3.2e}'.format(amp_network_mean[2], amp_network_std[2]))
# print('{:3.2e}+/-{:3.2e}'.format(amp_network_mean[3], amp_network_std[3]))
# #
# print('Phase Network Results')
# phi_network = np.zeros((4,5))
# #SEt train
# # phi_network[0,0] =
# # phi_network[0,1] =
# # phi_network[0,2] =
# # phi_network[0,3] =
# # phi_network[0,4] =
# # #SEt train before
# # phi_network[1,0] =
# # phi_network[1,1] =
# # phi_network[1,2] =
# # phi_network[1,3] =
# # phi_network[1,4] =
# #SEt val
# phi_network[2,0] =
# phi_network[2,1] =
# phi_network[2,2] =
# phi_network[2,3] =
# phi_network[2,4] =
# #SEt val before
# phi_network[3,0] =
# phi_network[3,1] =
# phi_network[3,2] =
# phi_network[3,3] =
# phi_network[3,4] =
# # print(phi_network)
# phi_network_mean = phi_network.mean(axis=1)
# phi_network_std = phi_network.std(axis=1)
# print('{:3.2e}+/-{:3.2e}'.format(phi_network_mean[0], phi_network_std[0]))
# print('{:3.2e}+/-{:3.2e}'.format(phi_network_mean[1], phi_network_std[1]))
# print('{:3.2e}+/-{:3.2e}'.format(phi_network_mean[2], phi_network_std[2]))
# print('{:3.2e}+/-{:3.2e}'.format(phi_network_mean[3], phi_network_std[3]))
# #
# import numpy as np
# print('Mismatch Results without Residual')
# mm_network = np.zeros((5,5))
# #MIN
# mm_network[0,0] =
# mm_network[0,1] =
# mm_network[0,2] =
# mm_network[0,3] =
# mm_network[0,4] =
# #Max
# mm_network[1,0] =
# mm_network[1,1] =
# mm_network[1,2] =
# mm_network[1,3] =
# mm_network[1,4] =
# # 99th percentile
# mm_network[2,0] =
# mm_network[2,1] =
# mm_network[2,2] =
# mm_network[2,3] =
# mm_network[2,4] =
# #95 percentile
# mm_network[3,0] =
# mm_network[3,1] =
# mm_network[3,2] =
# mm_network[3,3] =
# mm_network[3,4] =
# #Median
# mm_network[4,0] =
# mm_network[4,1] =
# mm_network[4,2] =
# mm_network[4,3] =
# mm_network[4,4] =
#
#
# mm_network_mean = mm_network.mean(axis=1)
# mm_network_std = mm_network.std(axis=1)
# print('Min: {:3.2e}+/-{:3.2e}'.format(mm_network_mean[0], mm_network_std[0]))
# print('Max: {:3.2e}+/-{:3.2e}'.format(mm_network_mean[1], mm_network_std[1]))
# print('99th: {:3.2e}+/-{:3.2e}'.format(mm_network_mean[2], mm_network_std[2]))
# print('95th: {:3.2e}+/-{:3.2e}'.format(mm_network_mean[3], mm_network_std[3]))
# print('Median: {:3.2e}+/-{:3.2e}'.format(mm_network_mean[4], mm_network_std[4]))

# print('Mismatch Results with Residual')
# mm_res_network = np.zeros((5,5))
# #MIN
# mm_res_network[0,0] =
# mm_res_network[0,1] =
# mm_res_network[0,2] =
# mm_res_network[0,3] =
# mm_res_network[0,4] =
# #Max
# mm_res_network[1,0] =
# mm_res_network[1,1] =
# mm_res_network[1,2] =
# mm_res_network[1,3] =
# mm_res_network[1,4] =
# # 99th percentile
# mm_res_network[2,0] =
# mm_res_network[2,1] =
# mm_res_network[2,2] =
# mm_res_network[2,3] =
# mm_res_network[2,4] =
# #95 percentile
# mm_res_network[3,0] =
# mm_res_network[3,1] =
# mm_res_network[3,2] =
# mm_res_network[3,3] =
# mm_res_network[3,4] =
# #Median
# mm_res_network[4,0] =
# mm_res_network[4,1] =
# mm_res_network[4,2] =
# mm_res_network[4,3] =
# mm_res_network[4,4] =
#
#
# mm_res_network_mean = mm_res_network.mean(axis=1)
# mm_res_network_std = mm_res_network.std(axis=1)
# print('Min: {:3.2e}+/-{:3.2e}'.format(mm_res_network_mean[0], mm_res_network_std[0]))
# print('Max: {:3.2e}+/-{:3.2e}'.format(mm_res_network_mean[1], mm_res_network_std[1]))
# print('99th: {:3.2e}+/-{:3.2e}'.format(mm_res_network_mean[2], mm_res_network_std[2]))
# print('95th: {:3.2e}+/-{:3.2e}'.format(mm_res_network_mean[3], mm_res_network_std[3]))
# print('Median: {:3.2e}+/-{:3.2e}'.format(mm_res_network_mean[4], mm_res_network_std[4]))
