# import numpy as np
# print('Amplitude Network Results')
# amp_network = np.zeros((4,5))
# #SEt TRAINING STATS
# amp_network[0,0] =
# amp_network[0,1] =
# amp_network[0,2] =
# amp_network[0,3] =
# amp_network[0,4] =
# #SEt VALIDATION STATS
# amp_network[1,0] =
# amp_network[1,1] =
# amp_network[1,2] =
# amp_network[1,3] =
# amp_network[1,4] =
# #SEt RESIDUAL STATS
# amp_network[2,0] =
# amp_network[2,1] =
# amp_network[2,2] =
# amp_network[2,3] =
# amp_network[2,4] =
# # mean
# amp_network[3,0] =
# amp_network[3,1] =
# amp_network[3,2] =
# amp_network[3,3] =
# amp_network[3,4] =
# print(amp_network)
# amp_network_mean = amp_network.mean(axis=1)
# amp_network_std = amp_network.std(axis=1)
# print('Mean of train after : ', amp_network_mean[0])
# print('Std of train after : ', amp_network_std[0])
# print('Mean of train before : ', amp_network_mean[1])
# print('Std of train before : ', amp_network_std[1])
# print('Mean of val after: ', amp_network_mean[2])
# print('Std of val after: ', amp_network_std[2])
# print('Mean of val before: ', amp_network_mean[3])
# print('Std of val before: ', amp_network_std[3])
# #
# print('Phase Network Results')
# phi_network = np.zeros((3,5))
# #SEt TRAINING STATS
# phi_network[0,0] =
# phi_network[0,1] =
# phi_network[0,2] =
# phi_network[0,3] =
# phi_network[0,4] =
# #SEt VALIDATION STATS
# phi_network[1,0] =
# phi_network[1,1] =
# phi_network[1,2] =
# phi_network[1,3] =
# phi_network[1,4] =
# #SEt ENS STATS
# phi_network[2,0] =
# phi_network[2,1] =
# phi_network[2,2] =
# phi_network[2,3] =
# phi_network[2,4] =
# # print(phi_network)
# phi_network_mean = phi_network.mean(axis=1)
# phi_network_std = phi_network.std(axis=1)
# print('Mean of Q training: ', phi_network_mean[0])
# print('Std of Q training: ', phi_network_std[0])
# print('Mean of 1/Q training: ', phi_network_mean[1])
# print('Std of 1/Q training: ', phi_network_std[1])
# print('Mean of ENS validation: ', phi_network_mean[2])
# print('Std of ENS validation: ', phi_network_std[2])

# import numpy as np
# print('Mismatch Results')
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
# print('Mean of min : ', mm_network_mean[0])
# print('Std of min: ', mm_network_std[0])
#
# print('Mean of max: ', mm_network_mean[1])
# print('Std of max: ', mm_network_std[1])
#
# print('Mean of 99th: ', mm_network_mean[2])
# print('Std of 99th: ', mm_network_std[2])
#
# print('Mean of 95th: ', mm_network_mean[3])
# print('Std of 95th: ', mm_network_std[3])
#
# print('Mean of Median: ', mm_network_mean[4])
# print('Std of Median: ', mm_network_std[4])