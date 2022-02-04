import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import pickle
import numpy as np
import matplotlib.pyplot as plt



with open('Train Mismatch/Train_Mismatch_200k.pkl', 'rb') as f:
    [mm_final] = pickle.load(f)



mm99=np.percentile(mm_final, q=99)
print("99 Percentile: ", mm99)

mm95=np.percentile(mm_final, q=95)
print("95 Percentile: ", mm95)

mm50=np.percentile(mm_final, q=50)
print("Median: ", mm50)
#
print('Min:',mm_final.min())
print('Max:',mm_final.max())
#
mm_plot=[]
for i in mm_final:
    if i > mm95:
        mm_plot.append(i)

mm_plot_np = np.array(mm_plot)

mask=(mm_final>mm95)
ind_mask = np.where(mm_final>mm95)[0]

with open('train_augmentation.pkl', 'wb') as f:
    pickle.dump([mask, ind_mask], f)


with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_train.pkl', 'rb') as f:
    [all_lambda_values_eim, amp_eim_coeffs, amp_eim_basis, amp_eim_indices] = pickle.load(f)

plt.figure()
plt.subplot(221)
plt.scatter(all_lambda_values_eim[mask, 0], all_lambda_values_eim[mask, 1],s=5, c=mm_plot , cmap='plasma')
plt.ylabel('$x_1$')
plt.xlabel('$q$')
plt.subplot(222)
plt.scatter(all_lambda_values_eim[mask, 0], all_lambda_values_eim[mask, 2],s=5, c=mm_plot, cmap='plasma')
plt.xlabel('$q$')
plt.ylabel('$x_2$')
plt.subplot(223)
plt.scatter(all_lambda_values_eim[mask, 1], all_lambda_values_eim[mask, 2],s=5, c=mm_plot, cmap='plasma')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
cabr=plt.colorbar(orientation='vertical')
# plt.clim(6e-08, 10e-08)
plt.tight_layout()
plt.savefig('train mismatch')
plt.show()
#
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(all_lambda_values_eim[mask, 1],all_lambda_values_eim[mask, 2], all_lambda_values_eim[mask, 0], c=mm_plot,  cmap='plasma' );
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$q$')
# fig.colorbar(matplotlib.cm.ScalarMappable( cmap='plasma'), orientation='vertical', ax=ax)
plt.show()