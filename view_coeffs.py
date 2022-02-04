import pickle
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class SEOBNRv4AmpPhase(Dataset):
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)

        self.x = lambda_values
        self.y = coeffs
        self.basis = eim_basis
        self.n_coeffs = eim_indices.shape[0]

    def __getitem__(self, item):
        return self.x[item, :], self.y[item, :]

    def __len__(self):
        return self.x.shape[0]


# train_dataset_amp = SEOBNRv4AmpPhase(filename='q1to8_s0.99_both/amp_rel_sur/tol_1e-10.pkl')
train_dataset_phi = SEOBNRv4AmpPhase(filename='q1to8_s0.99_both/phi_rel_sur/tol_1e-10.pkl')

cmap = matplotlib.cm.get_cmap('plasma')
#
# fig = plt.figure(figsize=(14, 4))
# ax = fig.add_subplot(131, projection='3d')
# sc = ax.scatter(train_dataset_amp.x[::30, 1], train_dataset_amp.x[::30, 2], train_dataset_amp.y[::30, 15],
#             c=train_dataset_amp.x[::30, 0], cmap=cmap)
# # plt.suptitle('Amplitude 1st coefficient')
# ax.set_ylabel(r'$\chi_2$')
# ax.set_xlabel(r'$\chi_1$')
#
# ax = fig.add_subplot(132, projection='3d')
# sc = ax.scatter(train_dataset_amp.x[::30, 1], train_dataset_amp.x[::30, 2], train_dataset_amp.y[::30, 16],
#             c=train_dataset_amp.x[::30, 0], cmap=cmap)
# # plt.suptitle('Amplitude 2nd coefficient')
# ax.set_ylabel(r'$\chi_2$')
# ax.set_xlabel(r'$\chi_1$')
# # plt.savefig('amp_coeff_2.pdf', bbox_inches='tight')
#
# ax = fig.add_subplot(133, projection='3d')
# sc = ax.scatter(train_dataset_amp.x[::30, 1], train_dataset_amp.x[::30, 2], train_dataset_amp.y[::30, 17],
#             c=train_dataset_amp.x[::30, 0], cmap=cmap)
# # plt.suptitle('Amplitude 3rd coefficient')
# ax.set_ylabel(r'$\chi_2$')
# ax.set_xlabel(r'$\chi_1$')
# # plt.savefig('amp_coeff_3.pdf', bbox_inches='tight')
#
# ax.set_ylabel(r'$\chi_2$')
# ax.set_xlabel(r'$\chi_1$')
#
# # cbar = plt.colorbar(sc)
# cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8])
# cbar = plt.colorbar(sc, cax=cbaxes)
# # plt.title('Phase coefficients')
# cbar.set_label(r'$q$', rotation=0)


# plt.subplots_adjust(wspace=0, hspace=0)
# plt.suptitle('Amplitude coefficients')
# plt.savefig('amp_coeffs_15-17.pdf', bbox_inches='tight')





# PHASE
fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(131, projection='3d')
sc = ax.scatter(train_dataset_phi.x[::30, 1], train_dataset_phi.x[::30, 2], train_dataset_phi.y[::30, 5],
            c=train_dataset_phi.x[::30, 0], cmap=cmap)
ax.set_ylabel(r'$\chi_2$')
ax.set_xlabel(r'$\chi_1$')
# plt.suptitle('Phase 1st coefficient')

ax = fig.add_subplot(132, projection='3d')
sc = ax.scatter(train_dataset_phi.x[::30, 1], train_dataset_phi.x[::30, 2], train_dataset_phi.y[::30, 6],
            c=train_dataset_phi.x[::30, 0], cmap=cmap)
# plt.suptitle('Phase 2nd coefficient')
ax.set_ylabel(r'$\chi_2$')
ax.set_xlabel(r'$\chi_1$')
# plt.savefig('phi_coeff_2.pdf', bbox_inches='tight')

ax = fig.add_subplot(133, projection='3d')
sc = ax.scatter(train_dataset_phi.x[::30, 1], train_dataset_phi.x[::30, 2], train_dataset_phi.y[::30, 7],
            c=train_dataset_phi.x[::30, 0], cmap=cmap)
# plt.suptitle('Phase 3rd coefficient')
ax.set_ylabel(r'$\chi_2$')
ax.set_xlabel(r'$\chi_1$')
# plt.savefig('phi_coeff_3.pdf', bbox_inches='tight')

# plt.ylabel(r'$\chi_2$')
# plt.xlabel(r'$\chi_1$')

# cbar = plt.colorbar(sc)
cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8])
cbar = plt.colorbar(sc, cax=cbaxes)
# plt.title('Phase coefficients')
cbar.set_label(r'$q$', rotation=0)


plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle('Phase coefficients')
plt.savefig('phi_coeffs_5-7.pdf', bbox_inches='tight')

# fig = plt.figure(figsize=(4, 4))


# fig = plt.figure(figsize=(4, 4))


# fig = plt.figure(figsize=(4, 4))


# fig = plt.figure(figsize=(4, 4))


plt.show()
