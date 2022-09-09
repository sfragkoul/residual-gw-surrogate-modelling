import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from pycbc.psd import analytical
from pycbc.pnutils import f_FRD

from time import time
start = time()

G = 6.67408e-11
c = 2.99792458e8
MSUN = 1.9891e30
TIME = G * MSUN / c ** 3

Mtot = 60
delta_t_05M = 0.5 * Mtot * TIME
flen = 48677
duration = flen * delta_t_05M
delta_f = 1.0 / duration
# delta_f_psd = 1 / 16
# flen_psd = int(1024 / delta_f) + 1
ligopsd = analytical.from_string('aLIGOaLIGODesignSensitivityT1800044',
                                  length=flen,
                                  delta_f=delta_f,
                                  low_freq_cutoff=15.)

import os
import pickle
import numpy as np
import rompy as rp
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter.matchedfilter import match

# print(ligopsd.data)
# ligopsd = ligopsd.trim_zeros()
# eps = 1e-52
# ligopsd += eps

# load val waveforms & val EIM data
subset_name = 'val'
if subset_name == 'val':
    amp_filenames = ['q1to8_s0.99_both/amp/{}.pkl'.format(idx) for idx in [21, 22, 23]]
else:
    amp_filenames = ['q1to8_s0.99_both/amp/{}.pkl'.format(idx) for idx in [24, 25, 26]]  # test

if subset_name == 'val':
    phi_filenames = ['q1to8_s0.99_both/phase/{}.pkl'.format(idx) for idx in [21, 22, 23]]
else:
    phi_filenames = ['q1to8_s0.99_both/phase/{}.pkl'.format(idx) for idx in [24, 25, 26]]  # test

print('Loading 1 to 10000...')
n_files = len(amp_filenames)
with open(amp_filenames[0], 'rb') as f:
    [lambda_values, amp_ro] = pickle.load(f)
print(lambda_values[0, :])
all_lambda_values = np.zeros_like(lambda_values, shape=(n_files * 10000, lambda_values.shape[1]))
all_amp_ro = np.zeros_like(amp_ro, shape=(n_files * 10000, amp_ro.shape[1]), dtype=np.complex64)
all_lambda_values[:10000, :] = lambda_values
all_amp_ro[:10000, :] = amp_ro.astype(np.complex64)
del lambda_values, amp_ro

for idx in range(1, n_files):
    print('Loading {} to {}...'.format(idx * 10000 + 1, (idx + 1) * 10000))
    with open(amp_filenames[idx], 'rb') as f:
        [lambda_values, amp_ro] = pickle.load(f)
    print(lambda_values[0, :])
    all_lambda_values[idx * 10000:(idx + 1) * 10000, :] = lambda_values
    all_amp_ro[idx * 10000:(idx + 1) * 10000, :] = amp_ro
    del lambda_values, amp_ro
n_val = all_amp_ro.shape[0]


print('Loading 1 to 10000...')
n_files = len(phi_filenames)
with open(phi_filenames[0], 'rb') as f:
    [lambda_values, phi_ro] = pickle.load(f)
print(lambda_values[0, :])
all_lambda_values = np.zeros_like(lambda_values, shape=(n_files * 10000, lambda_values.shape[1]))
all_phi_ro = np.zeros_like(phi_ro, shape=(n_files * 10000, phi_ro.shape[1]), dtype=np.complex64)
all_lambda_values[:10000, :] = lambda_values
all_phi_ro[:10000, :] = phi_ro.astype(np.complex64)
del lambda_values, phi_ro

for idx in range(1, n_files):
    print('Loading {} to {}...'.format(idx * 10000 + 1, (idx + 1) * 10000))
    with open(phi_filenames[idx], 'rb') as f:
        [lambda_values, phi_ro] = pickle.load(f)
    print(lambda_values[0, :])
    all_lambda_values[idx * 10000:(idx + 1) * 10000, :] = lambda_values
    all_phi_ro[idx * 10000:(idx + 1) * 10000, :] = phi_ro
    del lambda_values, phi_ro
n_val = all_phi_ro.shape[0]

# SURROGATE

# all_h_ro = all_amp_ro * np.exp(1j * all_phi_ro)
times = np.arange(all_amp_ro.shape[1]) * delta_t_05M
integration = rp.Integration([times.min(), times.max()],
                             num=all_amp_ro.shape[1], rule='trapezoidal')
duration = all_amp_ro.shape[1] * delta_t_05M
delta_f = 1.0 / duration


# print(analytical.get_psd_model_list())
nz_inds = np.where(ligopsd.data > 0)[0]
# amp_tol = 1e-08
# phi_tol = 1e-08
tols = [1e-10]
for tol in tols:
    amp_tol = tol
    phi_tol = tol
    # print('Amp tol: {}, phi tol: {}', amp_tol, phi_tol)
    with open('q1to8_s0.99_both/amp_rel_sur/tol_{}_{}.pkl'.format(amp_tol, subset_name), 'rb') as f:
        [all_lambda_values_eim, amp_eim_coeffs, amp_eim_basis, amp_eim_indices] = pickle.load(f)
    with open('q1to8_s0.99_both/phi_rel_sur/tol_{}_{}.pkl'.format(phi_tol, subset_name), 'rb') as f:
        [all_lambda_values_eim, phi_eim_coeffs, phi_eim_basis, phi_eim_indices] = pickle.load(f)

    print('Amp coeffs: ', len(amp_eim_indices))
    print('Phi coeffs: ', len(phi_eim_indices))

    with open('Amplitude_nn_predictions#5.pkl', 'rb') as f:
        [amp_val_final_predictions, amp_val_final_predictions_with_res] = pickle.load(f)
    #
    with open('Phase_nn_predictions#5.pkl', 'rb') as f:
        [phi_val_final_predictions, phi_val_final_predictions_with_res] = pickle.load(f)

    mm = np.zeros((30000,))
    for i in range(30000):
        print(i)
        amp_rec = amp_val_final_predictions[i, :].dot(amp_eim_basis)
        phi_rec = phi_val_final_predictions[i, :].dot(phi_eim_basis)
        # amp_rec = amp_eim_coeffs[i, :].dot(amp_eim_basis)
        # phi_rec = phi_eim_coeffs[i, :].dot(phi_eim_basis)
        h = all_amp_ro[i, :] * np.exp(1j * all_phi_ro[i, :])
        h_s = amp_rec * np.exp(1j * phi_rec)
        print('h, hs made')
        # ht = TimeSeries(h, delta_t=delta_t_05M)
        # ht_s = TimeSeries(h_s, delta_t=delta_t_05M)
        # print('timeseries made')
        # delta_f = 1.0 / duration
        hf = FrequencySeries(np.fft.fft(h), delta_f=delta_f)
        hf_s = FrequencySeries(np.fft.fft(h_s), delta_f=delta_f)
        print('frequencyseries made')
        mass_ratio = all_lambda_values_eim[i, 0]
        mass_1 = (mass_ratio * Mtot)/ (1+ mass_ratio)
        mass_2 = Mtot - mass_1
        high_freq_cutoff = 1.4 * f_FRD (mass_1, mass_2)
        m, _ = match(hf, hf_s, psd=ligopsd, low_frequency_cutoff=15.,  high_frequency_cutoff=high_freq_cutoff)
        # m, _ = match(hf, hf_s, low_frequency_cutoff=15., high_frequency_cutoff=high_freq_cutoff)
        mm[i] = 1 - m
        print(1 - m)
    print('Tol amp: {} phase: {}'.format(amp_tol, phi_tol))
    print('Mismatch min: {:3.2e}, max: {:3.2e}, avg: {:3.2e}'.format(min(mm),
                                                                     max(mm),
                                                                     sum(mm) / len(mm)))

    # plt.show()

mm99=np.percentile(mm, q=99)
print("99 Percentile: ", mm99)

mm95=np.percentile(mm, q=95)
print("95 Percentile: ", mm95)

mm50=np.percentile(mm, q=50)
print("Median: ", mm50)
#
#
mm_plot=[]
for j in mm:
    if j > mm95:
        mm_plot.append(j)

mm_plot_np = np.array(mm_plot)

mask=(mm>mm95)

plt.figure()
# plt.title('Pycbc validation mismatch for tol e-06', fontsize='x-large' )
plt.subplot(221)
# plt.title('Pycbc validation mismatch for tol e-16', fontsize='x-large' )
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
plt.savefig('Pycbc validation mismatch for tol{} y space no res #5'.format(tols))
# plt.savefig('Pycbc validation mismatch for tol{} -y space no res #1'.format(tols))
# plt.savefig('Pycbc validation mismatch for tol{} ens no res #5'.format(tols))
plt.show()
#
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(all_lambda_values_eim[mask, 1],all_lambda_values_eim[mask, 2], all_lambda_values_eim[mask, 0], c=mm_plot,  cmap='plasma' );
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$q$')
# # plt.title('Pycbc validation mismatch for tol%' %tols)
# # plt.savefig('Pycbc validation mismatch for tol{}'.format(tols))
# plt.show()

print(' mismatch script time in minutes:', (time()-start)/60)

