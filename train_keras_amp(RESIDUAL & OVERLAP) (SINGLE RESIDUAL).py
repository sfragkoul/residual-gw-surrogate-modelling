import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)

import pickle
import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(seed)
rn.seed(seed)
tf.random.set_seed(seed)

# #put the these lines before importing any module from keras.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import backend
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from tensorflow.keras import initializers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from time import time
start = time()


# monitor the learning rate
class LearningRateMonitor(Callback):
    # start of training
    def on_train_begin(self, logs={}):
        self.lrates = list()

    # end of each training epoch
    def on_epoch_end(self, epoch, logs={}):
        # get and store the learning rate
        lrate = float(backend.get_value(self.model.optimizer.lr))
        self.lrates.append(lrate)



with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10.pkl', 'rb') as f:
    [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)


# SPLITTING TO K-NETWORKS
# K=1 for q [1, 4)
mask_train_k1 = (lambda_values[:,0] < 4.2)
lambda_values_k1 = lambda_values[mask_train_k1]

# K=2 for q [4,8]
mask_train_k2 = (lambda_values[:,0] >= 3.8)
lambda_values_k2 = lambda_values[mask_train_k2]


#TRAINING DATASET
# K=1 Network
lambda_values_k1[:, 0] = np.log10(lambda_values_k1[:, 0])
scaler_x_k1 = StandardScaler()

lambda_values_k1_scaled = scaler_x_k1.fit_transform(lambda_values_k1)
amp_train_x_k1 = lambda_values_k1_scaled
amp_train_y_k1 = coeffs[mask_train_k1,:]


ind_train_k1_first = np.where(lambda_values[:, 0] < 3.8)[0]
ind_train_overlap = np.where((lambda_values[:, 0] >= 3.8) & (lambda_values[:, 0] < 4.2))[0]
ind_train_k2_first = np.where(lambda_values[:, 0] >= 4.2)[0]

print(amp_train_x_k1.shape)
print(amp_train_y_k1.shape)

# K=2 Network
lambda_values_k2[:, 0] = np.log10(lambda_values_k2[:, 0])
scaler_x_k2 = StandardScaler()
lambda_values_k2_scaled = scaler_x_k2.fit_transform(lambda_values_k2)
amp_train_x_k2 = lambda_values_k2_scaled
amp_train_y_k2 = coeffs[mask_train_k2,:]

print(amp_train_x_k2.shape)
print(amp_train_y_k2.shape)

amp_train_x = np.zeros((200000,3))
amp_train_x[mask_train_k1,:] = amp_train_x_k1
amp_train_x[mask_train_k2,:] = amp_train_x_k2

# VALIDATION DATASET
with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)

amp_val_y_all = coeffs_val

# K=1 for q [1, 4)
mask_val_k1 = (lambda_values_val[:,0] < 4.2)
lambda_values_val_k1 = lambda_values_val[mask_val_k1]
coeffs_val_k1 = coeffs_val[mask_val_k1, :]

ind_val_k1_first = np.where(lambda_values_val[:, 0] < 3.8)[0]
ind_val_overlap = np.where((lambda_values_val[:, 0] >= 3.8) & (lambda_values_val[:, 0] < 4.2))[0]


lambda_values_val_k1_log = lambda_values_val_k1.copy()
lambda_values_val_k1_log[:, 0] = np.log10(lambda_values_val_k1_log[:, 0])
lambda_values_val_k1_scaled = scaler_x_k1.transform(lambda_values_val_k1_log)
amp_val_x_k1 = lambda_values_val_k1_scaled
amp_val_y_k1 = coeffs_val[mask_val_k1,:]

print(amp_val_x_k1.shape)
print(amp_val_y_k1.shape)

# K=2 for q [4,8]
mask_val_k2 = (lambda_values_val[:,0] >= 3.8)
lambda_values_val_k2 = lambda_values_val[mask_val_k2]

ind_val_k2_first = np.where(lambda_values_val[:, 0] >= 4.2)[0]


lambda_values_val_k2_log = lambda_values_val_k2.copy()
lambda_values_val_k2_log[:, 0] = np.log10(lambda_values_val_k2_log[:, 0])
lambda_values_val_k2_scaled = scaler_x_k2.transform(lambda_values_val_k2_log)
amp_val_x_k2 = lambda_values_val_k2_scaled
amp_val_y_k2 = coeffs_val[mask_val_k2,:]

print(amp_val_x_k2.shape)
print(amp_val_y_k2.shape)

amp_val_x = np.zeros((30000,3))
amp_val_x[mask_val_k1,:] = amp_val_x_k1
amp_val_x[mask_val_k2,:] = amp_val_x_k2


# K=1 NETWORK

amp_model_k1 = Sequential()
amp_model_k1.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k1.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k1.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k1.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k1.add(Dense(18, activation=None))

# compile model
opt_k1 = Adam(lr=0.001)
amp_model_k1.compile(loss='mse', optimizer=opt_k1, metrics=['mse'])

amp_rlrp_k1 = ReduceLROnPlateau(monitor='mse', factor=0.9 , patience=15)
amp_lrm_k1 = LearningRateMonitor()
amp_stop_k1 = EarlyStopping(monitor='mse', patience=1000)
amp_history_k1 = amp_model_k1.fit(amp_train_x_k1, amp_train_y_k1, validation_data=(amp_val_x_k1, amp_val_y_k1),
                            epochs=1000, batch_size=256,
                            verbose=1, callbacks=[amp_rlrp_k1, amp_lrm_k1, amp_stop_k1])


#thelw kai predictions apo validaion
amp_val_predictions_k1 = amp_model_k1.predict(amp_val_x_k1)
amp_val_mse_k1 = np.square(amp_val_y_k1-amp_val_predictions_k1).mean()
print("Amplitude validation before residual mse K=1: ", amp_val_mse_k1)



# K=2 NETWORK

amp_model_k2 = Sequential()
amp_model_k2.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k2.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k2.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k2.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model_k2.add(Dense(18, activation=None))

# compile model
opt_k2 = Adam(lr=0.001)
amp_model_k2.compile(loss='mse', optimizer=opt_k2, metrics=['mse'])

amp_rlrp_k2 = ReduceLROnPlateau(monitor='mse', factor=0.9 , patience=15)
amp_lrm_k2 = LearningRateMonitor()
amp_stop_k2 = EarlyStopping(monitor='mse', patience=1000)
amp_history_k2 = amp_model_k2.fit(amp_train_x_k2, amp_train_y_k2, validation_data=(amp_val_x_k2, amp_val_y_k2),
                            epochs=1000, batch_size=256,
                            verbose=1, callbacks=[amp_rlrp_k2, amp_lrm_k2, amp_stop_k2])



amp_val_predictions_k2 = amp_model_k2.predict(amp_val_x_k2)
amp_val_mse_k2 = np.square(amp_val_y_k2 - amp_val_predictions_k2).mean()
print("Amplitude validation  before residual mse K=2: ", amp_val_mse_k2)

amp_final_mse = (amp_val_mse_k1 + amp_val_mse_k2)/2
print("AMplitude final mse  before residual: ", amp_final_mse)


lambda_train_overlap = lambda_values[ind_train_overlap]
lambda_train_overlap_log = lambda_train_overlap.copy()
lambda_train_overlap_log[: ,0] = np.log10(lambda_train_overlap_log[: ,0])
lambda_overlap_log_train_scaled_k1 = scaler_x_k1.transform(lambda_train_overlap_log)
lambda_overlap_log_train_scaled_k2 = scaler_x_k2.transform(lambda_train_overlap_log)


amp_train_predictions_k1 = amp_model_k1.predict(amp_train_x[ind_train_k1_first])
amp_train_predictions_o1 = amp_model_k1.predict(lambda_overlap_log_train_scaled_k1)

amp_train_predictions_k2 = amp_model_k2.predict(amp_train_x[ind_train_k2_first])
amp_train_predictions_o2 = amp_model_k2.predict(lambda_overlap_log_train_scaled_k2)

amp_train_predictions_overlap = (amp_train_predictions_o1 + amp_train_predictions_o2)/2

amp_train_final_predictions = np.zeros((200000,18))
amp_train_final_predictions[ind_train_k1_first,:] = amp_train_predictions_k1
amp_train_final_predictions[ind_train_overlap,:] = amp_train_predictions_overlap
amp_train_final_predictions[ind_train_k2_first,:] = amp_train_predictions_k2




#SAVE VALIDATION PREDICTIONS
lambda_overlap = lambda_values_val[ind_val_overlap]
lambda_overlap_log = lambda_overlap.copy()
lambda_overlap_log[: ,0] = np.log10(lambda_overlap_log[: ,0])
lambda_overlap_log_scaled_k1 = scaler_x_k1.transform(lambda_overlap_log)
lambda_overlap_log_scaled_k2 = scaler_x_k2.transform(lambda_overlap_log)


amp_val_predictions_k1 = amp_model_k1.predict(amp_val_x[ind_val_k1_first])
amp_val_predictions_o1 = amp_model_k1.predict(lambda_overlap_log_scaled_k1)

amp_val_predictions_k2 = amp_model_k2.predict(amp_val_x[ind_val_k2_first])
amp_val_predictions_o2 = amp_model_k2.predict(lambda_overlap_log_scaled_k2)

amp_val_predictions_overlap = (amp_val_predictions_o1 + amp_val_predictions_o2)/2

amp_val_final_predictions = np.zeros((30000,18))
amp_val_final_predictions[ind_val_k1_first,:] = amp_val_predictions_k1
amp_val_final_predictions[ind_val_overlap,:] = amp_val_predictions_overlap
amp_val_final_predictions[ind_val_k2_first,:] = amp_val_predictions_k2


amp_train_final_predictions = np.float64(amp_train_final_predictions)
amp_errors = coeffs - amp_train_final_predictions


amp_val_final_predictions = np.float64(amp_val_final_predictions)
amp_errors_val = coeffs_val - amp_val_final_predictions


# input for the residual error network
amp_res_scaler = MinMaxScaler()
amp_errors_scaled = amp_res_scaler.fit_transform(amp_errors)
amp_errors_val_scaled = amp_res_scaler.transform(amp_errors_val)


# Residual Error
amp_res_model = Sequential()
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(18, activation=None))


opt_res = Adam(lr=0.001)
amp_res_model.compile(loss='mse', optimizer=opt_res, metrics=['mse'])
amp_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=50)
amp_res_lrm = LearningRateMonitor()
amp_res_stop = EarlyStopping(monitor='mse', patience=50)
amp_res_history = amp_res_model.fit(amp_train_x, amp_errors_scaled, validation_data=(amp_val_x, amp_errors_val_scaled),
                                    epochs=50, batch_size=256,
                                    verbose=1,
                                    callbacks=[amp_res_rlrp, amp_res_lrm, amp_res_stop])

# Plot training & validation accuracy values
plt.figure()
plt.plot(amp_res_history.history['mse'])
plt.plot(amp_res_history.history['val_mse'])
plt.yscale("log")
plt.title('Residual Amp Model MSE')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('Fig3: Amp Res Model Train MSE')
plt.show()

# Learning Rate
# plt.figure()
# plt.plot(amp_res_lrm.lrates)
# plt.yscale("log")
# plt.title('Amp Residual Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig4: Amp Res Model Learning Rate')
# plt.show()
#
# # FINAL MSE
#
# predictions from residual and reverse transform to the INITIAL data space
amp_train_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_train_x))
amp_val_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_val_x))

amp_train_predictions_corrected = amp_train_final_predictions + amp_train_y_error
amp_val_final_predictions_with_res = amp_val_final_predictions + amp_val_y_error

amp_final_train_mse = np.square(amp_train_predictions_corrected - coeffs).mean()
amp_final_val_mse = np.square(amp_val_final_predictions_with_res - coeffs_val).mean()
print('Amplitude')
print("Amplitude validation before residual mse K=1: ", amp_val_mse_k1)
print("Amplitude validation  before residual mse K=2: ", amp_val_mse_k2)
print("AMplitude final mse  before residual: ", amp_final_mse)
print('Validation mse after residual: ',  amp_final_val_mse)

# SAVE PREDICTIONS FOR MISMATCH SCRIPT
with open('Amplitude_nn_predictions#5.pkl', 'wb') as f:
    pickle.dump([amp_val_final_predictions, amp_val_final_predictions_with_res], f)

print('minutes:', (time()-start)/60)