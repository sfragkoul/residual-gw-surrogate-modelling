import os

import keras.models

os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)

import pickle
import numpy as np
import tensorflow as tf
import random as rn
from time import time
start = time()

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
from keras.optimizers import Adam, SGD, Adamax
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from tensorflow.keras import initializers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


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



with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10.pkl', 'rb') as f:
    [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)

x=coeffs
scaler_y = MinMaxScaler()
coeffs_scaled = scaler_y.fit_transform(x)

# SPLITTING TO K-NETWORKS
#TRAINING DATASET
# K=1 for q [1, 4)
mask_train_k1 = (lambda_values[:,0] < 4.2)
lambda_values_k1 = lambda_values[mask_train_k1]
coeffs_k1 = coeffs_scaled[mask_train_k1,:]

# K=2 for q [4,8]
mask_train_k2 = (lambda_values[:,0] >= 3.8)
lambda_values_k2 = lambda_values[mask_train_k2]
coeffs_k2 = coeffs_scaled[mask_train_k2,:]

ind_train_k1_first = np.where(lambda_values[:, 0] < 3.8)[0]
ind_train_overlap = np.where((lambda_values[:, 0] >= 3.8) & (lambda_values[:, 0] < 4.2))[0]
ind_train_k2_first = np.where(lambda_values[:, 0] >= 4.2)[0]

# K=1 Network
lambda_values_k1[:, 0] = np.log10(lambda_values_k1[:, 0])
scaler_x_k1 = StandardScaler()
lambda_values_k1_scaled = scaler_x_k1.fit_transform(lambda_values_k1)
phi_train_x_k1 = lambda_values_k1_scaled
phi_train_y_k1 = coeffs_k1

print(phi_train_x_k1.shape)
print(phi_train_y_k1.shape)

# K=2 Network
lambda_values_k2[:, 0] = np.log10(lambda_values_k2[:, 0])
scaler_x_k2 = StandardScaler()


lambda_values_k2_scaled = scaler_x_k2.fit_transform(lambda_values_k2)
phi_train_x_k2 = lambda_values_k2_scaled
coeffs_k2_scaled = coeffs_k2
phi_train_y_k2 = coeffs_k2_scaled

print(phi_train_x_k2.shape)
print(phi_train_y_k2.shape)

phi_train_x = np.zeros((200000,3))
phi_train_x[mask_train_k1,:] = phi_train_x_k1
phi_train_x[mask_train_k2,:] = phi_train_x_k2


# VALIDATION DATASET
with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)

x_val = coeffs_val
coeffs_val_scaled = scaler_y.transform(x_val)

# K=1 for q [1, 4)
mask_val_k1 = (lambda_values_val[:,0] < 4.2)
lambda_values_val_k1 = lambda_values_val[mask_val_k1]
coeffs_val_k1 = coeffs_val_scaled[mask_val_k1,:]

ind_val_k1_first = np.where(lambda_values_val[:,0] < 3.8)[0]
ind_val_overlap = np.where((lambda_values_val[:,0] >= 3.8) & (lambda_values_val[:,0] < 4.2))[0]

lambda_values_val_k1_log = lambda_values_val_k1.copy()
lambda_values_val_k1_log[:, 0] = np.log10(lambda_values_val_k1_log[:, 0])
lambda_values_val_k1_scaled = scaler_x_k1.transform(lambda_values_val_k1_log)
phi_val_x_k1 = lambda_values_val_k1_scaled
phi_val_y_k1 = coeffs_val_k1

print(phi_val_x_k1.shape)
print(phi_val_y_k1.shape)


# K=2 for q [4,8]
mask_val_k2 = (lambda_values_val[:,0] >= 3.8)
lambda_values_val_k2 = lambda_values_val[mask_val_k2]
coeffs_val_k2 = coeffs_val_scaled[mask_val_k2,:]

ind_val_k2_first = np.where(lambda_values_val[:,0] >= 4.2)[0]

lambda_values_val_k2_log = lambda_values_val_k2.copy()
lambda_values_val_k2_log[:, 0] = np.log10(lambda_values_val_k2_log[:, 0])
lambda_values_val_k2_scaled = scaler_x_k2.transform(lambda_values_val_k2_log)
phi_val_x_k2 = lambda_values_val_k2_scaled
phi_val_y_k2 = coeffs_val_k2

print(phi_val_x_k2.shape)
print(phi_val_y_k2.shape)

phi_val_x = np.zeros((30000,3))
phi_val_x[mask_val_k1,:] = phi_val_x_k1
phi_val_x[mask_val_k2,:] = phi_val_x_k2


# K=1 NETWORK
phi_model_k1 = Sequential()
phi_model_k1.add(Dense(320, activation= 'softplus'))
phi_model_k1.add(Dense(320, activation='softplus'))
phi_model_k1.add(Dense(320, activation='softplus'))
phi_model_k1.add(Dense(320, activation='softplus'))
phi_model_k1.add(Dense(8, activation=None))

# compile model
opt_k1 = Adamax(lr=0.01)
phi_model_k1.compile(loss='mse', optimizer=opt_k1, metrics=['mse'])
phi_rlrp_k1 = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
phi_lrm_k1 = LearningRateMonitor()
phi_stop_k1 = EarlyStopping(monitor='mse', patience=1000)
phi_history_k1 = phi_model_k1.fit(phi_train_x_k1, phi_train_y_k1, validation_data=(phi_val_x_k1, phi_val_y_k1),
                            epochs=1000, batch_size=256,
                            verbose=1, callbacks=[phi_rlrp_k1, phi_lrm_k1, phi_stop_k1])

# train_evaluation_k1 = phi_model_k1.evaluate(phi_train_x_k1, phi_train_y_k1, batch_size=1000)
phi_model_k1.save('phi_model_k1', save_format='h5')
# phi_model_k1 = tf.keras.models.load_model('phi_model_k1')

phi_val_norm_k1=np.square(phi_val_y_k1 - phi_model_k1.predict(phi_val_x_k1))
print('Phase K=1 Validation before residual mse:',phi_val_norm_k1.mean())


# K=2 NETWORK
# constructing the model
phi_model_k2 = Sequential()
phi_model_k2.add(Dense(320, activation='softplus'))
phi_model_k2.add(Dense(320, activation='softplus'))
phi_model_k2.add(Dense(320, activation='softplus'))
phi_model_k2.add(Dense(320, activation='softplus'))
phi_model_k2.add(Dense(8, activation=None))

# compile model
opt_k2 = Adamax(lr=0.01)
phi_model_k2.compile(loss='mse', optimizer=opt_k2, metrics=['mse'])
phi_rlrp_k2 = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
phi_lrm_k2 = LearningRateMonitor()
phi_stop_k2 = EarlyStopping(monitor='mse', patience=1000)
phi_history_k2 = phi_model_k2.fit(phi_train_x_k2, phi_train_y_k2, validation_data=(phi_val_x_k2, phi_val_y_k2),
                            epochs=1000, batch_size=256,
                            verbose=1, callbacks=[phi_rlrp_k2, phi_lrm_k2, phi_stop_k2])

phi_model_k2.save('phi_model_k2', save_format='h5')
# phi_model_k2 = tf.keras.models.load_model('phi_model_k2')

phi_val_norm_k1=np.square(phi_val_y_k1 - phi_model_k1.predict(phi_val_x_k1))
phi_val_norm_k1_mse = phi_val_norm_k1.mean()
print('Phase K=1 Validation before res mse:',phi_val_norm_k1_mse)

phi_val_norm_k2=np.square(phi_val_y_k2 - phi_model_k2.predict(phi_val_x_k2))
phi_val_norm_k2_mse = phi_val_norm_k2.mean()
print('Phase K=2 Validation before res  mse:',phi_val_norm_k2_mse)


# phi_val_final_predictions = np.vstack([phi_val_norm_k1, phi_val_norm_k2])
phi_final_mse = (phi_val_norm_k1_mse + phi_val_norm_k2_mse)/2
print("Phase final validation before residual mse: ", phi_final_mse)

#train overlap section
lambda_train_overlap = lambda_values[ind_train_overlap]
lambda_train_overlap_log = lambda_train_overlap.copy()
lambda_train_overlap_log[: ,0] = np.log10(lambda_train_overlap_log[: ,0])
lambda_overlap_log_train_scaled_k1 = scaler_x_k1.transform(lambda_train_overlap_log)
lambda_overlap_log_train_scaled_k2 = scaler_x_k2.transform(lambda_train_overlap_log)

#predictions from 3 areas
phi_train_predictions_k1_scaled = phi_model_k1.predict(phi_train_x[ind_train_k1_first])
phi_train_predictions_o1_scaled = phi_model_k1.predict(lambda_overlap_log_train_scaled_k1)
phi_train_predictions_k2_scaled = phi_model_k2.predict(phi_train_x[ind_train_k2_first])
phi_train_predictions_o2_scaled = phi_model_k2.predict(lambda_overlap_log_train_scaled_k2)
phi_train_predictions_overlap_scaled = (phi_train_predictions_o1_scaled + phi_train_predictions_o2_scaled)/2

#train predictions into final array
phi_train_final_predictions_scaled = np.zeros((200000,8))
phi_train_final_predictions_scaled[ind_train_k1_first,:] = phi_train_predictions_k1_scaled
phi_train_final_predictions_scaled[ind_train_overlap,:] = phi_train_predictions_overlap_scaled
phi_train_final_predictions_scaled[ind_train_k2_first,:] = phi_train_predictions_k2_scaled


phi_train_predictions_k1 = scaler_y.inverse_transform(phi_model_k1.predict(phi_train_x[ind_train_k1_first]))
phi_train_predictions_o1 = scaler_y.inverse_transform(phi_model_k1.predict(lambda_overlap_log_train_scaled_k1))
phi_train_predictions_k2 = scaler_y.inverse_transform(phi_model_k2.predict(phi_train_x[ind_train_k2_first]))
phi_train_predictions_o2 = scaler_y.inverse_transform(phi_model_k2.predict(lambda_overlap_log_train_scaled_k2))
phi_train_predictions_overlap = (phi_train_predictions_o1 + phi_train_predictions_o2)/2

#inversed train predictions into final array
phi_train_final_predictions = np.zeros((200000,8))
phi_train_final_predictions[ind_train_k1_first,:] = phi_train_predictions_k1
phi_train_final_predictions[ind_train_overlap,:] = phi_train_predictions_overlap
phi_train_final_predictions[ind_train_k2_first,:] = phi_train_predictions_k2



#SAVE VALIDATION PREDICTIONS
#validation overlap sector
lambda_overlap = lambda_values_val[ind_val_overlap]
lambda_overlap_log = lambda_overlap.copy()
lambda_overlap_log[: ,0] = np.log10(lambda_overlap_log[: ,0])
lambda_overlap_log_scaled_k1 = scaler_x_k1.transform(lambda_overlap_log)
lambda_overlap_log_scaled_k2 = scaler_x_k2.transform(lambda_overlap_log)

phi_val_predictions_k1 = scaler_y.inverse_transform(phi_model_k1.predict(phi_val_x[ind_val_k1_first]))
phi_val_predictions_o1 = scaler_y.inverse_transform(phi_model_k1.predict(lambda_overlap_log_scaled_k1))
phi_val_predictions_k2 = scaler_y.inverse_transform(phi_model_k2.predict(phi_val_x[ind_val_k2_first]))
phi_val_predictions_o2 = scaler_y.inverse_transform(phi_model_k2.predict(lambda_overlap_log_scaled_k2))
phi_val_predictions_overlap = (phi_val_predictions_o1 + phi_val_predictions_o2)/2

# inverse validation final predictions
phi_val_final_predictions = np.zeros((30000,8))
phi_val_final_predictions[ind_val_k1_first,:] = phi_val_predictions_k1
phi_val_final_predictions[ind_val_overlap,:] = phi_val_predictions_overlap
phi_val_final_predictions[ind_val_k2_first,:] = phi_val_predictions_k2

phi_val_predictions_k1_scaled = phi_model_k1.predict(phi_val_x[ind_val_k1_first])
phi_val_predictions_o1_scaled = phi_model_k1.predict(lambda_overlap_log_scaled_k1)
phi_val_predictions_k2_scaled = phi_model_k2.predict(phi_val_x[ind_val_k2_first])
phi_val_predictions_o2_scaled = phi_model_k2.predict(lambda_overlap_log_scaled_k2)
phi_val_predictions_overlap_scaled = (phi_val_predictions_o1_scaled + phi_val_predictions_o2_scaled)/2

# scaled validation final predictions
phi_val_final_predictions_scaled = np.zeros((30000,8))
phi_val_final_predictions_scaled[ind_val_k1_first,:] = phi_val_predictions_k1_scaled
phi_val_final_predictions_scaled[ind_val_overlap,:] = phi_val_predictions_overlap_scaled
phi_val_final_predictions_scaled[ind_val_k2_first,:] = phi_val_predictions_k2_scaled


print('mse scaled before residual', np.square(coeffs_val_scaled - phi_val_final_predictions_scaled).mean())
print('mse inversed before residual', np.square(coeffs_val - phi_val_final_predictions).mean())


phi_y_pred = np.float64(phi_train_final_predictions_scaled)
phi_errors = coeffs_scaled - phi_y_pred


phi_y_pred_val = np.float64(phi_val_final_predictions_scaled)
phi_errors_val = coeffs_val_scaled - phi_y_pred_val


# input for the residual error network
phi_res_scaler = MinMaxScaler()
phi_errors_scaled = phi_res_scaler.fit_transform(phi_errors)
phi_errors_val_scaled = phi_res_scaler.transform(phi_errors_val)


# Residual Error
phi_res_model = Sequential()
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(8, activation=None))
#
# opt_res = Adam(lr=0.001)
opt_res = Adam(lr=0.01)
phi_res_model.compile(loss='mse', optimizer=opt_res, metrics=['mse'])
phi_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
phi_res_lrm = LearningRateMonitor()
phi_res_stop = EarlyStopping(monitor='mse', patience=400)
phi_res_history = phi_res_model.fit(phi_train_x, phi_errors_scaled, validation_data=(phi_val_x, phi_errors_val_scaled),
                                    epochs=400, batch_size=1000,
                                    verbose=1,
                                    callbacks=[phi_res_rlrp, phi_res_lrm, phi_res_stop])


# Plot training & validation accuracy values
plt.figure()
plt.plot(phi_res_history.history['mse'])
plt.plot(phi_res_history.history['val_mse'])
plt.yscale("log")
plt.title('Residual phi Model Mean Square Error')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('Fig3: phi Res Model Train MSE')
plt.show()

# Learning Rate
# plt.figure()
# plt.plot(phi_res_lrm.lrates)
# plt.yscale("log")
# plt.title('Residual phi Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig4: phi Res Model Learning Rate')
# plt.show()

# # FINAL MSE
# predictions from residual and reverse transform to the scaled data space
phi_train_y_error = phi_res_scaler.inverse_transform(phi_res_model.predict(phi_train_x))
phi_val_y_error = phi_res_scaler.inverse_transform(phi_res_model.predict(phi_val_x))


phi_train_predictions_corrected_scaled = phi_train_final_predictions_scaled + phi_train_y_error
phi_val_predictions_corrected_scaled = phi_val_final_predictions_scaled + phi_val_y_error


phi_train_predictions_k1_with_res = scaler_y.inverse_transform(phi_train_predictions_corrected_scaled[ind_train_k1_first])
phi_train_predictions_o1_with_res = scaler_y.inverse_transform(phi_train_predictions_corrected_scaled[ind_train_overlap])

phi_train_predictions_k2_with_res = scaler_y.inverse_transform(phi_train_predictions_corrected_scaled[ind_train_k2_first])
phi_train_predictions_o2_with_res = scaler_y.inverse_transform(phi_train_predictions_corrected_scaled[ind_train_overlap])

phi_train_predictions_overlap_with_res = (phi_train_predictions_o1_with_res  + phi_train_predictions_o2_with_res )/2

phi_train_final_predictions_with_res  = np.zeros((200000,8))
phi_train_final_predictions_with_res [ind_train_k1_first,:] = phi_train_predictions_k1_with_res
phi_train_final_predictions_with_res [ind_train_overlap,:] = phi_train_predictions_overlap_with_res
phi_train_final_predictions_with_res [ind_train_k2_first,:] = phi_train_predictions_k2_with_res


phi_val_predictions_k1_with_res = scaler_y.inverse_transform(phi_val_predictions_corrected_scaled[ind_val_k1_first])
phi_val_predictions_o1_with_res = scaler_y.inverse_transform(phi_val_predictions_corrected_scaled[ind_val_overlap])

phi_val_predictions_k2_with_res = scaler_y.inverse_transform(phi_val_predictions_corrected_scaled[ind_val_k2_first])
phi_val_predictions_o2_with_res = scaler_y.inverse_transform(phi_val_predictions_corrected_scaled[ind_val_overlap])

phi_val_predictions_overlap_with_res  = (phi_val_predictions_o1_with_res  + phi_val_predictions_o2_with_res )/2

phi_val_final_predictions_with_res  = np.zeros((30000,8))
phi_val_final_predictions_with_res [ind_val_k1_first,:] = phi_val_predictions_k1_with_res
phi_val_final_predictions_with_res [ind_val_overlap,:] = phi_val_predictions_overlap_with_res
phi_val_final_predictions_with_res [ind_val_k2_first,:] = phi_val_predictions_k2_with_res



phi_val_predictions_k1_with_res_scaled = phi_val_predictions_corrected_scaled[ind_val_k1_first]
phi_val_predictions_o1_with_res_scaled  = phi_val_predictions_corrected_scaled[ind_val_overlap]
phi_val_predictions_k2_with_res_scaled  = phi_val_predictions_corrected_scaled[ind_val_k2_first]
phi_val_predictions_o2_with_res_scaled  = phi_val_predictions_corrected_scaled[ind_val_overlap]
phi_val_predictions_overlap_with_res_scaled   = (phi_val_predictions_o1_with_res_scaled   + phi_val_predictions_o2_with_res_scaled  )/2

phi_val_final_predictions_with_res_scaled   = np.zeros((30000,8))
phi_val_final_predictions_with_res_scaled[ind_val_k1_first,:] = phi_val_predictions_k1_with_res_scaled
phi_val_final_predictions_with_res_scaled[ind_val_overlap,:] = phi_val_predictions_overlap_with_res_scaled
phi_val_final_predictions_with_res_scaled[ind_val_k2_first,:] = phi_val_predictions_k2_with_res_scaled


# val_with_res_scaled = np.square(coeffs_val_scaled - phi_val_final_predictions_with_res_scaled)
# print('scaled after res predictions',val_with_res_scaled.mean())

# val_with_res = np.square(coeffs_val - phi_val_final_predictions_with_res)
# print('inversed res predictions',val_with_res.mean())
#
#
# val_before_res = np.square(coeffs_val - phi_val_final_predictions).mean()
# print('inversed before res predictions',val_before_res)

print('Phase')
print('Phase K=1 Validation before res mse:',phi_val_norm_k1_mse)
print('Phase K=2 Validation before res  mse:',phi_val_norm_k2_mse)
val_before_res_scaled = np.square(coeffs_val_scaled - phi_val_final_predictions_scaled).mean()
print('scaled before res predictions',val_before_res_scaled)
val_with_res_scaled = np.square(coeffs_val_scaled - phi_val_final_predictions_with_res_scaled)
print('scaled after res predictions',val_with_res_scaled.mean())

with open('Phase_nn_predictions#5.pkl', 'wb') as f:
    pickle.dump([phi_val_final_predictions, phi_val_final_predictions_with_res], f)


print('minutes:', (time()-start)/60)
