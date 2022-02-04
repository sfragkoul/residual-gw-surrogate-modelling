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


def beta_function(q, s1,s2):
    beta = ((113.0 / 12.0) + (25.0 / (4.0*q))) * s1 * ((q*q)/np.square(1+q))+\
           ((113.0 / 12.0) + ((25.0 *q)/ 4.0)) * s2 * (1.0/np.square(1+q))
    return beta


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

original = np.zeros((200000,4))
original[:,0] = lambda_values[:, 0]
original[:,1] = lambda_values[:, 1]
original[:,2] = lambda_values[:, 2]
# Mtot=60
beta=[]
for i in range(200000):
    print(i)
    spin_1 = lambda_values[i, 1]
    spin_2 = lambda_values[i, 2]
    mass_ratio = lambda_values[i, 0]
    # mass_1 = (mass_ratio * Mtot) / (1 + mass_ratio)
    # mass_2 = Mtot - mass_1
    beta_i = beta_function(mass_ratio, spin_1, spin_2)
    beta.append(beta_i)
    original[i, 3] = beta_i
print(original[:,3].min(), original[:,3].max())

original[:, 0] = np.log10(original[:, 0])
scaler_x = StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(original)
amp_train_x = lambda_values_scaled
amp_train_y = coeffs

print(amp_train_x.shape)
print(amp_train_y.shape)

# plt.figure()
# plt.plot(amp_train_y[:, 3])
# plt.show()

# amp_scaler = MinMaxScaler()  # scaling data to (0,1)
# amp_train_y_scaled = amp_scaler.fit_transform(amp_train_y)  # fitting scaler to training data .fit_transform

with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)

original_val = np.zeros((30000,4))
original_val[:,0] = lambda_values_val[:, 0]
original_val[:,1] = lambda_values_val[:, 1]
original_val[:,2] = lambda_values_val[:, 2]

beta_val=[]
for i in range(30000):
    print(i)
    spin_1 = lambda_values_val[i, 1]
    spin_2 = lambda_values_val[i, 2]
    mass_ratio = lambda_values_val[i, 0]
    beta_i = beta_function(mass_ratio, spin_1, spin_2)
    beta_val.append(beta_i)
    original_val[i, 3] = beta_i
print(original_val[:,3].min(), original_val[:,3].max())

original_val[:, 0] = np.log10(original_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(original_val)
amp_val_x = lambda_values_val_scaled
amp_val_y = coeffs_val
print(amp_val_x.shape)
print(amp_val_y.shape)

# amp_val_y_scaled = amp_scaler.transform(amp_val_y)  # fitting validation data according to training data .transform

# constructing the model
amp_model = Sequential()
amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_model.add(Dense(18, activation=None))

# compile model
opt = Adam(lr=0.001)
amp_model.compile(loss='mse', optimizer=opt, metrics=['mse'])

# fit model
# amp_checkpoint_filepath = './Model_checkpoint_amp/weights.{epoch:02d}.hdf5'
# amp_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=amp_checkpoint_filepath,
#                                                          save_best_only=False, verbose=1,
#                                                          monitor='mse', save_freq=20 * 40)

amp_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9 , patience=15)
amp_lrm = LearningRateMonitor()
amp_stop = EarlyStopping(monitor='mse', patience=1000)
amp_history = amp_model.fit(amp_train_x, amp_train_y, validation_data=(amp_val_x, amp_val_y),
                            epochs=1000, batch_size=1000,
                            verbose=1, callbacks=[amp_rlrp, amp_lrm, amp_stop])

# from tensorflow import keras
# # amp_model.save('amplitude baseline model')
# amp_model = keras.models.load_model('amplitude baseline model')

# Plot training & validation learning curves
plt.figure()
plt.plot(amp_history.history['mse'])
plt.plot(amp_history.history['val_mse'])
plt.yscale("log")
plt.title('Amplitude Model Mean Square Error')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('Fig1: Amp Model Train MSE')
plt.savefig('Fig1: Amp tol e-8 Model Train MSE')
plt.show()
#
# # Learning Rate
# plt.figure()
# plt.plot(amp_lrm.lrates)
# plt.yscale("log")
# plt.title('Amplitude Learning Rate')
# plt.xlabel('Epoch')
# # plt.savefig('Fig2: Amp Model Learning Rate ')
# plt.savefig('Fig2: Amp tol e-10 Model Learning Rate ')
# plt.show()

train_evaluation = amp_model.evaluate(amp_train_x, amp_train_y, batch_size=1000)

# # TESTING
# with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_test.pkl', 'rb') as f:
#     [lambda_values_test, coeffs_test, eim_basis_test, eim_indices_test] = pickle.load(f)
#
# # print(lambda_values_test[712,:])
# lambda_values_test[:, 0] = np.log10(lambda_values_test[:, 0])
# lambda_values_test_scaled = scaler_x.transform(lambda_values_test)
# amp_test_x = lambda_values_test_scaled
# amp_test_y = coeffs_test
# print(amp_test_x.shape)
# print(amp_test_y.shape)
# amp_test_y_scaled = amp_scaler.transform(amp_test_y)

#thelw kai predictions apo validaion
# amp_val_predictions = amp_model.predict(amp_val_x)
# amp_test_predictions = amp_model.predict(amp_test_x)
#
# amp_test_mse = np.square(amp_test_y-amp_test_predictions).mean()
# print(amp_test_mse)

#SAVE PREDICTIONS FOR MISMATCH SCRIPT
# with open('Amplitude_nn_predictions.pkl', 'wb') as f:
#     pickle.dump([amp_val_predictions, amp_test_predictions], f)
########################################################################################
# errors from amplitude training
amp_y_pred = amp_model.predict(amp_train_x)
amp_y_pred = np.float64(amp_y_pred)
amp_errors = amp_train_y - amp_y_pred
amp_mse_train = np.mean(amp_errors ** 2)
print(amp_mse_train)

amp_y_pred_val = amp_model.predict(amp_val_x)
amp_y_pred_val = np.float64(amp_y_pred_val)
amp_errors_val = amp_val_y - amp_y_pred_val
amp_mse_val = np.mean(amp_errors_val ** 2)
print(amp_mse_val)
#
# # input for the residual error network
amp_res_scaler = MinMaxScaler()
amp_errors_new = amp_res_scaler.fit_transform(amp_errors)
amp_errors_val_new = amp_res_scaler.transform(amp_errors_val)
print(amp_errors_new)


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
amp_res_stop = EarlyStopping(monitor='val_mse', patience=20, mode='min')
amp_res_history = amp_res_model.fit(amp_train_x, amp_errors_new, validation_data=(amp_val_x, amp_errors_val_new),
                                    epochs=1000, batch_size=1000,
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
plt.figure()
plt.plot(amp_res_lrm.lrates)
plt.yscale("log")
plt.title('Amp Residual Learning Rate')
plt.xlabel('Epoch')
plt.savefig('Fig4: Amp Res Model Learning Rate')
plt.show()
#
# # FINAL MSE
#
# predictions from residual and reverse transform to the INITIAL data space
amp_train_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_train_x))
amp_val_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_val_x))
#
# # adding corrections from residual
amp_train_y_pred = amp_model.predict(amp_train_x) + amp_train_y_error
amp_val_predictions = amp_model.predict(amp_val_x) + amp_val_y_error
amp_val_predictions_before_residual = amp_model.predict(amp_val_x)
amp_train_predictions_before_residual = amp_model.predict(amp_train_x)
#
# # reverse transform to the initial data space
# amp_train_y_pred = amp_scaler.inverse_transform(amp_train_y_pred_corrected)
# amp_val_y_pred = amp_scaler.inverse_transform(amp_val_y_pred_corrected)
#
amp_final_train_mse = np.square(amp_train_y_pred - amp_train_y).mean()
amp_final_train_mse_before_res = np.square(amp_train_predictions_before_residual - amp_train_y).mean()
amp_final_val_mse = np.square(amp_val_predictions - amp_val_y).mean()
amp_final_val_mse_before_res = np.square(amp_val_predictions_before_residual - amp_val_y).mean()

print('Amplitude')
print('Train mse: ',amp_final_train_mse)
print('Train mse before residual: ', amp_final_train_mse_before_res)

print('Validation mse: ',  amp_final_val_mse)
print('Validation mse before residual: ',  amp_final_val_mse_before_res)

with open('Amplitude_nn_predictions#5.pkl', 'wb') as f:
    pickle.dump([amp_val_predictions_before_residual, amp_val_predictions ], f)