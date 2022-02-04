import pickle
import rompy

seed = 0

import os

os.environ['PYTHONHASHSEED'] = str(seed)

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
from keras.optimizers import Adam
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



with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10.pkl', 'rb') as f:
    [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)


# SPLIT DATASETS
#gia kanonikopoiis khan kai green

lambda_values[:, 0] = np.log10(lambda_values[:, 0])
scaler_x = StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values)
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

for_now = coeffs_val.mean(axis=0)


lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
amp_val_x = lambda_values_val_scaled
amp_val_y = coeffs_val
print(amp_val_x.shape)
print(amp_val_y.shape)

# amp_val_y_scaled = amp_scaler.transform(amp_val_y)  # fitting validation data according to training data .transform
#
# # constructing the model
# amp_model = Sequential()
# amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# amp_model.add(Dense(18, activation=None))
#
# # compile model
# opt = Adam(lr=0.001)
# amp_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
#
# # fit model
# # amp_checkpoint_filepath = './Model_checkpoint_amp/weights.{epoch:02d}.hdf5'
# # amp_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=amp_checkpoint_filepath,
# #                                                          save_best_only=False, verbose=1,
# #                                                          monitor='mse', save_freq=20 * 40)
#
# amp_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9 , patience=15)
# amp_lrm = LearningRateMonitor()
# amp_stop = EarlyStopping(monitor='mse', patience=100)
# amp_history = amp_model.fit(amp_train_x, amp_train_y, validation_data=(amp_val_x, amp_val_y),
#                             epochs=1000, batch_size=1000,
#                             verbose=1, callbacks=[amp_rlrp, amp_lrm, amp_stop])
#
# # amp_model.save('best amp model norm x and raw y')
# # Plot training & validation learning curves
# plt.figure()
# plt.plot(amp_history.history['mse'])
# plt.plot(amp_history.history['val_mse'])
# plt.yscale("log")
# plt.title('Amplitude Model Mean Square Error')
# plt.ylabel('mse')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# # plt.savefig('Fig1: Amp Model Train MSE')
# plt.savefig('Fig1: Amp tol e-8 Model Train MSE')
# plt.show()
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
#
# train_evaluation = amp_model.evaluate(amp_train_x, amp_train_y, batch_size=1000)
#
# # TESTING
# with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_test.pkl', 'rb') as f:
#     [lambda_values_test, coeffs_test, eim_basis_test, eim_indices_test] = pickle.load(f)
#
# print(lambda_values_test[712,:])
# lambda_values_test[:, 0] = np.log10(lambda_values_test[:, 0])
# lambda_values_test_scaled = scaler_x.transform(lambda_values_test)
# amp_test_x = lambda_values_test_scaled
# amp_test_y = coeffs_test
# print(amp_test_x.shape)
# print(amp_test_y.shape)
# # amp_test_y_scaled = amp_scaler.transform(amp_test_y)
#
# #thelw kai predictions apo validaion
# amp_val_predictions = amp_model.predict(amp_val_x)
# amp_test_predictions = amp_model.predict(amp_test_x)
#
# amp_test_mse = np.square(amp_test_y-amp_test_predictions).mean()
# print(amp_test_mse)
#
# amp_val_mse = np.square(amp_val_y-amp_val_predictions).mean()
# print('Vadilation mse: ',amp_val_mse)
#
# with open('Amplitude_nn_predictions.pkl', 'wb') as f:
#     pickle.dump([amp_val_predictions, amp_test_predictions], f)
########################################################################################
# # errors from amplitude training
# amp_y_pred = amp_model.predict(amp_train_x)
# amp_y_pred = np.float64(amp_y_pred)
# amp_errors = amp_train_y_scaled - amp_y_pred
# amp_mse_train = np.mean(amp_errors ** 2)
# print(amp_mse_train)
#
# amp_y_pred_val = amp_model.predict(amp_val_x)
# amp_y_pred_val = np.float64(amp_y_pred_val)
# amp_errors_val = amp_val_y_scaled - amp_y_pred_val
# amp_mse_val = np.mean(amp_errors_val ** 2)
# print(amp_mse_val)
#
# # input for the residual error network
# amp_res_scaler = MinMaxScaler()
# amp_errors_new = amp_res_scaler.fit_transform(amp_errors)
# amp_errors_val_new = amp_res_scaler.transform(amp_errors_val)
# print(amp_errors_new)
#
# # with open('amp_scaler.pkl', 'wb') as f:
# #     pickle.dump([amp_scaler, amp_res_scaler ], f)
#
# # Residual Error
# amp_res_model = Sequential()
# amp_res_model.add(Dense(64, activation=tf.keras.layers.PReLU()))
# amp_res_model.add(Dense(128, activation=tf.keras.layers.PReLU()))
# amp_res_model.add(Dense(256, activation=tf.keras.layers.PReLU()))
# amp_res_model.add(Dense(512, activation=tf.keras.layers.PReLU()))
# amp_res_model.add(Dense(17, activation=None))
#
# opt = Adam(lr=0.001)
# amp_res_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# amp_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.8, patience=15)
# amp_res_lrm = LearningRateMonitor()
# amp_res_stop = EarlyStopping(monitor='mse', patience=50)
#
# amp_res_checkpoint_filepath = './Model_checkpoint_amp_res/weights.{epoch:02d}.hdf5'
# amp_res_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=amp_res_checkpoint_filepath,
#                                                              save_best_only=False, verbose=1,
#                                                              monitor='mse', save_freq=20 * 40)
#
# # history = model_res.fit(train_x_shuffled, errors*1e4, validation_data=(val_x, errors_val*1e4),
# amp_res_history = amp_res_model.fit(amp_train_x, amp_errors_new, validation_data=(amp_val_x, amp_errors_val_new),
#                                     epochs=600, batch_size=128,
#                                     verbose=1,
#                                     callbacks=[amp_res_rlrp, amp_res_lrm, amp_res_stop, amp_res_modelcheckpoint])
#
# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(amp_res_history.history['mse'])
# plt.plot(amp_res_history.history['val_mse'])
# plt.yscale("log")
# plt.title('Residual Amp Model Mean Square Error')
# plt.ylabel('mse')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('Fig3: Amp Res Model Train MSE')
# plt.show()
#
# # Learning Rate
# plt.figure()
# plt.plot(amp_res_lrm.lrates)
# plt.yscale("log")
# plt.title('Residual Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig4: Amp Res Model Learning Rate')
# plt.show()
#
# # FINAL MSE
#
# # predictions from residual and reverse transform to the scaled data space
# amp_train_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_train_x))
# amp_val_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_val_x))
#
# # adding corrections from residual
# amp_train_y_pred_corrected = amp_model.predict(amp_train_x) + amp_train_y_error
# amp_val_y_pred_corrected = amp_model.predict(amp_val_x) + amp_val_y_error
#
# # reverse transform to the initial data space
# amp_train_y_pred = amp_scaler.inverse_transform(amp_train_y_pred_corrected)
# amp_val_y_pred = amp_scaler.inverse_transform(amp_val_y_pred_corrected)
#
# amp_final_train_mse = np.square(amp_train_y_pred - amp_train_y).mean()
# amp_final_val_mse = np.square(amp_val_y_pred - amp_val_y).mean()
# print(amp_final_train_mse, amp_final_val_mse)
#
# amp_final_train_mse_plot = np.square(amp_train_y_pred - amp_train_y)
#
# plt.subplot(3, 1, 1)
# plt.plot(amp_final_train_mse_plot[:, 0], '+')
# plt.subplot(3, 1, 2)
# plt.plot(amp_final_train_mse_plot[:, 1], '+')
# plt.subplot(3, 1, 3)
# plt.plot(amp_final_train_mse_plot[:, 2], '+')
# plt.tight_layout()
# plt.show()
#
# amp_final_val_mse_plot = np.square(amp_val_y_pred - amp_val_y)
#
# plt.subplot(3, 1, 1)
# plt.plot(amp_final_val_mse_plot[:, 0], '+')
# plt.subplot(3, 1, 2)
# plt.plot(amp_final_val_mse_plot[:, 1], '+')
# plt.subplot(3, 1, 3)
# plt.plot(amp_final_val_mse_plot[:, 2], '+')
# plt.tight_layout()
# plt.show()
#
# amp_final_train_mse_plot = np.square(amp_train_y_pred - amp_train_y).mean(axis=0)
#
# x = np.arange(17)
#
# plt.figure()
# plt.plot(x, amp_final_train_mse_plot)
# plt.yscale("log")
# plt.title('MSE per Coefficient')
# plt.ylabel('mse')
# plt.xlabel('coef')
# plt.show()
#
# # plt.figure()
# # plt.plot(amp_train_y_pred[:, 2], '+', label='prediction')
# # plt.plot(amp_train_y[:, 2], '+', alpha=0.55, label='ground truth')
# # plt.title('Amplitude Predictions vs Ground Truth')
# # plt.legend()
# # plt.show()
# #
# # # Training Set Reconstruct from NN predictions and Mismatch
# # amp_nn_reconstruct = amp_train_y_pred.dot(eim_A_basis)
# #
# # mismatch_amp_nn_reconstruct = [integration_A.mismatch(A_ro[i], amp_nn_reconstruct[i]) for i in
# #                                range(amp_nn_reconstruct.shape[0])]
# #
# # mismatch_amp_nn_reconstruct_np = np.asarray(mismatch_amp_nn_reconstruct)
# #
# #
# # #
# # # # Validation Set Reconstruct from NN predictions and Mismatch
# # # amp_val_nn_reconstruct = amp_val_y_pred.dot(eim_A_basis)
# # #
# # # mismatch_amp_val_nn_reconstruct = [integration_A.mismatch(A_ro_val[i], amp_val_nn_reconstruct[i]) for i in
# # #                                    range(amp_val_nn_reconstruct.shape[0])]
# # # mismatch_amp_val_nn_reconstruct_np = np.asarray(mismatch_amp_val_nn_reconstruct)
# # #
# # # plt.subplot(1, 2, 1)
# # # plt.semilogy(np.abs(mismatch_amp_nn_reconstruct_np), '+', label='NN')
# # # plt.semilogy(np.abs(mismatch_eim_A), 'o', label='EIM', alpha=.3)
# # # plt.xlabel('$i$')
# # # plt.ylabel('mismatch')
# # # plt.title('Amp Mismatch for Training')
# # # plt.legend()
# # #
# # # plt.subplot(1, 2, 2)
# # # plt.semilogy(np.abs(mismatch_amp_val_nn_reconstruct_np), '+', label='NN')
# # # plt.semilogy(np.abs(mismatch_eim_A), 'o', label='EIM', alpha=.3)
# # # plt.xlabel('$i$')
# # # plt.ylabel('mismatch')
# # # plt.title('Amp Mismatch for Validation')
# # # plt.legend()
# # # plt.tight_layout()
# # # plt.show()
#
#
# # with open('seobnrv4_amp_nn_reconstruct.pkl', 'wb') as f:
# #     pickle.dump([amp_nn_reconstruct, amp_val_nn_reconstruct ], f)