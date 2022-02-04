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
from keras.optimizers import Adam, Adamax
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
# gia na tsekarw to evros twn coeffs
    print(coeffs.max())
    print(coeffs.min())

# SPLIT DATASETS
#gia kanonikopoiis khan kai green

lambda_values[:,0]=np.log10(lambda_values[:,0])
scaler_x=StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values)
phi_train_x = lambda_values_scaled
phi_train_y = coeffs

print(phi_train_x.shape)
print(phi_train_y.shape)

# plt.figure()
# plt.plot(phi_train_y[:, 3])
# plt.show()

phi_scaler = MinMaxScaler()  # scaling data to (0,1)
phi_train_y_scaled = phi_scaler.fit_transform(phi_train_y)  # fitting scaler to training data .fit_transform

with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)

for_now = coeffs_val.mean(axis=0)

lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
phi_val_x = lambda_values_val_scaled
phi_val_y = coeffs_val
print(phi_val_x.shape)
print(phi_val_y.shape)




fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(lambda_values_val[:,1],coeffs_val[:, 1], coeffs_val[:, 2], c=lambda_values_val[:,0],  cmap='plasma' );
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$q$')
# plt.title('Pycbc validation mismatch for tol%' %tols)
# plt.savefig('Pycbc validation mismatch for tol{}'.format(tols))
plt.show()













#
# phi_val_y_scaled = phi_scaler.transform(phi_val_y)  # fitting validation data according to training data .transform
#
# # constructing the model
# phi_model = Sequential()
# # phi_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# # phi_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# # phi_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# # phi_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
# # phi_model.add(Dense(39, activation=None))
# phi_model.add(Dense(320, activation= 'softplus'))
# phi_model.add(Dense(320, activation='softplus'))
# phi_model.add(Dense(320, activation='softplus'))
# phi_model.add(Dense(320, activation='softplus'))
# phi_model.add(Dense(8, activation=None))
#
# # compile model
# opt = Adamax(lr=0.01)
# phi_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
#
# # fit model
# # phi_checkpoint_filepath = './Model_checkpoint_phi/weights.{epoch:02d}.hdf5'
# # phi_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=phi_checkpoint_filepath,
# #                                                          save_best_only=False, verbose=1,
# #                                                          monitor='mse', save_freq=20 * 40)
#
# phi_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
# phi_lrm = LearningRateMonitor()
# phi_stop = EarlyStopping(monitor='mse', patience=100)
# phi_history = phi_model.fit(phi_train_x, phi_train_y_scaled, validation_data=(phi_val_x, phi_val_y_scaled),
#                             epochs=1000, batch_size=1000,
#                             verbose=1, callbacks=[phi_rlrp, phi_lrm, phi_stop])
#
#
# # Plot training & validation learning curves
# plt.figure()
# plt.plot(phi_history.history['mse'])
# plt.plot(phi_history.history['val_mse'])
# plt.yscale("log")
# plt.title('Phase Model Mean Square Error')
# plt.ylabel('mse')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('Fig1: Phi Model Train MSE')
# plt.show()
#
# # Learning Rate
# plt.figure()
# plt.plot(phi_lrm.lrates)
# plt.yscale("log")
# plt.title('Phase Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig2: Phi Model Learning Rate ')
# plt.show()
#
# train_evaluation = phi_model.evaluate(phi_train_x, phi_train_y_scaled, batch_size=1000)
#
# # TESTING
# with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10_test.pkl', 'rb') as f:
#     [lambda_values_test, coeffs_test, eim_basis_test, eim_indices_test] = pickle.load(f)
#
# # print(lambda_values_test[712,:])
#
# lambda_values_test[:, 0] = np.log10(lambda_values_test[:, 0])
# lambda_values_test_scaled = scaler_x.transform(lambda_values_test)
# phi_test_x = lambda_values_test_scaled
# phi_test_y = coeffs_test
# print(phi_test_x.shape)
# print(phi_test_y.shape)
# phi_test_y_scaled = phi_scaler.transform(phi_test_y)
#
# phi_val_norm=np.square(phi_val_y_scaled-phi_model.predict(phi_val_x))
# print('Validation mse:',phi_val_norm.mean())
#
# phi_test_norm=np.square(phi_test_y_scaled-phi_model.predict(phi_test_x))
# print('Test mse:',phi_test_norm.mean())
#
# phi_val_predictions = phi_scaler.inverse_transform(phi_model.predict(phi_val_x))
# phi_test_predictions = phi_scaler.inverse_transform(phi_model.predict(phi_test_x)) #inverse prin ta kanw save
# phi_test_mse = np.square(phi_test_y-phi_test_predictions).mean()
# print(phi_test_mse)
#
# phi_val_mse = np.square(phi_val_y-phi_val_predictions).mean()
# print(phi_val_mse)
#
# with open('Phase_nn_predictions.pkl', 'wb') as f:
#     pickle.dump([phi_val_predictions, phi_test_predictions], f)
# # ########################################################################################
# # # errors from philitude training
# phi_y_pred = phi_model.predict(phi_train_x)
# phi_y_pred = np.float64(phi_y_pred)
# phi_errors = phi_train_y_scaled - phi_y_pred
# phi_mse_train = np.mean(phi_errors ** 2)
# print(phi_mse_train)
# 
# phi_y_pred_val = phi_model.predict(phi_val_x)
# phi_y_pred_val = np.float64(phi_y_pred_val)
# phi_errors_val = phi_val_y_scaled - phi_y_pred_val
# phi_mse_val = np.mean(phi_errors_val ** 2)
# print(phi_mse_val)
# 
# # input for the residual error network
# phi_res_scaler = MinMaxScaler()
# phi_errors_new = phi_res_scaler.fit_transform(phi_errors)
# phi_errors_val_new = phi_res_scaler.transform(phi_errors_val)
# print(phi_errors_new)
# 
# # with open('phi_scaler.pkl', 'wb') as f:
# #     pickle.dump([phi_scaler, phi_res_scaler ], f)
# 
# # Residual Error
# phi_res_model = Sequential()
# phi_res_model.add(Dense(64, activation=tf.keras.layers.PReLU()))
# phi_res_model.add(Dense(128, activation=tf.keras.layers.PReLU()))
# phi_res_model.add(Dense(256, activation=tf.keras.layers.PReLU()))
# phi_res_model.add(Dense(512, activation=tf.keras.layers.PReLU()))
# phi_res_model.add(Dense(17, activation=None))
# 
# opt = Adam(lr=0.001)
# phi_res_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# phi_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.8, patience=15)
# phi_res_lrm = LearningRateMonitor()
# phi_res_stop = EarlyStopping(monitor='mse', patience=50)
# 
# phi_res_checkpoint_filepath = './Model_checkpoint_phi_res/weights.{epoch:02d}.hdf5'
# phi_res_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=phi_res_checkpoint_filepath,
#                                                              save_best_only=False, verbose=1,
#                                                              monitor='mse', save_freq=20 * 40)
# 
# # history = model_res.fit(train_x_shuffled, errors*1e4, validation_data=(val_x, errors_val*1e4),
# phi_res_history = phi_res_model.fit(phi_train_x, phi_errors_new, validation_data=(phi_val_x, phi_errors_val_new),
#                                     epochs=600, batch_size=128,
#                                     verbose=1,
#                                     callbacks=[phi_res_rlrp, phi_res_lrm, phi_res_stop, phi_res_modelcheckpoint])
# 
# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(phi_res_history.history['mse'])
# plt.plot(phi_res_history.history['val_mse'])
# plt.yscale("log")
# plt.title('Residual phi Model Mean Square Error')
# plt.ylabel('mse')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('Fig3: phi Res Model Train MSE')
# plt.show()
# 
# # Learning Rate
# plt.figure()
# plt.plot(phi_res_lrm.lrates)
# plt.yscale("log")
# plt.title('Residual Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig4: phi Res Model Learning Rate')
# plt.show()
# 
# # FINAL MSE
# 
# # predictions from residual and reverse transform to the scaled data space
# phi_train_y_error = phi_res_scaler.inverse_transform(phi_res_model.predict(phi_train_x))
# phi_val_y_error = phi_res_scaler.inverse_transform(phi_res_model.predict(phi_val_x))
# 
# # adding corrections from residual
# phi_train_y_pred_corrected = phi_model.predict(phi_train_x) + phi_train_y_error
# phi_val_y_pred_corrected = phi_model.predict(phi_val_x) + phi_val_y_error
# 
# # reverse transform to the initial data space
# phi_train_y_pred = phi_scaler.inverse_transform(phi_train_y_pred_corrected)
# phi_val_y_pred = phi_scaler.inverse_transform(phi_val_y_pred_corrected)
# 
# phi_final_train_mse = np.square(phi_train_y_pred - phi_train_y).mean()
# phi_final_val_mse = np.square(phi_val_y_pred - phi_val_y).mean()
# print(phi_final_train_mse, phi_final_val_mse)
# 
# phi_final_train_mse_plot = np.square(phi_train_y_pred - phi_train_y)
# 
# plt.subplot(3, 1, 1)
# plt.plot(phi_final_train_mse_plot[:, 0], '+')
# plt.subplot(3, 1, 2)
# plt.plot(phi_final_train_mse_plot[:, 1], '+')
# plt.subplot(3, 1, 3)
# plt.plot(phi_final_train_mse_plot[:, 2], '+')
# plt.tight_layout()
# plt.show()
# 
# phi_final_val_mse_plot = np.square(phi_val_y_pred - phi_val_y)
# 
# plt.subplot(3, 1, 1)
# plt.plot(phi_final_val_mse_plot[:, 0], '+')
# plt.subplot(3, 1, 2)
# plt.plot(phi_final_val_mse_plot[:, 1], '+')
# plt.subplot(3, 1, 3)
# plt.plot(phi_final_val_mse_plot[:, 2], '+')
# plt.tight_layout()
# plt.show()
# 
# phi_final_train_mse_plot = np.square(phi_train_y_pred - phi_train_y).mean(axis=0)
# 
# x = np.arange(17)
# 
# plt.figure()
# plt.plot(x, phi_final_train_mse_plot)
# plt.yscale("log")
# plt.title('MSE per Coefficient')
# plt.ylabel('mse')
# plt.xlabel('coef')
# plt.show()
# 
# # plt.figure()
# # plt.plot(phi_train_y_pred[:, 2], '+', label='prediction')
# # plt.plot(phi_train_y[:, 2], '+', alpha=0.55, label='ground truth')
# # plt.title('philitude Predictions vs Ground Truth')
# # plt.legend()
# # plt.show()
# #
# # # Training Set Reconstruct from NN predictions and Mismatch
# # phi_nn_reconstruct = phi_train_y_pred.dot(eim_A_basis)
# #
# # mismatch_phi_nn_reconstruct = [integration_A.mismatch(A_ro[i], phi_nn_reconstruct[i]) for i in
# #                                range(phi_nn_reconstruct.shape[0])]
# #
# # mismatch_phi_nn_reconstruct_np = np.asarray(mismatch_phi_nn_reconstruct)
# #
# #
# # #
# # # # Validation Set Reconstruct from NN predictions and Mismatch
# # # phi_val_nn_reconstruct = phi_val_y_pred.dot(eim_A_basis)
# # #
# # # mismatch_phi_val_nn_reconstruct = [integration_A.mismatch(A_ro_val[i], phi_val_nn_reconstruct[i]) for i in
# # #                                    range(phi_val_nn_reconstruct.shape[0])]
# # # mismatch_phi_val_nn_reconstruct_np = np.asarray(mismatch_phi_val_nn_reconstruct)
# # #
# # # plt.subplot(1, 2, 1)
# # # plt.semilogy(np.abs(mismatch_phi_nn_reconstruct_np), '+', label='NN')
# # # plt.semilogy(np.abs(mismatch_eim_A), 'o', label='EIM', alpha=.3)
# # # plt.xlabel('$i$')
# # # plt.ylabel('mismatch')
# # # plt.title('phi Mismatch for Training')
# # # plt.legend()
# # #
# # # plt.subplot(1, 2, 2)
# # # plt.semilogy(np.abs(mismatch_phi_val_nn_reconstruct_np), '+', label='NN')
# # # plt.semilogy(np.abs(mismatch_eim_A), 'o', label='EIM', alpha=.3)
# # # plt.xlabel('$i$')
# # # plt.ylabel('mismatch')
# # # plt.title('phi Mismatch for Validation')
# # # plt.legend()
# # # plt.tight_layout()
# # # plt.show()
# 
# 
# # with open('seobnrv4_phi_nn_reconstruct.pkl', 'wb') as f:
#     pickle.dump([phi_nn_reconstruct, phi_val_nn_reconstruct ], f)