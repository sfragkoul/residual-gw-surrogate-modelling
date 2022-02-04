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
from keras.optimizers import Adam, Adamax, RMSprop, SGD
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

# gia -log(q)
original = lambda_values.copy()
original[:, 0] = -(np.log10(original[:, 0]))
print(original[:,0].max())
print(original[:,0].min())


new_original = np.zeros((200000,3))
new_original[:,0] = original[:,0]
#x2 to x1
new_original[:,1] = lambda_values[:, 2]
#x1 to x2
new_original[:,2] = lambda_values[:, 1]

# gia log(q)
original2 = lambda_values.copy()
original2[:, 0] = np.log10(original2[:, 0])
print(original2[:,0].max())
print(original2[:,0].min())


lambda_values_new = np.concatenate((original2 , new_original), axis=0)
coeffs_new = np.concatenate((coeffs , coeffs), axis=0)

# lambda_values[:, 0] = np.log10(lambda_values[:, 0])
scaler_x = StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values_new)
amp_train_x = lambda_values_scaled
amp_train_y = coeffs_new

# lambda_values_test = np.concatenate((original2 , new_original), axis=1)
# lambda_values_test_scaled = np.zeros((200000,6))
# lambda_values_test_scaled[:,:3] = scaler_x.transform(lambda_values_test[:,:3])
# lambda_values_test_scaled[:,3:] = scaler_x.transform(lambda_values_test[:,3:])


print(amp_train_x.shape)
print(amp_train_y.shape)

with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)


# gia -log(q)
original_val = lambda_values_val.copy()
original_val[:, 0] = -(np.log10(original_val[:, 0]))
print(original_val[:,0].max())
print(original_val[:,0].min())


new_original_val = np.zeros((30000,3))
new_original_val[:,0] = original_val[:,0]
#x2 to x1
new_original_val[:,1] = lambda_values_val[:, 2]
#x1 to x2
new_original_val[:,2] = lambda_values_val[:, 1]

# gia log(q)
original2_val = lambda_values_val.copy()
original2_val[:, 0] = np.log10(original2_val[:, 0])
print(original2_val[:,0].max())
print(original2_val[:,0].min())


lambda_values_new_val = np.concatenate((original2_val , new_original_val), axis=0)
# coeffs_new_val = np.concatenate((coeffs_val , coeffs_val), axis=0)
new_original_val_scaled = scaler_x.transform(new_original_val)


lambda_values_val_scaled = scaler_x.transform(original2_val)
amp_val_x = lambda_values_val_scaled
amp_val_y = coeffs_val
print(amp_val_x.shape)
print(amp_val_y.shape)


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


amp_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9 , patience=15)
amp_lrm = LearningRateMonitor()
amp_stop = EarlyStopping(monitor='mse', patience=100)
amp_history = amp_model.fit(amp_train_x, amp_train_y, validation_data=(amp_val_x, amp_val_y),
                            epochs=500, batch_size=1000,
                            verbose=1, callbacks=[amp_rlrp, amp_lrm, amp_stop])

# amp_model.save('best amp model norm x and raw y')
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
plt.savefig('Fig1: Amp Model Train MSE')
plt.show()

# # Learning Rate
# plt.figure()
# plt.plot(amp_lrm.lrates)
# plt.yscale("log")
# plt.title('Amplitude Learning Rate')
# plt.xlabel('Epoch')
# # plt.savefig('Fig2: Amp Model Learning Rate ')
# plt.savefig('Fig2: Amp Model Learning Rate ')
# plt.show()

# train_evaluation = amp_model.evaluate(amp_train_x, amp_train_y, batch_size=1000)

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


# # input for the residual error network
amp_res_scaler = MinMaxScaler()
amp_errors_scaled = amp_res_scaler.fit_transform(amp_errors)
amp_errors_val_scaled = amp_res_scaler.transform(amp_errors_val)
print(amp_errors_scaled)




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
amp_res_stop = EarlyStopping(monitor='mse', patience=200)
amp_res_history = amp_res_model.fit(amp_train_x, amp_errors_scaled, validation_data=(amp_val_x, amp_errors_val_scaled),
                                    epochs=200, batch_size=1000,
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

# # Learning Rate
# plt.figure()
# plt.plot(amp_res_lrm.lrates)
# plt.yscale("log")
# plt.title('Amp Residual Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig4: Amp Res Model Learning Rate')
# plt.show()


# # FINAL MSE

# predictions from residual and reverse transform to the INITIAL data space
amp_train_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_train_x))
amp_val_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(amp_val_x))


#thelw kai predictions apo validaion
amp_val_predictions = amp_model.predict(amp_val_x)
amp_val_predictions_new = amp_model.predict(new_original_val_scaled)
amp_val_predictions_ens = (amp_val_predictions + amp_val_predictions_new)/2

amp_val_mse = np.square(amp_val_y - amp_val_predictions_ens).mean()
print('ensemble Validation mse before residual: ',amp_val_mse)

amp_val_mse_q = np.square(amp_val_y - amp_val_predictions).mean()
print('log(q) Validation mse before residual: ',amp_val_mse_q)

amp_val_mse_1_over_q = np.square(amp_val_y - amp_val_predictions_new).mean()
print('-log(q) Validation mse before residual: ',amp_val_mse_1_over_q)






#thelw kai predictions apo validaion me residual
amp_val_predictions_with_res = amp_model.predict(amp_val_x)+ amp_val_y_error
amp_val_predictions_new_with_res = amp_model.predict(new_original_val_scaled)+ amp_val_y_error
amp_val_predictions_ens_with_res = (amp_val_predictions_with_res + amp_val_predictions_new_with_res)/2

amp_val_mse_with_res = np.square(amp_val_y - amp_val_predictions_ens_with_res).mean()
print('ensemble Validation mse with residual: ', amp_val_mse_with_res)

amp_val_mse_q_with_res = np.square(amp_val_y - amp_val_predictions_with_res).mean()
print('log(q) Validation mse with residual: ',amp_val_mse_q_with_res)

amp_val_mse_1_over_q_with_res = np.square(amp_val_y - amp_val_predictions_new_with_res).mean()
print('-log(q) Validation mse with residual: ',amp_val_mse_1_over_q_with_res)


# # with open('Amplitude_nn_predictions.pkl', 'wb') as f:
# #     pickle.dump([amp_val_predictions_ens_with_res, amp_val_predictions_with_res, amp_val_predictions_new_with_res], f)
#
# amp_train_predictions_with_res = amp_model.predict(amp_train_x[:200000, :])+ amp_train_y_error[:200000, :]
# amp_train_predictions_new_with_res = amp_model.predict(amp_train_x[200000:, :])+ amp_train_y_error[200000:, :]
# amp_train_predictions_stack =  np.concatenate((amp_train_predictions_with_res , amp_train_predictions_new_with_res), axis=1)
#
# amp_val_predictions_with_res = amp_model.predict(amp_val_x)+ amp_val_y_error
# amp_val_predictions_new_with_res = amp_model.predict(new_original_val_scaled)+ amp_val_y_error
# amp_val_predictions_stack =  np.concatenate((amp_val_predictions_with_res , amp_val_predictions_new_with_res), axis=1)
#
#
# identity = np.identity(18)
# identity_stack = np.concatenate((identity, identity), axis=0) * 0.5
# bias = np.zeros(18)
#
# # initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=0)
# amp_ens_model = Sequential()
# # phi_ens_model.add(Dense(8, activation=None, kernel_initializer=initializer))
# amp_ens_model.add(Dense(18, activation=None))
# amp_ens_model.build((None,36))
# amp_ens_model.layers[0].set_weights((identity_stack, bias))
#
#
# opt_ens = SGD(lr=0.00001)
# amp_ens_model.compile(loss='mse', optimizer=opt_ens, metrics=['mse'])
# amp_ens_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=50)
# amp_ens_lrm = LearningRateMonitor()
# amp_ens_stop = EarlyStopping(monitor='mse', patience=200)
# amp_ens_history = amp_ens_model.fit(amp_train_predictions_stack, coeffs, validation_data=(amp_val_predictions_stack, coeffs_val),
#                                     epochs=200, batch_size=1000,
#                                     verbose=1,
#                                    callbacks=[amp_ens_rlrp, amp_ens_lrm, amp_ens_stop])
#
# # wght = amp_ens_model.layers[0].get_weights()[0]
#
# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(amp_ens_history.history['mse'])
# plt.plot(amp_ens_history.history['val_mse'])
# plt.yscale("log")
# plt.title('Ensemble Amp Model MSE')
# plt.ylabel('mse')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('Fig3: Amp ENS Model Train MSE')
# plt.show()
#
# # COEFFS = np.concatenate((coeffs, coeffs), axis=1)
# predictions_modl_ens = amp_ens_model.predict(amp_train_predictions_stack)
# mse_model = np.square(coeffs - predictions_modl_ens)
# print('ensemble network training mse', mse_model.mean())
#
# # COEFFS_val = np.concatenate((coeffs_val, coeffs_val), axis=1)
# predictions_model_ens_val = amp_ens_model.predict(amp_val_predictions_stack)
# mse_model_val = np.square(coeffs_val - predictions_model_ens_val)
# print('Ensemble network validation mse',mse_model_val.mean())


print('Amplitude')
print('log(q) Validation mse before residual: ',amp_val_mse_q)
print('-log(q) Validation mse before residual: ',amp_val_mse_1_over_q)
print('ensemble Validation mse before residual: ',amp_val_mse)
print('log(q) Validation mse with residual: ',amp_val_mse_q_with_res)
print('-log(q) Validation mse with residual: ',amp_val_mse_1_over_q_with_res)
print('ensemble Validation mse with residual: ', amp_val_mse_with_res)
# print('Ensemble network validation mse',mse_model_val.mean())




with open('Amplitude_nn_predictions#5.pkl', 'wb') as f:
    pickle.dump([amp_val_predictions, amp_val_predictions_with_res, amp_val_predictions_new, amp_val_predictions_new_with_res, amp_val_predictions_ens ,amp_val_predictions_ens_with_res], f)