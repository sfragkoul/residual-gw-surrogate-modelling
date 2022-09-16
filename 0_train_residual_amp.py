import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

from datetime import datetime as dt
start = dt.now().time()

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


lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
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
opt = Adam(learning_rate=0.001)
amp_model.compile(loss='mse', optimizer=opt, metrics=['mse'])

# fit model
# amp_checkpoint_filepath = './Model_checkpoint_amp/weights.{epoch:02d}.hdf5'
# amp_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=amp_checkpoint_filepath,
#                                                          save_best_only=False, verbose=1,
#                                                          monitor='mse', save_freq=20 * 40)

amp_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9 , patience=15)
amp_lrm = LearningRateMonitor()
amp_stop = EarlyStopping(monitor='mse', patience=100)
amp_history = amp_model.fit(amp_train_x, amp_train_y, validation_data=(amp_val_x, amp_val_y),
                            epochs=1000, batch_size=1000,
                            verbose=1, callbacks=[amp_rlrp, amp_lrm, amp_stop])
model_time = dt.now().time()

# from tensorflow import keras
# amp_model.save('amplitude baseline model')
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
plt.savefig('Fig1: Amp Model Train MSE')
plt.show()
# #
# # # Learning Rate
plt.figure()
plt.plot(amp_lrm.lrates)
plt.yscale("log")
plt.title('Amplitude model Learning Rate')
plt.xlabel('Epoch')
# plt.savefig('Fig2: Amp Model Learning Rate ')
plt.savefig('Fig2: Amp Model Learning Rate ')
plt.show()

train_evaluation = amp_model.evaluate(amp_train_x, amp_train_y, batch_size=1000)

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

# plot train errors in 3D graph thelw #0, 2, 16
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(amp_errors[:,0], lambda_values[:, 1], lambda_values[:, 2], c=lambda_values[:, 0],  cmap='plasma' );
# ax.set_xlabel('errors')
# ax.set_ylabel('$x_1$')
# ax.set_zlabel('$x_2$')
# # plt.title('Pycbc validation mismatch for tol%' %tols)
# plt.savefig('amp error #0.pdf')
# plt.show()


# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(amp_errors[:,2], lambda_values[:, 1], lambda_values[:, 2], c=lambda_values[:, 0],  cmap='plasma' );
# ax.set_xlabel('errors')
# ax.set_ylabel('$x_1$')
# ax.set_zlabel('$x_2$')
# # plt.title('Pycbc validation mismatch for tol%' %tols)
# plt.savefig('amp error #2.pdf')
# plt.show()


# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(amp_errors[:,16], lambda_values[:, 1], lambda_values[:, 2], c=lambda_values[:, 0],  cmap='plasma' );
# ax.set_xlabel('errors')
# ax.set_ylabel('$x_1$')
# ax.set_zlabel('$x_2$')
# # plt.title('Pycbc validation mismatch for tol%' %tols)
# plt.savefig('amp error #16.pdf')
# plt.show()


# Residual Error
amp_res_model = Sequential()
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
amp_res_model.add(Dense(18, activation=None))


opt_res = Adam(learning_rate=0.001)
amp_res_model.compile(loss='mse', optimizer=opt_res, metrics=['mse'])
amp_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=50)
amp_res_lrm = LearningRateMonitor()
amp_res_stop = EarlyStopping(monitor='mse', patience=50)
amp_res_history = amp_res_model.fit(amp_train_x, amp_errors_new, validation_data=(amp_val_x, amp_errors_val_new),
                                    epochs=200, batch_size=256,
                                    verbose=1,
                                    callbacks=[amp_res_rlrp, amp_res_lrm, amp_res_stop])

res_time = dt.now().time()
# # Plot training & validation accuracy values
plt.figure()
plt.plot(amp_res_history.history['mse'])
plt.plot(amp_res_history.history['val_mse'])
plt.yscale("log")
plt.title('Residual Amp Model MSE')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('Fig3: Resdual Amp Model Train MSE')
plt.show()
# #
# # # Learning Rate
plt.figure()
plt.plot(amp_res_lrm.lrates)
plt.yscale("log")
plt.title('Amp model Residual Learning Rate')
plt.xlabel('Epoch')
plt.savefig('Fig4: Amp res Model Learning Rate')
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
amp_final_val_mse = np.square(amp_val_predictions - amp_val_y).mean()
amp_final_val_mse_before_res = np.square(amp_val_predictions_before_residual - amp_val_y).mean()
amp_final_train_mse_before_res = np.square(amp_train_predictions_before_residual - amp_train_y).mean()

print('Train mse: ',amp_final_train_mse)
print('Train mse before residual: ',  amp_final_train_mse_before_res)
print('Validation mse: ',  amp_final_val_mse)
print('Validation mse before residual: ',  amp_final_val_mse_before_res)

with open('Amplitude_nn_predictions_1.pkl', 'wb') as f:
    pickle.dump([amp_val_predictions_before_residual, amp_val_predictions ], f)

amp_model.summary()
print('Start:', start)
print('Model finish:', model_time)
print('Res model finish:', res_time)
