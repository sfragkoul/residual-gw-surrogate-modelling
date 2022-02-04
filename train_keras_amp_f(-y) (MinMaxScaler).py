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


with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10.pkl', 'rb') as f:
    [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)
# gia na tsekarw to evros twn coeffs
    print(coeffs.max())
    print(coeffs.min())

# SPLIT DATASETS
#gia kanonikopoiis khan kai green

lambda_values[:,0]=np.log10(lambda_values[:,0])
scaler_x=StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values)
amp_train_x = lambda_values_scaled
amp_train_y = coeffs
# print(amp_train_x.shape)
# print(amp_train_y.shape)

amp_scaler = MinMaxScaler()  # scaling data to (0,1)
amp_train_y_scaled = amp_scaler.fit_transform(amp_train_y)  # fitting scaler to training data .fit_transform
amp_train_y1 = 1 - amp_train_y_scaled
amp_train_y_aug=np.hstack([amp_train_y_scaled, amp_train_y1])
print(amp_train_y_aug.shape)


with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)


lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
amp_val_x = lambda_values_val_scaled
amp_val_y = coeffs_val
print(amp_val_x.shape)
print(amp_val_y.shape)

amp_val_y_scaled = amp_scaler.transform(amp_val_y)  # fitting validation data according to training data .transform
amp_val_y1 = 1 - amp_val_y_scaled
amp_val_y_aug = np.hstack([amp_val_y_scaled, amp_val_y1])
print(amp_val_y_aug.shape)


# constructing the model
amp_model = Sequential()
amp_model.add(Dense(320, activation= 'softplus'))
amp_model.add(Dense(320, activation='softplus'))
amp_model.add(Dense(320, activation='softplus'))
amp_model.add(Dense(320, activation='softplus'))
amp_model.add(Dense(36, activation=None))

# compile model
opt = Adamax(lr=0.01)
amp_model.compile(loss='mse', optimizer=opt, metrics=['mse'])

amp_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
amp_lrm = LearningRateMonitor()
amp_stop = EarlyStopping(monitor='mse', patience=1000)
amp_history = amp_model.fit(amp_train_x, amp_train_y_aug, validation_data=(amp_val_x, amp_val_y_aug),
                            epochs=1000, batch_size=1000,
                            verbose=1, callbacks=[amp_rlrp, amp_lrm, amp_stop])


# Plot training & validation learning curves
plt.figure()
plt.plot(amp_history.history['mse'])
plt.plot(amp_history.history['val_mse'])
plt.yscale("log")
plt.title('Amplitude Model Mean Square Error')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('Fig1: amp Model Train MSE')
plt.show()

# Learning Rate
plt.figure()
plt.plot(amp_lrm.lrates)
plt.yscale("log")
plt.title('Amplitude Learning Rate')
plt.xlabel('Epoch')
plt.savefig('Fig2: amp Model Learning Rate ')
plt.show()

train_evaluation = amp_model.evaluate(amp_train_x, amp_train_y_aug, batch_size=1000)


#thelw kai predictions apo validaion
amp_val_predictions_all_scaled = amp_model.predict(amp_val_x)
amp_val_predictions_18_scaled = amp_val_predictions_all_scaled[:,:18]
amp_val_predictions_36_scaled = amp_val_predictions_all_scaled[:,18:]
amp_val_predictions_36_scaled_inv = (amp_val_predictions_36_scaled*(-1))+1


amp_val_predictions_18 = amp_scaler.inverse_transform(amp_val_predictions_18_scaled)
amp_val_predictions_36_inv = amp_scaler.inverse_transform(amp_val_predictions_36_scaled_inv)
amp_val_predictions_ens  = (amp_val_predictions_18 + amp_val_predictions_36_inv)/2


amp_val_mse_18_scaled = (np.square(amp_val_y_scaled - amp_val_predictions_18_scaled)).mean()
print('Validation y space mse: ', amp_val_mse_18_scaled)

amp_val_mse_36_scaled = (np.square(amp_val_y_scaled - amp_val_predictions_36_scaled_inv)).mean()
print('Validation -y space mse: ', amp_val_mse_36_scaled)

amp_val_predictions_ens_scaled = (amp_val_predictions_18_scaled + amp_val_predictions_36_scaled_inv)/2
amp_val_mse_ens =  (np.square(amp_val_y_scaled - amp_val_predictions_ens_scaled)).mean()
print('Validation ensemble  mse: ', amp_val_mse_ens)



with open('Amplitude_nn_predictions.pkl', 'wb') as f:
    pickle.dump([amp_val_predictions_18, amp_val_predictions_36_inv, amp_val_predictions_ens], f)