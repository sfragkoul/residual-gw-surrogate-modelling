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
# print(phi_train_x.shape)
# print(phi_train_y.shape)

phi_scaler = MinMaxScaler()  # scaling data to (0,1)
phi_train_y_scaled = phi_scaler.fit_transform(phi_train_y)  # fitting scaler to training data .fit_transform
phi_train_y1 = 1 - phi_train_y_scaled
phi_train_y_aug=np.hstack([phi_train_y_scaled, phi_train_y1])
print(phi_train_y_aug.shape)


with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)


lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
phi_val_x = lambda_values_val_scaled
phi_val_y = coeffs_val
print(phi_val_x.shape)
print(phi_val_y.shape)

phi_val_y_scaled = phi_scaler.transform(phi_val_y)  # fitting validation data according to training data .transform
phi_val_y1 = 1 - phi_val_y_scaled
phi_val_y_aug = np.hstack([phi_val_y_scaled, phi_val_y1])
print(phi_val_y_aug.shape)


# constructing the model
phi_model = Sequential()
phi_model.add(Dense(320, activation= 'softplus'))
phi_model.add(Dense(320, activation='softplus'))
phi_model.add(Dense(320, activation='softplus'))
phi_model.add(Dense(320, activation='softplus'))
phi_model.add(Dense(16, activation=None))

# compile model
opt = Adamax(lr=0.01)
phi_model.compile(loss='mse', optimizer=opt, metrics=['mse'])

phi_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
phi_lrm = LearningRateMonitor()
phi_stop = EarlyStopping(monitor='mse', patience=1000)
phi_history = phi_model.fit(phi_train_x, phi_train_y_aug, validation_data=(phi_val_x, phi_val_y_aug),
                            epochs=1000, batch_size=1000,
                            verbose=1, callbacks=[phi_rlrp, phi_lrm, phi_stop])


# Plot training & validation learning curves
plt.figure()
plt.plot(phi_history.history['mse'])
plt.plot(phi_history.history['val_mse'])
plt.yscale("log")
plt.title('Phase Model Mean Square Error')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('Fig1: Phi Model Train MSE')
plt.show()

# Learning Rate
plt.figure()
plt.plot(phi_lrm.lrates)
plt.yscale("log")
plt.title('Phase Learning Rate')
plt.xlabel('Epoch')
plt.savefig('Fig2: Phi Model Learning Rate ')
plt.show()

train_evaluation = phi_model.evaluate(phi_train_x, phi_train_y_aug, batch_size=1000)


#thelw kai predictions apo validaion
phi_val_predictions_all_scaled = phi_model.predict(phi_val_x)
phi_val_predictions_8_scaled = phi_val_predictions_all_scaled[:,:8]
phi_val_predictions_16_scaled = phi_val_predictions_all_scaled[:,8:]
phi_val_predictions_16_scaled_inv = (phi_val_predictions_16_scaled*(-1))+1


phi_val_predictions_8 = phi_scaler.inverse_transform(phi_val_predictions_8_scaled)
phi_val_predictions_16_inv = phi_scaler.inverse_transform(phi_val_predictions_16_scaled_inv)
phi_val_predictions_ens  = (phi_val_predictions_8 + phi_val_predictions_16_inv)/2


phi_val_mse_8_scaled = (np.square(phi_val_y_scaled - phi_val_predictions_8_scaled)).mean()
print('Validation y space mse: ', phi_val_mse_8_scaled)

phi_val_mse_16_scaled = (np.square(phi_val_y_scaled - phi_val_predictions_16_scaled_inv)).mean()
print('Validation -y space mse: ', phi_val_mse_16_scaled)

phi_val_predictions_ens_scaled = (phi_val_predictions_8_scaled + phi_val_predictions_16_scaled_inv)/2
phi_val_mse_ens =  (np.square(phi_val_y_scaled - phi_val_predictions_ens_scaled)).mean()
print('Validation ensemble  mse: ', phi_val_mse_ens)



with open('Phase_nn_predictions.pkl', 'wb') as f:
    pickle.dump([phi_val_predictions_8, phi_val_predictions_16_inv, phi_val_predictions_ens], f)