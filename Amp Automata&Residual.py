import pickle
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
import datetime as dt
import re
from time import time

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

start_all = time()
with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10.pkl', 'rb') as f:
  [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)

lambda_values[:, 0] = np.log10(lambda_values[:, 0])
scaler_x = StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values)
amp_train_x = lambda_values_scaled
amp_train_y = coeffs

print(amp_train_x.shape)
print(amp_train_y.shape)


with open('q1to8_s0.99_both/amp_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
  [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)

lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
amp_val_x = lambda_values_val_scaled
amp_val_y = coeffs_val
print(amp_val_x.shape)
print(amp_val_y.shape)

i=0
train_mse=[]
train_mse_before_res=[]
val_mse = []
validation_mse_before_res = []
final_amp_predictions = np.zeros((30000,18))
final_amp_predictions_before_res = np.zeros((30000,18))
for i in range(18):
  print(i)
  train_x= amp_train_x
  train_y= amp_train_y
  train_y = train_y[:, i]

  val_x= amp_val_x
  val_y= amp_val_y
  val_y= val_y[:,i]

  amp_model = Sequential()
  amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
  amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
  amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
  amp_model.add(Dense(320, activation=tf.keras.layers.ReLU()))
  amp_model.add(Dense(1, activation=None))


  opt = Adam(lr=0.001)
  amp_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
  
  start = time()

  amp_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
  amp_lrm = LearningRateMonitor()
  amp_stop = EarlyStopping(monitor='mse', patience=1000)
  amp_history = amp_model.fit(train_x, train_y, validation_data=(val_x, val_y),
                              epochs=1000, batch_size=1000,
                              verbose=1, callbacks=[amp_rlrp, amp_lrm, amp_stop])

  print('Training time in minutes :', (time()-start)/60)

  # plt.figure()
  # plt.plot(amp_history.history['mse'])
  # plt.plot(amp_history.history['val_mse'])
  # plt.yscale("log")
  # plt.title('Amplitude Model Mean Square Error')
  # plt.ylabel('mse')
  # plt.xlabel('Epoch')
  # plt.legend(['Train', 'Validation'], loc='upper right')
  # # plt.savefig('Fig1: Amp Model Train MSE')
  # plt.savefig('Fig1: Amp  Model Train MSE')
  # plt.show()

  # errors from amplitude training
  amp_y_pred = amp_model.predict(train_x)
  amp_y_pred = np.float64(amp_y_pred)
  train_y = train_y.reshape(200000,1)
  amp_errors = train_y - amp_y_pred
  amp_mse_train = np.mean(amp_errors ** 2)
  print('train errors: ',amp_mse_train)

  amp_y_pred_val = amp_model.predict(val_x)
  amp_y_pred_val = np.float64(amp_y_pred_val)
  val_y = val_y.reshape(30000, 1)
  amp_errors_val = val_y - amp_y_pred_val
  amp_mse_val = np.mean(amp_errors_val ** 2)
  print('validation errors: ',amp_mse_val)
  #
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
  amp_res_model.add(Dense(1, activation=None))

  opt_res = Adam(lr=0.001)
  amp_res_model.compile(loss='mse', optimizer=opt_res, metrics=['mse'])
  amp_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=50)
  amp_res_lrm = LearningRateMonitor()
  amp_res_stop = EarlyStopping(monitor='mse', patience=300)
  amp_res_history = amp_res_model.fit(train_x, amp_errors_scaled,
                                      validation_data=(val_x, amp_errors_val_scaled),
                                      epochs=300, batch_size=1000,
                                      verbose=1,
                                      callbacks=[amp_res_rlrp, amp_res_lrm, amp_res_stop])

  # Plot training & validation accuracy values
  # plt.figure()
  # plt.plot(amp_res_history.history['mse'])
  # plt.plot(amp_res_history.history['val_mse'])
  # plt.yscale("log")
  # plt.title('Residual Amp Model MSE')
  # plt.ylabel('mse')
  # plt.xlabel('Epoch')
  # plt.legend(['Train', 'Validation'], loc='upper right')
  # plt.savefig('Fig3: Amp Res Model Train MSE')
  # plt.show()

  # # Learning Rate
  # plt.figure()
  # plt.plot(amp_res_lrm.lrates)
  # plt.yscale("log")
  # plt.title('Amp Residual Learning Rate')
  # plt.xlabel('Epoch')
  # plt.savefig('Fig4: Amp Res Model Learning Rate')
  # plt.show()


  amp_train_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(train_x))
  amp_val_y_error = amp_res_scaler.inverse_transform(amp_res_model.predict(val_x))
  # # adding corrections from residual
  amp_train_y_pred = amp_model.predict(train_x) + amp_train_y_error
  amp_val_predictions = amp_model.predict(val_x) + amp_val_y_error
  amp_val_predictions_before_residual = amp_model.predict(val_x)
  amp_train_predictions_before_residual = amp_model.predict(train_x)
  #
  amp_final_train_mse = np.square(amp_train_y_pred - train_y).mean()
  amp_final_mse_before_res = np.square(amp_train_predictions_before_residual - train_y).mean()
  amp_final_val_mse = np.square(amp_val_predictions - val_y).mean()
  amp_final_val_mse_before_res = np.square(amp_val_predictions_before_residual - val_y).mean()



  # amp_val_predictions = amp_model.predict(val_x)
  final_amp_predictions[:,i] = amp_val_predictions[:,0]
  final_amp_predictions_before_res[:,i] = amp_val_predictions_before_residual[:,0]


  folder_name = ('coeff{}'.format(i))
  save_dir = os.path.join('Amp_Automata_Residual#1', folder_name)
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)

  timestamp = str(dt.datetime.now())[:19]
  timestamp = re.sub(r'[\:-]', '', timestamp)
  timestamp = re.sub(r'[\s]', '_', timestamp)
  save_file = os.path.join(save_dir, '{}_loss{:.1e}.py'.format(timestamp, amp_final_val_mse))
  amp_model.save(os.path.join(save_dir, '{}_loss{:.1e}.h5'.format(timestamp, amp_final_val_mse_before_res)),save_format='h5')
  amp_res_model.save(os.path.join(save_dir, 'res{}_loss{:.1e}.h5'.format(timestamp, amp_final_val_mse)), save_format='h5')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  with open(__file__, 'r') as f:
    with open(save_file, 'w') as out:
        for line in (f.readlines()[:-15]):
            print(line, end='', file=out)

  train_mse.append(amp_final_train_mse)
  train_mse_before_res.append(amp_final_mse_before_res)
  val_mse.append(amp_final_val_mse)
  validation_mse_before_res.append(amp_final_val_mse_before_res)

with open('Amplitude_nn_predictions_automata#1.pkl', 'wb') as f:
    pickle.dump([final_amp_predictions_before_res, final_amp_predictions], f)

print('Automata amplitude script time in minutes:', (time()-start_all)/60)

#
final_mse_mean_np = np.asarray(val_mse)
final_mse_mean_np = final_mse_mean_np.mean()
print(final_mse_mean_np)