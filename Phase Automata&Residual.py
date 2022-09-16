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
from keras.optimizers import Adam, Adamax

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
with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10.pkl', 'rb') as f:
    [lambda_values, coeffs, eim_basis, eim_indices] = pickle.load(f)


lambda_values[:,0]=np.log10(lambda_values[:,0])
scaler_x=StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values)
phi_train_x = lambda_values_scaled
phi_train_y = coeffs

print(phi_train_x.shape)
print(phi_train_y.shape)


with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)


lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(lambda_values_val)
phi_val_x = lambda_values_val_scaled
phi_val_y = coeffs_val
print(phi_val_x.shape)
print(phi_val_y.shape)


i=0
train_mse=[]
train_mse_before_res=[]
val_mse=[]
validation_mse_before_res=[]
final_phi_predictions = np.zeros((30000,8))
final_phi_predictions_scaled =  np.zeros((30000,8))
final_phi_predictions_before_res = np.zeros((30000,8))
final_phi_predictions_before_res_scaled = np.zeros((30000,8))
for i in range(8):
  print(i)
  train_x= phi_train_x
  train_y= phi_train_y
  train_y = train_y[:, i]

  val_x= phi_val_x
  val_y= phi_val_y
  val_y= val_y[:,i]

  phi_scaler = MinMaxScaler((0,1))

  #RESHAPING AND SCALING
  train_y = train_y.reshape(-1,1)
  train_y_scaled = phi_scaler.fit_transform(train_y)

  val_y = val_y.reshape(-1,1)
  val_y_scaled = phi_scaler.transform(val_y)

  phi_model = Sequential()
  phi_model.add(Dense(320, activation='softplus'))
  phi_model.add(Dense(320, activation='softplus'))
  phi_model.add(Dense(320, activation='softplus'))
  phi_model.add(Dense(320, activation='softplus'))
  phi_model.add(Dense(1, activation=None))

  opt = Adamax(lr=0.01)
  phi_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
  
  start = time()

  phi_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
  phi_lrm = LearningRateMonitor()
  phi_stop = EarlyStopping(monitor='mse', patience=1000)
  phi_history = phi_model.fit(train_x, train_y_scaled, validation_data=(val_x, val_y_scaled),
                              epochs=1000, batch_size=1000,
                              verbose=1, callbacks=[phi_rlrp, phi_lrm, phi_stop])

  print('Training time in minutes:', (time()-start)/60)

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

  phi_y_pred = phi_model.predict(train_x)
  phi_y_pred = np.float64(phi_y_pred)
  train_y = train_y.reshape(200000, 1)
  phi_errors = train_y_scaled - phi_y_pred
  phi_mse_train = np.mean(phi_errors ** 2)
  print(phi_mse_train)

  phi_y_pred_val = phi_model.predict(val_x)
  phi_y_pred_val = np.float64(phi_y_pred_val)
  val_y = val_y.reshape(30000, 1)
  phi_errors_val = val_y_scaled - phi_y_pred_val
  phi_mse_val = np.mean(phi_errors_val ** 2)
  print(phi_mse_val)

  # input for the residual error network
  phi_res_scaler = MinMaxScaler()
  phi_errors_new = phi_res_scaler.fit_transform(phi_errors)
  phi_errors_val_new = phi_res_scaler.transform(phi_errors_val)
  print(phi_errors_new)

  phi_res_model = Sequential()
  phi_res_model.add(Dense(320, activation='softplus'))
  phi_res_model.add(Dense(320, activation='softplus'))
  phi_res_model.add(Dense(320, activation='softplus'))
  phi_res_model.add(Dense(320, activation='softplus'))
  phi_res_model.add(Dense(1, activation=None))

  opt_res = Adamax(lr=0.01)
  phi_res_model.compile(loss='mse', optimizer=opt_res, metrics=['mse'])
  phi_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
  phi_res_lrm = LearningRateMonitor()
  phi_res_stop = EarlyStopping(monitor='mse', patience=400)
  phi_res_history = phi_res_model.fit(train_x, phi_errors_new, validation_data=(val_x, phi_errors_val_new),
                                      epochs=400, batch_size=1000,
                                      verbose=1,
                                      callbacks=[phi_res_rlrp, phi_res_lrm, phi_res_stop])

  # Plot training & validation accuracy values
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

  # # Learning Rate
  # plt.figure()
  # plt.plot(phi_res_lrm.lrates)
  # plt.yscale("log")
  # plt.title('Residual phi Learning Rate')
  # plt.xlabel('Epoch')
  # plt.savefig('Fig4: phi Res Model Learning Rate')
  # plt.show()

  # predictions from residual and reverse transform to the scaled data space
  phi_train_y_error = phi_res_scaler.inverse_transform(phi_res_model.predict(phi_train_x))
  phi_val_y_error = phi_res_scaler.inverse_transform(phi_res_model.predict(phi_val_x))

  # adding corrections from residual
  phi_train_y_pred_corrected = phi_model.predict(phi_train_x) + phi_train_y_error
  phi_val_y_pred_corrected = phi_model.predict(phi_val_x) + phi_val_y_error
  phi_val_predictions_before_residual = phi_scaler.inverse_transform(phi_model.predict(phi_val_x))
  phi_train_predictions_before_residual = phi_scaler.inverse_transform(phi_model.predict(phi_train_x))
  phi_final_train_mse_before_res = np.square((phi_model.predict(phi_train_x)) - train_y_scaled).mean()
  #
  # reverse transform to the initial data space
  phi_train_y_predictions = phi_scaler.inverse_transform(phi_train_y_pred_corrected)
  phi_val_predictions = phi_scaler.inverse_transform(phi_val_y_pred_corrected)

  phi_val_predictions_before_res_scaled = phi_model.predict(phi_val_x)
  phi_val_predictions_before_res = phi_scaler.inverse_transform(phi_val_predictions_before_res_scaled)
  phi_final_val_mse_before_res = np.square((phi_model.predict(phi_val_x)) - val_y_scaled).mean()
  #
  phi_final_train_mse = np.square(phi_train_y_pred_corrected - train_y_scaled).mean()
  phi_final_val_mse = np.square(phi_val_y_pred_corrected - val_y_scaled).mean()
  print('Train mse: ', phi_final_train_mse)
  print('Train mse before residual: ', phi_final_train_mse_before_res)
  print('Validation mse: ', phi_final_val_mse)
  print('Validation mse before residual: ', phi_final_val_mse_before_res)


  # phi_val_norm = np.square(val_y_scaled - phi_model.predict(val_x))
  # print('Validation mse:', phi_val_norm.mean())
  #
  # phi_val_predictions = scaler.inverse_transform(phi_model.predict(phi_val_x))
  final_phi_predictions[:, i] = phi_val_predictions[:, 0]
  final_phi_predictions_scaled [:, i] = phi_val_y_pred_corrected[:, 0]

  final_phi_predictions_before_res[:, i] = phi_val_predictions_before_res[:, 0]
  final_phi_predictions_before_res_scaled[:, i]= phi_val_predictions_before_res_scaled[:, 0]
  #
  # phi_val_mse = np.square(val_y - phi_val_predictions).mean()
  # print(phi_val_mse)

  folder_name = ('coeff{}'.format(i))
  save_dir = os.path.join('Phi_Automata_Residual#1', folder_name)
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)

  timestamp = str(dt.datetime.now())[:19]
  timestamp = re.sub(r'[\:-]', '', timestamp)
  timestamp = re.sub(r'[\s]', '_', timestamp)
  save_file = os.path.join(save_dir, '{}_loss{:.1e}.py'.format(timestamp, phi_final_val_mse))
  phi_model.save(os.path.join(save_dir, '{}_loss{:.1e}.h5'.format(timestamp, phi_final_val_mse_before_res)), save_format='h5')
  phi_res_model.save(os.path.join(save_dir, '{}_loss{:.1e}.h5'.format(timestamp, phi_final_val_mse)), save_format='h5')
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  with open(__file__, 'r') as f:
      with open(save_file, 'w') as out:
          for line in (f.readlines()[:-15]):
              print(line, end='', file=out)

  train_mse.append(phi_final_train_mse)
  train_mse_before_res.append(phi_final_train_mse_before_res)
  val_mse.append(phi_final_val_mse)
  validation_mse_before_res.append(phi_final_val_mse_before_res)


with open('Phase_nn_predictions_automata#1.pkl', 'wb') as f:
    pickle.dump([final_phi_predictions_before_res,final_phi_predictions_before_res_scaled, final_phi_predictions, final_phi_predictions_scaled], f)


print('Automata phase script time in minutes:', (time()-start_all)/60)

# with open('Phase_Results_automata.pkl', 'wb') as f:
#   pickle.dump([train_mse, train_mse_before_res, val_mse, validation_mse_before_res], f)
#
# with open('Network_per_Coef/With_Residual/Phi_Automata_Residual/Phase_nn_predictions.pkl', 'rb') as f:
#     [final_phi_predictions] = pickle.load(f)
#
# x = range(8)
# plt.figure()
# plt.plot(x, val_mse)
# plt.yscale("log")
# plt.title('Validation mse per coefficient')
# plt.show()
#
# final_mse_mean_np = np.asarray(val_mse)
# final_mse_mean_np = final_mse_mean_np.mean()
# print(final_mse_mean_np)