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
from keras.optimizers import Adam, Adamax, SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from tensorflow.keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
# from keras.layers.initializers import RandomNormal


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

original = lambda_values.copy()
original[:, 0] = -(np.log10(original[:, 0]))


new_original = np.zeros((200000,3))
new_original[:,0] = original[:,0]
#x2 to x1
new_original[:,1] = lambda_values[:, 2]
#x1 to x2
new_original[:,2] = lambda_values[:, 1]

new_original2 =lambda_values.copy()
new_original2[:, 0] = np.log10(new_original2[:, 0])

lambda_values_new = np.concatenate((new_original2 , new_original), axis=0)
coeffs_new = np.concatenate((coeffs , coeffs), axis=0)

# lambda_values[:,0]=np.log10(lambda_values[:,0])
scaler_x=StandardScaler()
lambda_values_scaled = scaler_x.fit_transform(lambda_values_new)
phi_train_x = lambda_values_scaled
phi_train_y = coeffs_new

print(phi_train_x.shape)
print(phi_train_y.shape)
# plt.figure()
# plt.plot(phi_train_y[:, 3])
# plt.show()
phi_scaler = MinMaxScaler()  # scaling data to (0,1)
phi_train_y_scaled = phi_scaler.fit_transform(phi_train_y)  # fitting scaler to training data .fit_transform

with open('q1to8_s0.99_both/phi_rel_sur/tol_1e-10_val.pkl', 'rb') as f:
    [lambda_values_val, coeffs_val, eim_basis_val, eim_indices_val] = pickle.load(f)

original_val = lambda_values_val.copy()
original_val[:, 0] = -(np.log10(original_val[:, 0]))

new_original_val = np.zeros((30000,3))
new_original_val[:,0] = original_val[:,0]
#x2 to x1
new_original_val[:,1] = lambda_values_val[:, 2]
#x1 to x2
new_original_val[:,2] = lambda_values_val[:, 1]

new_original_val_scaled = scaler_x.transform(new_original_val)

new_original2 =lambda_values_val.copy()
new_original2[:, 0] = np.log10(new_original2[:, 0])

# lambda_values_val[:, 0] = np.log10(lambda_values_val[:, 0])
lambda_values_val_scaled = scaler_x.transform(new_original2)
phi_val_x = lambda_values_val_scaled
phi_val_y = coeffs_val
print(phi_val_x.shape)
print(phi_val_y.shape)
phi_val_y_scaled = phi_scaler.transform(phi_val_y)  # fitting validation data according to training data .transform

# constructing the model
phi_model = Sequential()
phi_model.add(Dense(320, activation= 'softplus'))
phi_model.add(Dense(320, activation='softplus'))
phi_model.add(Dense(320, activation='softplus'))
phi_model.add(Dense(320, activation='softplus'))
phi_model.add(Dense(8, activation=None))

# compile model
opt = Adamax(lr=0.01)
phi_model.compile(loss='mse', optimizer=opt, metrics=['mse'])

# fit model
# phi_checkpoint_filepath = './Model_checkpoint_phi/weights.{epoch:02d}.hdf5'
# phi_modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=phi_checkpoint_filepath,
#                                                          save_best_only=False, verbose=1,
#                                                          monitor='mse', save_freq=20 * 40)

phi_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
phi_lrm = LearningRateMonitor()
phi_stop = EarlyStopping(monitor='mse', patience=1000)
phi_history = phi_model.fit(phi_train_x, phi_train_y_scaled, validation_data=(phi_val_x, phi_val_y_scaled),
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

# # Learning Rate
# plt.figure()
# plt.plot(phi_lrm.lrates)
# plt.yscale("log")
# plt.title('Phase Learning Rate')
# plt.xlabel('Epoch')
# plt.savefig('Fig2: Phi Model Learning Rate ')
# plt.show()

# train_evaluation = phi_model.evaluate(phi_train_x, phi_train_y_scaled, batch_size=1000)


# errors from phase training
phi_y_pred = phi_model.predict(phi_train_x)
phi_y_pred = np.float64(phi_y_pred)
phi_errors = phi_train_y_scaled - phi_y_pred
phi_mse_train = np.mean(phi_errors ** 2)
print(phi_mse_train)

phi_y_pred_val = phi_model.predict(phi_val_x)
phi_y_pred_val = np.float64(phi_y_pred_val)
phi_errors_val = phi_val_y_scaled - phi_y_pred_val
phi_mse_val = np.mean(phi_errors_val ** 2)
print(phi_mse_val)

# input for the residual error network
phi_res_scaler = MinMaxScaler()
phi_errors_new = phi_res_scaler.fit_transform(phi_errors)
phi_errors_val_new = phi_res_scaler.transform(phi_errors_val)
print(phi_errors_new)


# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(phi_errors_val[:,7],lambda_values_val[:, 1], lambda_values_val[:, 2], c=lambda_values_val[:,0],  cmap='plasma' );
# # ax.scatter3D(lambda_values_val[:,0],lambda_values_val[:, 1], lambda_values_val[:, 2], c=amp_errors_val[:,0],  cmap='plasma' );
# ax.set_xlabel('errors')
# ax.set_ylabel('$x_1$')
# ax.set_zlabel('$x_2$')
# # fig.colorbar(matplotlib.cm.ScalarMappable( cmap='plasma'), orientation='vertical', ax=ax)
# plt.title(' coeff #7')
# plt.show()


# Residual Error
phi_res_model = Sequential()
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(320,activation= 'softplus'))
phi_res_model.add(Dense(8, activation=None))
#
# opt_res = Adam(lr=0.001)
opt_res = Adamax(lr=0.01)
phi_res_model.compile(loss='mse', optimizer=opt_res, metrics=['mse'])
phi_res_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=15)
phi_res_lrm = LearningRateMonitor()
phi_res_stop = EarlyStopping(monitor='mse', patience=400)
phi_res_history = phi_res_model.fit(phi_train_x, phi_errors_new, validation_data=(phi_val_x, phi_errors_val_new),
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

# phi_val_norm=np.square(phi_val_y_scaled-phi_model.predict(phi_val_x))
# print('Validation mse:',phi_val_norm.mean())

phi_val_predictions_ens = (phi_scaler.inverse_transform(phi_model.predict(phi_val_x))+ phi_scaler.inverse_transform(phi_model.predict(new_original_val_scaled)))/2
phi_val_mse_ens = np.square(phi_val_y-phi_val_predictions_ens).mean()
print('ensemble Validation mse before residual: ', phi_val_mse_ens)

phi_val_predictions_new = phi_scaler.inverse_transform(phi_model.predict(new_original_val_scaled))
phi_val_mse_1_over_q = np.square(phi_val_y-phi_val_predictions_new).mean()
print(' log(q)  Validation mse before residual: ', phi_val_mse_1_over_q)

phi_val_predictions = phi_scaler.inverse_transform(phi_model.predict(phi_val_x))
phi_val_mse = np.square(phi_val_y-phi_val_predictions).mean()
print(' -log(q)  Validation mse before residual: ', phi_val_mse)



phi_val_predictions_new_with_res = phi_scaler.inverse_transform(phi_model.predict(new_original_val_scaled)+phi_val_y_error)
phi_val_mse_1_over_q_with_res = np.square(phi_val_y - phi_val_predictions_new_with_res).mean()
print(' -log(q)  Validation mse with residual: ', phi_val_mse_1_over_q_with_res)

phi_val_predictions_with_res = phi_scaler.inverse_transform(phi_model.predict(phi_val_x)+phi_val_y_error)
phi_val_mse_with_res = np.square(phi_val_y - phi_val_predictions_with_res).mean()
print(' log(q)  Validation mse with residual: ', phi_val_mse_with_res)


phi_val_predictions_ens_with_res = (phi_val_predictions_new_with_res + phi_val_predictions_with_res)/2
phi_val_mse_ens_with_res = np.square(phi_val_y - phi_val_predictions_ens_with_res).mean()
print('ensemble Validation mse with residual: ', phi_val_mse_ens_with_res)




phi_val_predictions_new_scaled  = phi_model.predict(new_original_val_scaled)
phi_val_mse_1_over_q_scaled  = np.square(phi_val_y_scaled - phi_val_predictions_new_scaled ).mean()
print(' -log(q)  Validation mse before residual scaled: ', phi_val_mse_1_over_q_scaled )

phi_val_predictions_scaled  = phi_model.predict(phi_val_x)
phi_val_mse_scaled  = np.square(phi_val_y_scaled - phi_val_predictions_scaled ).mean()
print(' log(q)  Validation mse before residualscaled: ', phi_val_mse_scaled)

phi_val_predictions_ens_scaled = (phi_val_predictions_new_scaled + phi_val_predictions_scaled)/2
phi_val_mse_ens_scaled  = np.square(phi_val_y_scaled-phi_val_predictions_ens_scaled ).mean()
print('ensebmle Validation mse before residual scaled: ', phi_val_mse_ens_scaled )

#############################
print('Residual Results: ')

phi_val_predictions_new_scaled  = phi_model.predict(new_original_val_scaled) + phi_val_y_error
phi_val_mse_1_over_q_with_res_scaled  = np.square(phi_val_y_scaled - phi_val_predictions_new_scaled ).mean()
print(' -log(q)  Validation mse with residual scaled: ', phi_val_mse_1_over_q_with_res_scaled )

phi_val_predictions_scaled  = phi_model.predict(phi_val_x) + phi_val_y_error
phi_val_mse_with_res_scaled  = np.square(phi_val_y_scaled - phi_val_predictions_scaled ).mean()
print(' log(q)  Validation mse with residual scaled: ', phi_val_mse_with_res_scaled)

phi_val_predictions_ens_scaled = (phi_val_predictions_new_scaled + phi_val_predictions_scaled)/2
phi_val_mse_ens_with_res_scaled  = np.square(phi_val_y_scaled-phi_val_predictions_ens_scaled ).mean()
print('ensebmle Validation mse with residual scaled: ', phi_val_mse_ens_with_res_scaled )





####################################
#
# phi_train_predictions_with_res = phi_model.predict(phi_train_x[:200000, :]) + phi_train_y_error[:200000, :]
# phi_train_predictions_new_with_res = phi_model.predict(phi_train_x[200000:, :]) + phi_train_y_error[200000:, :]
# phi_train_predictions_stack =  np.concatenate((phi_train_predictions_with_res , phi_train_predictions_new_with_res), axis=1)
#
# phi_val_predictions_with_res = phi_model.predict(phi_val_x)+phi_val_y_error
# phi_val_predictions_new_with_res = phi_model.predict(new_original_val_scaled)+phi_val_y_error
# phi_val_predictions_stack =  np.concatenate((phi_val_predictions_with_res , phi_val_predictions_new_with_res), axis=1)
#
# identity = np.identity(8)
# identity_stack = np.concatenate((identity, identity), axis=0) * 0.5
# bias = np.zeros(8)
#
# # dot = np.matmul(phi_val_predictions_stack, identity_stack)
# # mse = np.square(dot-phi_val_predictions_ens_scaled).mean()
#
# # plt.figure()
# # plt.plot(dot[:1000,1],'+')
# # plt.plot(phi_val_predictions_ens_scaled[:1000,1],'+')
# # plt.show()
#
# phi_val_predictions_ens_scaled = (phi_val_predictions_new_scaled + phi_val_predictions_scaled)/2
# phi_val_mse_ens_with_res_scaled  = np.square(phi_val_y_scaled-phi_val_predictions_ens_scaled ).mean()
# print('ensebmle Validation mse with residual scaled: ', phi_val_mse_ens_with_res_scaled )
#
# # initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=0)
# phi_ens_model = Sequential()
# # phi_ens_model.add(Dense(8, activation=None, kernel_initializer=initializer))
# phi_ens_model.add(Dense(8, activation=None))
# phi_ens_model.build((None,16))
# phi_ens_model.layers[0].set_weights((identity_stack, bias))
#
# # phi_ens_model.summary()
#
# # opt_ens = Adam(lr=0.001)
# opt_ens = SGD(lr=0.00001)
# phi_ens_model.compile(loss='mse', optimizer=opt_ens, metrics=['mse'])
# phi_ens_rlrp = ReduceLROnPlateau(monitor='mse', factor=0.9, patience=50)
# phi_ens_lrm = LearningRateMonitor()
# phi_ens_stop = EarlyStopping(monitor='mse', patience=200)
# phi_ens_history = phi_ens_model.fit(phi_train_predictions_stack, phi_train_y_scaled[:200000,:], validation_data=(phi_val_predictions_stack, phi_val_y_scaled[:200000,:]),
#                                     epochs=200, batch_size=1000,
#                                     verbose=1,
#                                    callbacks=[phi_ens_rlrp, phi_ens_lrm, phi_ens_stop])
#
# # wght = phi_ens_model.layers[0].get_weights()[0]
#
#
# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(phi_ens_history.history['mse'])
# plt.plot(phi_ens_history.history['val_mse'])
# plt.yscale("log")
# plt.title('ensemble phi Model Mean Square Error')
# plt.ylabel('mse')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('Fig3: phi ENS Model Train MSE')
# plt.show()
#
#
# # COEFFS_scaled = np.concatenate((phi_train_y_scaled[:200000,:], phi_train_y_scaled[:200000,:]), axis=1)
# predictions_model_ens = phi_ens_model.predict(phi_train_predictions_stack)
# mse_model = np.square(phi_train_y_scaled[:200000,:] - predictions_model_ens)
# print('ensemble network training mse', mse_model.mean())
#
#
# # COEFFS_val_scaled = np.concatenate((phi_val_y_scaled[:200000,:], phi_val_y_scaled[:200000,:]), axis=1)
# predictions_model_ens_val_scaled = phi_ens_model.predict(phi_val_predictions_stack)
# mse_model_val = np.square(phi_val_y_scaled[:200000,:] - predictions_model_ens_val_scaled)
# print('ensemble network validation mse',mse_model_val.mean())
#
# predictions_model_ens_val = phi_scaler.inverse_transform(predictions_model_ens_val_scaled)
# mse_ens_inversed = np.square(phi_val_y-predictions_model_ens_val)
# print(mse_ens_inversed.mean())
#
# print('ensebmle Validation mse with residual scaled: ', phi_val_mse_ens_scaled )


print('Phase')
print(' log(q)  Validation mse before residual: ', phi_val_mse_scaled)
print(' -log(q)  Validation mse before residual: ', phi_val_mse_1_over_q_scaled)
print('ensemble Validation mse before residual: ', phi_val_mse_ens_scaled)
print(' log(q)  Validation mse with residual: ', phi_val_mse_with_res_scaled)
print(' -log(q)  Validation mse with residual: ', phi_val_mse_1_over_q_with_res_scaled)
print('ensemble Validation mse with residual: ', phi_val_mse_ens_with_res_scaled)
# print('ensemble network validation mse',mse_model_val.mean())



with open('Phase_nn_predictions#5.pkl', 'wb') as f:
    pickle.dump([phi_val_predictions, phi_val_predictions_with_res, phi_val_predictions_new, phi_val_predictions_new_with_res, phi_val_predictions_ens , phi_val_predictions_ens_with_res], f)


