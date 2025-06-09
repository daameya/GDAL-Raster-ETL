from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import pandas as pd
from keras.utils.vis_utils import plot_model
import pickle as pkl
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

X_train_structured_std = pd.read_csv('/data/X_train_structured.csv', sep = ';')
X_test_structured_std = pd.read_csv('/data/X_test_structured.csv', sep = ';')

y_train_std = pd.read_csv('/data/y_train_std.csv', sep = ';')
y_test_std = pd.read_csv('/data/y_test_std.csv', sep = ';')


file = open("/data/model/y_numerical_scaler.pkl",'rb')
y_numerical_scaler = pkl.load(file)
file.close()

##### MODEL ARCHITECTURE
structured_input = layers.Input(shape=(X_train_structured_std.shape[1],)
                               , name='structured_features_input'
                              )

x = layers.Dense(X_train_structured_std.shape[1], 
                 activation='relu', 
                 kernel_initializer='he_normal')(structured_input) #80
dense_hidden = Dense(25, activation='relu', name='dense_hidden')(x)
last_layer = Dense(1, activation='linear')(dense_hidden)
    

# declare the final model inputs and outputs
final_model = Model(inputs=structured_input, outputs=last_layer)

# print a summary of the model
print(final_model.summary())

# set up learning rate decay schedule
initial_learning_rate = 0.1
lr_schedule = ExponentialDecay(
    initial_learning_rate,
        decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='/data/model/best_structured_model.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# compile the model
final_model.compile(optimizer=Adam(learning_rate=lr_schedule, epsilon=1), 
                    loss="mean_squared_error", 
                    metrics=[MeanSquaredError()])

plot_model(final_model, show_shapes=True, to_file='model1_structured.png')


results = final_model.fit(
            X_train_structured_std, y_train_std,
            epochs=500,
            callbacks=[stop, best],
            validation_data=([X_test_structured_std, y_test_std])
            )

final_model.save('/data/model/best_structured_model.h5')