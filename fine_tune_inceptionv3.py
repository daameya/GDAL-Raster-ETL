import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16


from keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.utils.vis_utils import plot_model

from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D,MaxPooling1D
from keras.layers import Activation,Dropout,Flatten,BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

X_train_image = pd.read_csv('/data/X_train_image.csv', sep = ';')
X_test_image = pd.read_csv('/data/X_test_image.csv', sep = ';')

y_train_std = pd.read_csv('/data/y_train_std.csv', sep = ';')
y_test_std = pd.read_csv('/data/y_test_std.csv', sep = ';')


X_train_image['homeImage'] = X_train_image['homeImage'].apply(lambda x: '/data/img/'+str(x))
X_train_image['price'] = y_train_std.values
X_train_image

X_test_image['homeImage'] = X_test_image['homeImage'].apply(lambda x: '/data/img/'+str(x))
X_test_image['price'] = y_test_std.values
X_test_image

img_size_shape = 128

from keras.preprocessing.image import ImageDataGenerator
image_train_generator = ImageDataGenerator(
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
        )
    
# test/val have only the pixel data normalization
image_test_generator = ImageDataGenerator(rescale = 1./1)
    
# visualize an image augmentation sample
#visualize_augmentations(image_train_generator, images_train.iloc[1])

# specify where the train generator pulls batches
image_train_generator = image_train_generator.flow_from_dataframe(
        dataframe=X_train_image,
        x_col="homeImage",  # this is where your image data is stored
        y_col="price",  # this is your target feature
        class_mode="raw",  # use "raw" for regressions
        color_mode='rgb',
        target_size=(img_size_shape, img_size_shape),
        batch_size=32, # increase or decrease to fit your GPU,
        )

# specify where the test generator pulls batches
image_test_generator = image_test_generator.flow_from_dataframe(
        dataframe=X_test_image,
        x_col="homeImage",
        y_col="price",
        class_mode="raw",
        color_mode='rgb',
        target_size=(img_size_shape, img_size_shape),
        batch_size=32,
        )



model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(img_size_shape, img_size_shape, 3)))

for layer in model.layers:
    layer.trainable=False

img_input = Input(shape=(img_size_shape, img_size_shape, 3))
x = model(img_input, training=False)
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
output_cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense_hidden')(x) #256
last_layer = Dense(1, activation='linear')(output_cnn)

model = Model(inputs=img_input, outputs=last_layer)

batch_size = 64 
epochs = 100 
verbose = 1
        

initial_learning_rate = 0.1 

"""
lr_schedule = ExponentialDecay(
            initial_learning_rate,
                decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
"""

model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1),
    metrics=['mse'],
    loss='mse'
)

model.summary()

plot_model(model, show_shapes=True, to_file='inceptionV3_image.png')

stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='/data/model/best_iv3_model_simple.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history = model.fit(
    image_train_generator, 
    validation_data = image_test_generator,
    batch_size=batch_size,
    verbose=verbose,
    epochs=epochs,
    callbacks=[stop, best]
)


model.save('/data/model/best_iv3_model_simple.h5')