import pandas as pd
import numpy as np
import pickle as pkl

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import BatchNormalization, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, Activation, Dropout, Dense, Input, Multiply
from keras.models import Sequential, Model
from keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History, LearningRateScheduler
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
#from keras.utils import plot_model, Sequence
from keras.initializers import Constant
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from attention import Attention
import keras

X_train_structured_std = pd.read_csv('/data/X_train_structured.csv', sep = ';')
X_test_structured_std = pd.read_csv('/data/X_test_structured.csv', sep = ';')

X_train_image = pd.read_csv('/data/X_train_image.csv', sep = ';')
X_test_image = pd.read_csv('/data/X_test_image.csv', sep = ';')

X_train_text = pd.read_csv('/data/X_train_text.csv', sep = ';')
X_test_text = pd.read_csv('/data/X_test_text.csv', sep = ';')

del X_train_text['Unnamed: 0']
del X_test_text['Unnamed: 0']



y_train_std = pd.read_csv('/data/y_train_std.csv', sep = ';')
y_test_std = pd.read_csv('/data/y_test_std.csv', sep = ';')


print('TRAIN', len(X_train_structured_std), len(X_train_image), len(X_train_text), len(y_train_std))
print('TEST', len(X_test_structured_std), len(X_test_image), len(X_test_text), len(y_test_std))

#file = open("/data/model/X_numerical_scaler.pkl",'rb')
#X_numerical_scaler = pkl.load(file)
#file.close()

file = open("/data/model/y_numerical_scaler.pkl",'rb')
y_numerical_scaler = pkl.load(file)
file.close()

X_train_image['homeImage'] = X_train_image['homeImage'].apply(lambda x: '/data/img/'+str(x))
X_train_image['price'] = y_train_std.values
X_train_image

X_test_image['homeImage'] = X_test_image['homeImage'].apply(lambda x: '/data/img/'+str(x))
X_test_image['price'] = y_test_std.values
X_test_image




def load_house_images(df, image_size):
    # initialize our images array (i.e., the house images themselves)
    images = []
    # loop over the indexes of the houses
    for i, image_file in enumerate(df['homeImage'].values):
        
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        # initialize our list of input images along with the output image
        # after *combining* the four input images
        inputImages = []
        # loop over the input house paths
        img = load_img(image_file, target_size=(image_size, image_size))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        
        if i%1000 == 0:
            print(i, image_file, img.shape)
        elif i == 0:
            print(i, image_file, img.shape)
            
        images.append(img)
        # tile the four input images in the output image such the first
        # image goes in the top-right corner, the second image in the
        # top-left corner, the third image in the bottom-right corner,
        # and the final image in the bottom-left corner
        # return our set of images
    images = np.array(images)
    images = images.reshape((len(images), image_size, image_size, 3))
    return images



X_train_image_array = load_house_images(X_train_image, image_size = 128)
X_test_image_array = load_house_images(X_test_image, image_size = 128)

pkl.dump(X_train_image_array, open('/data/model/X_train_image_array.pkl','wb'))
pkl.dump(X_test_image_array, open('/data/model/X_test_image_array.pkl','wb'))


file = open("/data/model/X_train_image_array.pkl",'rb')
X_train_image_array = pkl.load(file)
file.close()
        
file = open("/data/model/X_test_image_array.pkl",'rb')
X_test_image_array = pkl.load(file)
file.close()


def prepare_text_data_train(dataf):
    for col in dataf.columns:
        dataf[col] = dataf[col].str.lower()
        
        tokenizer = Tokenizer(num_words=2096) #5000
        tokenizer.fit_on_texts(dataf[col])
        pkl.dump(tokenizer, open('/data/model/text_tokenizer.pkl','wb'))
        
        vocab_size = len(tokenizer.word_index) + 1 
        print(vocab_size)
        
        pkl.dump(vocab_size, open('/data/model/text_vocab_size.pkl','wb'))

        maxlen = int(np.mean(dataf[col].str.len()))
        pkl.dump(maxlen, open('/data/model/text_maxlen.pkl','wb'))

def prepare_text_data_inference(dataf):
    for col in dataf.columns:
        dataf[col] = dataf[col].str.lower()
        
        file = open("/data/model/text_tokenizer.pkl",'rb')
        tokenizer = pkl.load(file)
        file.close()
        
        data_tokens = tokenizer.texts_to_sequences(dataf[col])
        
        file = open("/data/model/text_vocab_size.pkl",'rb')
        vocab_size = pkl.load(file)
        file.close()
        
        file = open("/data/model/text_maxlen.pkl",'rb')
        maxlen = pkl.load(file)
        file.close()

        data_tokens_pad = pad_sequences(data_tokens, padding='post', maxlen=max(maxlen, 300), truncating='post')
        
    return data_tokens_pad, data_tokens, vocab_size, max(maxlen, 300)

prepare_text_data_train(X_train_text)
train_tokens_pad, train_tokens, vocab_size, maxlen = prepare_text_data_inference(X_train_text)
test_tokens_pad, test_tokens, vocab_size, maxlen = prepare_text_data_inference(X_test_text)

file = open("/data/model/text_tokenizer.pkl",'rb')
tokenizer = pkl.load(file)
file.close()



fine_tuned_inceptionv3 = keras.models.load_model('/data/model/best_iv3_model_simple.hdf5', compile=False)
fine_tuned_glove_bilstm_att = keras.models.load_model('/data/model/best_nlp_model.hdf5', compile=False, custom_objects={'Attention': Attention})
fine_tuned_structured = keras.models.load_model('/data/model/best_structured_model.hdf5', compile=False)

fine_tuned_inceptionv3_layer = Model(inputs=fine_tuned_inceptionv3.input, outputs=fine_tuned_inceptionv3.layers[-2].output)
fine_tuned_glove_bilstm_att_layer = Model(inputs=fine_tuned_glove_bilstm_att.input, outputs=fine_tuned_glove_bilstm_att.layers[-2].output)
fine_tuned_structured_layer = Model(inputs=fine_tuned_structured.input, outputs=fine_tuned_structured.layers[-2].output)

for layer in fine_tuned_structured_layer.layers:
    layer._name = layer.name + '_structured'

for layer in fine_tuned_glove_bilstm_att_layer.layers:
    layer._name = layer.name + '_nlp'
    
for layer in fine_tuned_inceptionv3_layer.layers:
    layer._name = layer.name + '_img'
    
x = keras.layers.concatenate([fine_tuned_inceptionv3_layer.output, 
                        fine_tuned_glove_bilstm_att_layer.output,
                        fine_tuned_structured_layer.output])
x = Dense(50, activation='relu', name='dense_hidden_final')(x)
last_layer = Dense(1, activation='linear')(x)


model = keras.Model(inputs=[fine_tuned_inceptionv3_layer.input, 
                            fine_tuned_glove_bilstm_att_layer.input,
                            fine_tuned_structured_layer.input
                           ], 
                    outputs=[last_layer])

initial_learning_rate = 0.1

'''
lr_schedule = ExponentialDecay(
            initial_learning_rate,
                decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
'''

model.compile(optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1), loss="mean_squared_error", metrics=[ MeanSquaredError()])
print(model.summary())

from tensorflow.keras.utils import plot_model
plot_model(model, to_file="final_model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    dpi=350)

print('saving...')
model.save('/data/model/best_final_model_simple.h5')
print('saved DONE!')

print('loading...')
reconstructed_model = keras.models.load_model("/data/model/best_final_model_simple.h5", compile=False, custom_objects={'Attention': Attention})
print('loaded DONE!')

plot_model(reconstructed_model, show_shapes=True, to_file='final_model_image.png')

reconstructed_model.compile(optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1), loss="mean_squared_error", metrics=[ MeanSquaredError()])
print(reconstructed_model.summary())

stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='/data/model/best_final_model_weights_only.h5', 
                       save_best_only=True, 
                       save_weights_only=False, 
                       monitor='val_loss', 
                       mode='min', verbose=1)

results = reconstructed_model.fit([X_train_image_array,
                    train_tokens_pad, 
                    X_train_structured_std],
                    y_train_std,                   
                    epochs=500,
                    batch_size = 250,
                    validation_data=([X_test_image_array,
                                      test_tokens_pad, 
                                       X_test_structured_std], 
                                      y_test_std),
                    callbacks=[stop, best],
                    )

results.save('/data/model/final_hybrid_model.h5')