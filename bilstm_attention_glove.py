#Import Libraries

import pandas as pd
import numpy as np
import pickle as pkl

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# keras imports
from keras.models import Model


# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from keras.layers import Input, Attention, Embedding, Dropout, Dense , Bidirectional, LSTM
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
#from keras.utils import plot_model, Sequence
from tensorflow.keras.optimizers.schedules import ExponentialDecay


from keras.utils.vis_utils import plot_model


from keras.preprocessing.sequence import pad_sequences


from category_encoders import *


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim = 32, num_heads = 2, ff_dim = 32, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen = 100, vocab_size = 1000, embed_dim = 125):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

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


X_train_text = pd.read_csv('/data/X_train_text.csv', sep = ';')
X_test_text = pd.read_csv('/data/X_test_text.csv', sep = ';')

y_train_std = pd.read_csv('/data/y_train_std.csv', sep = ';')
y_test_std = pd.read_csv('/data/y_test_std.csv', sep = ';')

del X_train_text['Unnamed: 0']
del X_test_text['Unnamed: 0']

print('TRAIN', len(X_train_text), len(y_train_std))
print('TEST', len(X_test_text), len(y_test_std))


prepare_text_data_train(X_train_text)
train_tokens_pad, train_tokens, vocab_size, maxlen = prepare_text_data_inference(X_train_text)
test_tokens_pad, test_tokens, vocab_size, maxlen = prepare_text_data_inference(X_test_text)

file = open("/data/model/text_tokenizer.pkl",'rb')
tokenizer = pkl.load(file)
file.close()



##### LOAD GLOVE EMBEDDINGS
print('glove load...')
# identify the embedding filename; we are using the Glove 42B 300d embeddings
glove_file = "/data/model/glove.840B.300d.txt"
print('glove loaded!')
# create the embeddings index dictionary
embeddings_index = {} # create a lookup dictionary to store words and their vectors
f = open(glove_file, errors='ignore')# open our embedding file 
for line in f: # for each line in the file
    values = line.split(' ') #split the line on spaces between the word and its vectors
    word = values[0] # the word is the first entry
    if word in tokenizer.word_index.keys(): # we check if the word is in our tokenizer word index
        coefs = np.asarray(values[1:], dtype='float32') # if so, get the word's vectors
        embeddings_index[word] = coefs # add the word and its vectors to the embeddings_index dictionary
f.close()
print('Found %s word vectors.' % len(embeddings_index)) # report how many words in our corpus were found in the GloVe words

# amount of vocabulary to use, will pick the top 10000 words seen in the corpus
features = 10000
# max text sequence length, must match tokens in transfer file, we are using glove 300d so it is 300
max_words = 300

num_tokens = (len(tokenizer.word_index) + 1) # for num tokens we always do the length of our word index +1 for a pad token
hits = 0
misses = 0
embedding_dim = 300

embedding_matrix = np.zeros((num_tokens, max_words)) # setting up an array for our tokens with a row per token and 300 columns
for word, i in tokenizer.word_index.items(): # for each word in the tokenizer word index
    embedding_vector = embeddings_index.get(word) #get the vector from the embeddings index dictionary
    if embedding_vector is not None: # if the vector isn't None,
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # store the embedding vector in the matrix at that index
        hits += 1
    else:
        misses += 1        
print("Converted %d words (%d misses)" % (hits, misses))


####### CREATING TEXT NEURAL NETWORK ARCHITECTURE WITH GLOVE EMBEDDINGS
# for text embeddings the input shape can be none
nlp_input = Input(shape=(maxlen,), name='nlp_input')

# uses the embedding matrix dictionary to create word embeddings for the inputs
embedded_sequences = Embedding(
                input_dim = num_tokens, # number of unique tokens
                output_dim = embedding_dim, #number of features
                embeddings_initializer=Constant(embedding_matrix), # initialize with Glove embeddings
                input_length=maxlen, 
                trainable=False)(nlp_input)
# uses two bi-directional LSTM layers
lstm = Bidirectional(LSTM(300, return_sequences=True))(embedded_sequences)
att_lstm = Attention(units=300)(lstm)
nlp_out = Dropout(0.5)(att_lstm)
# adds a dense layer
#output_embed = Dense(350, activation="relu", kernel_initializer='he_normal')(x)
last_layer = Dense(1, activation='linear')(nlp_out)

# declare the final model inputs and outputs
final_model = Model(inputs=nlp_input, outputs=last_layer)

# print a summary of the model
print(final_model.summary())

# set up learning rate decay schedule
initial_learning_rate = 0.1
lr_schedule = ExponentialDecay(
    initial_learning_rate,
        decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='/data/model/best_nlp_model.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# compile the model
final_model.compile(optimizer=Adam(learning_rate=lr_schedule, epsilon=1), 
                    loss="mean_squared_error", 
                    metrics=[MeanSquaredError()])

plot_model(final_model, show_shapes=True, to_file='model2_nlp_attention.png')

results = final_model.fit(
            train_tokens_pad, y_train_std,
            epochs=500,
            callbacks=[stop, best],
            validation_data=([test_tokens_pad, y_test_std])
            )

final_model.save('/data/model/pretrained_nlp_model.h5')


