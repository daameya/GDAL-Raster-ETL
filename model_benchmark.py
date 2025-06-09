import pandas as pd
import numpy as np
import pickle as pkl
from attention import Attention
import keras
from sklearn import metrics

print('loading...')
final_model = keras.models.load_model("/data/model/best_final_model_weights_only.h5", compile=False, custom_objects={'Attention': Attention})
fine_tuned_inceptionv3 = keras.models.load_model('/data/model/best_iv3_model_simple.hdf5', compile=False)
fine_tuned_glove_bilstm_att = keras.models.load_model('/data/model/best_nlp_model.hdf5', compile=False, custom_objects={'Attention': Attention})
fine_tuned_structured = keras.models.load_model('/data/model/best_structured_model.hdf5', compile=False)
fine_tuned_structured_nlp = keras.models.load_model('/data/model/best_model_struct_nlp.h5', compile=False, custom_objects={'Attention': Attention})
print('loaded DONE!')

X_train_structured_std = pd.read_csv('/data/X_train_structured.csv', sep = ';')
X_test_structured_std = pd.read_csv('/data/X_test_structured.csv', sep = ';')

file = open("/data/model/X_train_image_array.pkl",'rb')
X_train_image_array = pkl.load(file)
file.close()
        
file = open("/data/model/X_test_image_array.pkl",'rb')
X_test_image_array = pkl.load(file)
file.close()

X_train_text = pd.read_csv('/data/X_train_text.csv', sep = ';')
X_test_text = pd.read_csv('/data/X_test_text.csv', sep = ';')

del X_train_text['Unnamed: 0']
del X_test_text['Unnamed: 0']

y_train_std = pd.read_csv('/data/y_train_std.csv', sep = ';')
y_test_std = pd.read_csv('/data/y_test_std.csv', sep = ';')

file = open("/data/model/y_numerical_scaler.pkl",'rb')
y_numerical_scaler = pkl.load(file)
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

y_pred_img = fine_tuned_inceptionv3.predict(X_test_image_array)
y_pred_nlp = fine_tuned_glove_bilstm_att.predict(test_tokens_pad)
y_pred_structured = fine_tuned_structured.predict(X_test_structured_std)
y_pred_structured_nlp = fine_tuned_structured_nlp.predict([test_tokens_pad, X_test_structured_std])
y_pred_all = final_model.predict([X_test_image_array, test_tokens_pad, X_test_structured_std])

y_true = y_numerical_scaler.inverse_transform(y_test_std)
y_true_train = y_numerical_scaler.inverse_transform(y_train_std)
y_pred_img = y_numerical_scaler.inverse_transform(y_pred_img)
y_pred_nlp = y_numerical_scaler.inverse_transform(y_pred_nlp)
y_pred_structured = y_numerical_scaler.inverse_transform(y_pred_structured)
y_pred_structured_nlp = y_numerical_scaler.inverse_transform(y_pred_structured_nlp)
y_pred_all = y_numerical_scaler.inverse_transform(y_pred_all)

y_pred_compare = pd.DataFrame(y_true)
y_pred_compare.columns = ['Y_TRUE']
y_pred_compare['Y_PRED_STRUCTURED'] = y_pred_structured
y_pred_compare['Y_PRED_NLP'] = y_pred_nlp
y_pred_compare['Y_PRED_IMG'] = y_pred_img
y_pred_compare['Y_PRED_HYBRID_NLP'] = y_pred_structured_nlp
y_pred_compare['Y_PRED_HYBRID_ALL'] = y_pred_all
y_pred_compare['MEAN'] = np.mean(y_true_train.ravel())
print('MEAN price:', np.mean(y_true_train.ravel()))


print('R2')
print('STRUCTURED', metrics.r2_score(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_STRUCTURED']))
print('NLP', metrics.r2_score(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_NLP']))
print('IMG', metrics.r2_score( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_IMG']))
print('HYBRID_ALL', metrics.r2_score( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_ALL']))
print('HYBRID_NLP', metrics.r2_score( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_NLP']))
print('DUMMY_MEAN_MODEL', metrics.r2_score( y_pred_compare['Y_TRUE'], y_pred_compare['MEAN']))
print('---------------')
print('RMSE')
print('STRUCTURED', np.sqrt(metrics.mean_squared_error(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_STRUCTURED'])))
print('NLP', np.sqrt(metrics.mean_squared_error(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_NLP'])))
print('IMG', np.sqrt(metrics.mean_squared_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_IMG'])))
print('HYBRID_ALL',np.sqrt( metrics.mean_squared_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_ALL'])))
print('HYBRID_NLP', np.sqrt(metrics.mean_squared_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_NLP'])))
print('DUMMY_MEAN_MODEL', np.sqrt(metrics.mean_squared_error( y_pred_compare['Y_TRUE'], y_pred_compare['MEAN'])))
print('---------------')
print('MAE')
print('STRUCTURED', metrics.mean_absolute_error(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_STRUCTURED']))
print('NLP', metrics.mean_absolute_error(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_NLP']))
print('IMG', metrics.mean_absolute_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_IMG']))
print('HYBRID_ALL', metrics.mean_absolute_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_ALL']))
print('HYBRID_NLP', metrics.mean_absolute_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_NLP']))
print('DUMMY_MEAN_MODEL', metrics.mean_absolute_error( y_pred_compare['Y_TRUE'], y_pred_compare['MEAN']))