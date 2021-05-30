import sys
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

sys.path.append("..")
from musiclearn.single_note_processing import *
from musiclearn import config

# path to Schubert collection
schubert_dir = pathlib.Path(config.MUSICNET_MIDI_DIR) / "Schubert"

# path to location of tensorboard logs
#logs_dir = pathlib.Path(config.LOGS_DIR)
logs_dir = pathlib.Path("/csslab-localdata/csslab-si/jack_working_dir/test_logs")

# length of an input sequence
SEQUENCE_LENGTH = 100


def lstm_model(input_shape, n_vocab):
    """ Build a simple LSTM model """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(input_shape[1], input_shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add((LSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    
    return model


def load_lstm_model():
    """ Load the saved LSTM model """
    model = tf.keras.models.load_model('lstm_saved_weights.hdf5')
    return model


def bidirectional_lstm_model(input_shape, n_vocab):
    """ Build a bidirectional LSTM model """
    model = Sequential()
    model.add(Bidirectional(LSTM(512, return_sequences=True), input_shape=(input_shape[1], input_shape[2])))
    
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    
    return model


def load_bidirectional_lstm_model():
    """ Load the saved LSTM model """
    
    model = tf.keras.models.load_model('bidirect_lstm_saved_weights.hdf5')
    
    return model


class Customized_Attention(Layer):
    """ Attention block """
    def __init__(self, return_sequences=True):
        self.return_sequences=return_sequences
        super().__init__()
        
        
    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        super().build(input_shape)
        
        
    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'return_sequences': self.return_sequences})
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
def attention_lstm_model(input_shape, n_vocab):
    """ Construct a model based on LSTM and attention """
    model = Sequential()
    
    # Bidirectional LSTM layer with attention
    model.add(Bidirectional(LSTM(512, return_sequences=True), 
                            input_shape=(input_shape[1], input_shape[2])
                           )
             )
    model.add(Customized_Attention(return_sequences=True))
    model.add(Dropout(0.3))
    
    # Second bidirectional LSTM layer with attention
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Customized_Attention(return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    # compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    return model


def load_attention_lstm_model(input_shape, n_vocab):
    """ Load the saved weights """
    #model = tf.keras.models.load_model("attention_lstm_saved_weight.hdf5")
    
    model = Sequential()
    
    # Bidirectional LSTM layer with attention
    model.add(Bidirectional(LSTM(512, return_sequences=True), 
                            input_shape=(input_shape[1], input_shape[2])
                           )
             )
    model.add(Customized_Attention(return_sequences=True))
    model.add(Dropout(0.3))
    
    # Second bidirectional LSTM layer with attention
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Customized_Attention(return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    # compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    
    model.load_weights("attention_lstm_saved_weights.hdf5")
    return model


def simplified_wavenet(n_vocab, sequence_length, input_shape):
    """ A simplified version of WaveNet without residual and skip connection """
    model = Sequential()
    
    # causal convolution
    model.add(Conv1D(64, 3, padding='causal', activation='relu', 
                     input_shape=(input_shape[1], input_shape[2])
                    )
             )
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
    
    # dialated causal convolution
    model.add(Conv1D(128, 3,activation='relu',dilation_rate=2,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(256, 3,activation='relu',dilation_rate=4,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
    
    #model.add(Conv1D(256,5,activation='relu'))    
    model.add(GlobalMaxPool1D())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_vocab, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    
    return model


def load_wavenet_model():
    """ Load the saved weights of the WaveNet model """
    model = tf.keras.models.load_model('wavenet_saved_weights.hdf5')
    return model


def train_model(model, sequence_length, model_name, epochs=20, batch_size=128):
    """ Train a neural network to generate music """
    notes = read_midi(schubert_dir)
    
    # get training data
    network_input, network_output = prepare_sequences(sequence_length, notes)
    
    # train the network
    #filepath = model_name + "_saved_weights.hdf5"                       # uncomment later
    filepath = "test_" + model_name + "_saved_weights.hdf5"
    
    checkpoint_cb = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        node='min'
    )
    
    log_dir = str(logs_dir / "logs" / model_name)
    tensorboard_cb = TensorBoard(log_dir=log_dir,
                                     write_graph=True,
                                     write_images=True,
                                     update_freq='epoch'
                                )
    
    callback_list = [checkpoint_cb, tensorboard_cb]
    
    model.fit(network_input, 
              network_output, 
              epochs=epochs, 
              batch_size=batch_size, 
              callbacks=callback_list,
              validation_split=0.2
             )


def main():
    notes = read_midi(schubert_dir)
    
    n_vocab = get_num_unique_notes(notes)
    
    network_input, network_output = prepare_sequences(SEQUENCE_LENGTH, notes)
    
    model = lstm_model(network_input.shape, n_vocab)
    
    train_model(model, SEQUENCE_LENGTH, "lstm", 10, 64)
    

if __name__ == "__main__":
    main()







