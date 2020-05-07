from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Activation, Attention, Bidirectional, Concatenate, Dense, LSTM, Conv1D
# from tensorflow.keras.layers import Dropout
import tensorflow as tf
import tensorflow.keras.backend as K




def keras_perplexity(y_true, y_pred):
    cross_entropy = K.mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    perplexity = K.exp(cross_entropy)
    return perplexity




class SingleLSTM(object):
    """
    LSTM model with one layer, and num_units of cells
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1):
        """Return the model"""
        model = Sequential()
        model.add(LSTM(units=self.num_units,
                       input_shape=(sequence_length, features)))

        model.add(Dense(self.out_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model


class BiLSTM(object):
    """
    Bidirectional LSTM model with one layer, and num_units of cells
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1):
        """Return the model"""
        model = Sequential()
        model.add(Bidirectional(LSTM(self.num_units),
                                input_shape=(sequence_length, features)))
        model.add(Dense(self.out_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model


class AttentionLSTM(object):
    """
    Single LSTM followed by Attention
    Note: This class uses Functional API calls due to tf.keras.layers.Attention
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1):
        """Return the model"""
        inputs = Input(shape=(sequence_length, features), batch_size=sequence_length)
        (lstm, forward_h, forward_c) = LSTM(self.num_units, return_sequences=True, return_state=True)(inputs)
        context_vector, attention_weights = Attention(10)(lstm, forward_h)
        dense = Dense(self.out_classes)(context_vector)
        softmax = Activation('softmax')(dense)
        model = Model(inputs=input_shape, outputs=softmax)
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model


class AttentionBiLSTM(object):
    """
    Bidirectional LSTM followed by Attention
    Note: This class uses Functional API calls due to tf.keras.layers.Attention
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1):
        """Return the model"""
        input_shape = Input(shape=(sequence_length, features), batch_size=sequence_length)
        (lstm, forward_h, forward_c, backward_h, backward_c) = LSTM(self.num_units, return_sequences=True, return_state=True)(input_shape)
        state_h = Concatenate()([forward_h, backward_h])
        context_vector, attention_weights = Attention(10)(lstm, state_h)
        dense = Dense(self.out_classes)(context_vector)
        softmax = Activation('softmax')(dense)
        model = Model(inputs=input_shape, outputs=softmax)
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model


class ConvLSTM(object):
    """
    1D Conv followed by LSTM model with one layer, and num_units of cells
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1):
        """Return the model"""
        model = Sequential()
        model.add(Conv1D(filters=self.num_units, kernel_size=4, strides=1, input_shape=(sequence_length, features), activation='relu', padding='same'))
        model.add(LSTM(self.num_units)) #,
        #                        input_shape=(sequence_length, features)))
        model.add(Dense(self.out_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model