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

    def get_network(self, sequence_length=100, features=1, test=False):
        """Return the model"""
        model = Sequential()
        model.add(LSTM(units=self.num_units,
                       input_shape=(sequence_length, features)))

        model.add(Dense(self.out_classes))
        if not test:
            model.add(Activation('softmax'))
        if test:
            model.add(Activation('sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model


class BiLSTM(object):
    """
    Bidirectional LSTM model with one layer, and num_units of cells
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1, test=False):
        """Return the model"""
        model = Sequential()
        model.add(Bidirectional(LSTM(self.num_units),
                                input_shape=(sequence_length, features)))
        model.add(Dense(self.out_classes))
        if not test:
            model.add(Activation('softmax'))
        if test:
            model.add(Activation('sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        return model


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
          
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class AttentionLSTM(object):
    """
    Single LSTM followed by Attention
    Note: This class uses Functional API calls due to tf.keras.layers.Attention
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, sequence_length=100, features=1, test=False):
        """Return the model"""
        input_shape = Input(shape=(sequence_length, features), batch_size=64)
        (lstm, forward_h, forward_c) = LSTM(self.num_units, return_sequences=True, return_state=True)(input_shape)
        context_vector, attention_weights = Attention(10)(lstm, forward_h)
        dense = Dense(self.out_classes)(context_vector)
        if not test:
            output = Activation('softmax')(dense)
        if test:
            output = Activation('sigmoid')(dense)
        model = Model(inputs=input_shape, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[keras_perplexity, 'accuracy'])
        print(model)
        return model



'''
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
'''