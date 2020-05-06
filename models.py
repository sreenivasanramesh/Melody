from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Activation, Attention, Bidirectional, Concatenate, Dense, LSTM
# from tensorflow.keras.layers import Dropout


class SingleLSTM(object):
    """
    LSTM model with one layer, and num_units of cells
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, batch_size=64, features=1):
        """Return the model"""
        model = Sequential()
        model.add(LSTM(num_units = self.num_units,
                       input_shape=(batch_size, features)))

        model.add(Dense(self.out_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam')
        return model


class BiLSTM(object):
    """
    Bidirectional LSTM model with one layer, and num_units of cells
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, batch_size=64, features=1):
        """Return the model"""
        model = Sequential()
        model.add(Bidirectional(LSTM(512),
                                input_shape=(batch_size, features)))
        model.add(Dense(self.out_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Nadam')
        return model


class AttentionLSTM(object):
    """
    Single LSTM followed by Attention
    Note: This class uses Functional API calls due to tf.keras.layers.Attention
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, batch_size=64, features=1):
        """Return the model"""
        input_shape = Input(shape=(batch_size, features), batch_size=batch_size)
        (lstm, forward_h, forward_c) = LSTM(512, return_sequences=True, return_state=True)(input_shape)
        context_vector, attention_weights = Attention(10)(lstm, forward_h)
        dense = Dense(self.out_classes)(context_vector)
        softmax = Activation('softmax')(dense)
        model = Model(inputs=input_shape, outputs=softmax)
        model.compile(loss='categorical_crossentropy', optimizer='Nadam')
        return model


class AttentionBiLSTM(object):
    """
    Bidirectional LSTM followed by Attention
    Note: This class uses Functional API calls due to tf.keras.layers.Attention
    """

    def __init__(self, num_units, out_classes):
        self.num_units = num_units
        self.out_classes = out_classes

    def get_network(self, batch_size=64, features=1):
        """Return the model"""
        input_shape = Input(shape=(batch_size, features), batch_size=batch_size)
        (lstm, forward_h, forward_c, backward_h, backward_c) = LSTM(512, return_sequences=True, return_state=True)(input_shape)
        state_h = Concatenate()([forward_h, backward_h])
        context_vector, attention_weights = Attention(10)(lstm, state_h)
        dense = Dense(self.out_classes)(context_vector)
        softmax = Activation('softmax')(dense)
        model = Model(inputs=input_shape, outputs=softmax)
        model.compile(loss='categorical_crossentropy', optimizer='Nadam')
        return model
