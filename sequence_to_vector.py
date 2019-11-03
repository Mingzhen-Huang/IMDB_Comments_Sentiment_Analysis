# std lib imports
from typing import Dict
import sys

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models

class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._num_layers = num_layers
        self._dropout = dropout
        self._dense = list()
        for i in range(num_layers):
            self._dense.append(layers.Dense(units = input_dim,  activation=tf.nn.relu))
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:

        noise = 1e-10
        embedding_dim = tf.shape(vector_sequence)[2]
        max_tokens_num = tf.shape(vector_sequence)[1]
        batch_size = tf.shape(vector_sequence)[0]
        

        if training:
            bernoulli_mask = tf.random.uniform([batch_size, max_tokens_num],minval=0,maxval=1)
            bernoulli_mask = tf.cast(bernoulli_mask>self._dropout,dtype=tf.float32)
            mask = tf.multiply(sequence_mask, bernoulli_mask)
            total = tf.reduce_sum(mask, 1)
            total = tf.reshape(total,[batch_size, 1])
            mask = tf.reshape(mask, [batch_size, max_tokens_num, 1])
            x = tf.multiply(vector_sequence, mask)
            x = tf.reduce_sum(x, 1) / (total+noise)
        else:
            total = tf.reduce_sum(sequence_mask, 1)
            total = tf.reshape(total,[batch_size, 1])
            mask = tf.reshape(sequence_mask, [batch_size, max_tokens_num, 1])
            x = tf.multiply(vector_sequence, mask)
            x = tf.reduce_sum(x, 1) / (total+noise)

        for i in range(self._num_layers):
            x = self._dense[i](x, training=training)
            x_reshape = tf.reshape(x, [batch_size,1,self._input_dim])

            if i > 0:
                layer_representations = tf.concat([layer_representations,x_reshape], axis=1)
            else:
                layer_representations = x_reshape
        
        combined_vector = x
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)

        self._num_layers = num_layers
        print((input_dim,num_layers))
        self._gru = list()
        for i in range(num_layers):
            self._gru.append(layers.GRU(input_dim, activation='tanh', return_sequences = True, return_state = True))

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:

        embedding_dim = tf.shape(vector_sequence)[2]
        max_tokens_num = tf.shape(vector_sequence)[1]
        batch_size = tf.shape(vector_sequence)[0]

        x = vector_sequence
        for i in range(self._num_layers):
            x, layer = self._gru[i](x, training = training, mask = sequence_mask)

            layers = tf.reshape(layer, [batch_size,1,self._input_dim])
            if i > 0:
                layer_representations = tf.concat([layer_representations,layers], axis=1)
            else:
                layer_representations = layers
                
        combined_vector = layer_representations[:,-1,:]
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
