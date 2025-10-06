
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple
import numpy as np

class AttentionLayer(layers.Layer):
    """
    Description:
        A custom Keras layer that implements an attention mechanism for sequence data in BiasBusterAI.
    """
    def __init__(self):
        super(AttentionLayer, self).__init__()
        # Create the Dense layer for attention scores
        self.dense = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Description:
            Computes attention weights and a context vector for the input sequence.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - Context vector of shape (batch, hidden_size).
                - Attention weights of shape (batch, seq_len, 1).
        """
        # Compute attention scores
        scores = self.dense(inputs)            # (batch, seq_len, 1)
        # Normalize scores to obtain attention weights
        weights = tf.nn.softmax(scores, axis=1) # (batch, seq_len, 1)
        # Compute context vector as weighted sum of inputs
        context_vector = tf.reduce_sum(weights * inputs, axis=1)  # (batch, hidden_size)
        return context_vector, weights

class BiLSTMAttentionModel(tf.keras.Model):
    """
    Description:
        A Keras model for bias classification using a bidirectional LSTM and attention mechanism in BiasBusterAI.

    Args:
        vocab_size (int): Size of the vocabulary for the embedding layer.
        embedding_dim (int): Dimension of the embeddings.
        embedding_matrix: Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
        max_len (int): Maximum sequence length for input text.
        num_classes (int): Number of bias classes for classification.
        lstm_units (int): Number of units in the LSTM layer (default: 128).

    Returns:
        None
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_matrix: np.ndarray,
        max_len: int,
        num_classes: int,
        lstm_units: int = 128
    ):
        super(BiLSTMAttentionModel, self).__init__()
        # Non-trainable embedding layer with pre-trained weights
        self.embedding = layers.Embedding(
            input_dim=vocab_size+1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False
        )
        # Bidirectional LSTM to capture sequential dependencies
        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True)
        )
        # Custom attention layer
        self.attention = AttentionLayer()
        # Output layer for classification
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Description:
            Defines the forward pass of the model for bias classification.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, seq_len).
            training (bool): Whether the model is in training mode (default: False).

        Returns:
            tf.Tensor: Output probabilities of shape (batch, num_classes).
        """
        # Apply embedding layer
        x = self.embedding(inputs)               # (batch, seq_len, embed_dim)
        # Apply bidirectional LSTM
        x = self.bilstm(x)                      # (batch, seq_len, 2*lstm_units)
        # Apply attention to get context vector
        context, _ = self.attention(x)          # (batch, 2*lstm_units), (batch, seq_len, 1)
        # Compute output probabilities
        output = self.fc(context)               # (batch, num_classes)
        return output
