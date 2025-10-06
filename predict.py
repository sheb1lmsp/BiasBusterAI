
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple

def predict(
    text: str,
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    max_len: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Predicts bias class probabilities and attempts to extract attention weights for a text input in BiasBusterAI.

    Args:
        text (str): The input text to analyze for bias.
        model (tf.keras.Model): The trained BiLSTM model with an attention layer.
        tokenizer (Tokenizer): The fitted Keras Tokenizer for text processing.
        max_len (int): Maximum sequence length for padding (default: 50).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Predicted class probabilities of shape (num_classes,).
            - Attention weights of shape (max_len,) for heatmap visualization.
    """
    # Convert text to sequence and pad to max_len
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")

    # Run inference to get predicted probabilities
    predictions = model.predict(padded_sequence, verbose=0)

    # Initialize attention with input sequence
    attention = padded_sequence
    # Iterate through model layers to compute intermediate outputs
    for layer in model.layers:
        # Check for the attention layer (assumes model.attention exists)
        if layer == model.attention:
            _, attention = layer(attention)  # Apply attention layer to get context and weights
            break
        attention = layer(attention)  # Apply current layer to intermediate input

    # Reshape outputs
    predictions = predictions[0]  # Shape: (num_classes,)

    # Return predicted class index and reshaped attention weights
    return predictions, tf.reshape(attention, shape=(-1, 1))  # weights reshaped to (max_len, 1)
