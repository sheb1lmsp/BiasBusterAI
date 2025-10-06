
import pickle
from model_creation import BiLSTMAttentionModel
from predict import predict
import numpy as np

# Load the tokenizer pickle object
with open('pickle_files/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the embedding matrix pickle object
with open('pickle_files/embedding_mat.pkl', 'rb') as file:
    embedding_matrix = pickle.load(file)

# Initialize the embedding_dim, vocab_size, max_len, idx_to_class and num_classes
embedding_dim = embedding_matrix.shape[1] # shape = (vocab_size-1, embedding_dim)
vocab_size = len(tokenizer.word_index)
max_len = 50 # maximum length of the words to input
idx_to_class = {0 : 'race', 1 : 'gender', 2 : 'profession', 3 : 'religion'}
num_classes = len(idx_to_class)

# Initialize the model and load the weights
model = BiLSTMAttentionModel(vocab_size, embedding_dim, embedding_matrix, max_len, num_classes)
model.build(input_shape=(max_len, embedding_dim))
model.load_weights('models/bilstm_model.h5')

def run_inference(text: str) -> np.ndarray:
    """
    Args:
        text (str): The text that is going to be processed to detect biases. 
    """
    # Call predict function to predict the lables and get the attention weights
    pred, attention = predict(text, model, tokenizer, max_len)
    
    return pred, attention
