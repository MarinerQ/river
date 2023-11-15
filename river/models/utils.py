import numpy as np 
import pickle 

def save_model(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {filename}')

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from {filename}')
    return model 