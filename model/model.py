import numpy as np
from tensorflow import keras
from model_input_format import get_inputs

import tensorflow as tf
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass 


def build_model():
      
      grid = np.random.randint(-1, 2, (6, 7))
      grid = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0,-1],
            [0, 0, 0, 0,-1,-1, 1],
            [1, 0, 0,-1, 1, 1, 1],
            [-1,0,-1, 1,-1,-1, 1]
      ])
      inputs = get_inputs(grid)

      layers = [[], []]
      
      input_layers = [keras.Input(state.shape) for state in inputs]
      
      layers[0] = keras.layers.concatenate([
            keras.layers.Flatten()(keras.layers.Conv2D(128, (3, 3), activation='relu')(input_layers[0])), 
            keras.layers.Flatten()(keras.layers.Conv2D(128, (5, 5), activation='relu')(input_layers[0]))
      ])
      layers[0] = keras.layers.Dense(8, 'tanh', name='convolutional_assessment')(layers[0])
      
      layers[1] = keras.layers.Dense(256, 'relu')(input_layers[1])
      layers[1] = keras.layers.Dense(8, 'tanh', name='critical_assessment')(layers[1])


      output_layer = keras.layers.concatenate([layers[0], layers[1]])
      output_layer = keras.layers.Dense(8, 'tanh')(output_layer)
      output_layer = keras.layers.Dense(1, 'tanh', name='winning_probability')(output_layer)

      model = keras.Model(
            inputs=input_layers,
            outputs=output_layer
      )
      
      model.compile(keras.optimizers.Adam(learning_rate=0.005), keras.losses.mean_squared_error)
      return model

if __name__ == "__main__":
      model = build_model()
      model.summary()
      model.save_weights('model.h5')