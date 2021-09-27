import tensorflow as tf
import numpy as np

from os import chdir
from tqdm import tqdm

class trainer:
    """Training class for facillitating training BayesFlow model with MLL."""

    def __init__ (self, model:callable, optimizer:callable) -> None:

        self.model = model
        self.optimizer = optimizer

        if epochs is not None:
            self.training_errors = np.zeros(epochs)
            self.testing_errors  = np.zeros(epochs)
            self.learning_rates  = np.zeros(epochs)
        
        else:
            self.training_errors = np.zeros(1)
            self.testing_errors  = np.zeros(1)
            self.learning_rates  = np.zeros(1)
        
    
    def __call__(self, x, y, learning_rate:float or None = None, instance:str or None = None) -> tf.Tensor:

        # Update the learning rate if any was provided.
        if learning_rate is not None:
            self.optimizer.learning_rate.assign(learning_rate)
        
        # Use a specific instance if it was referenced.
        if instance is not None:
            with tf.device(instance):
                training_error = self._train(x, y, learning_rate = learning_rate) 
        
        else:
            training_error = self._train(x, y, learning_rate = learning_rate)


    def _train(self,x , y) -> tf.Tensor:
        """Performs a single training step.
        
        Arguments:
        ----------
        x : (tf.Tensor float32) -- Features tensor of shape (batch size, number of features).
        y : (tf.Tensor float32) -- Target tensor of shape (batch size, number of targets).

        Returns:
        --------
        training_error : (tf.Tensor float32) -- Training Error of shape (batch size, 1).
        """

        with tf.GradientTape() as tape:
            z_hat, log_det_J = self.model(x, y)
            training_error   = tf.reduce_mean(0.5 * tf.square(tf.norm(z_hat, axis=-1)) - log_det_J)

        gradients = tape.gradient(training_error, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return training_error
    
    def optimizer(self):
        """Returns the optimizer in its current state."""
        return self.optimizer
    
    def model(self):
        """Returns the model in its current state."""
        return self.model


if __name__ == '__main__':
    epochs = 1000

    # Obtain all available computing devices on the current machine.
    devices = tf.config.list_physical_devices()

    table_generator = lambda device_index, device_name, device_type: (  
                '{message: <{width}}'.format(message = device_index, width = 20) + ' | '
              + '{message: <{width}}'.format(message = device_name,  width = 40) + ' | '
              + '{message: <{width}}'.format(message = device_type,  width = 20))
    
    def passage(only_spacing = True):
        """Print a passage."""
        if not only_spacing:
            print('#'*100)
        print(' '*100)

    
    # Generate a table of possible devices to choose from.
    passage()
    print(table_generator('Device Index', 'Device Name', 'Device Type'))
    device_index = 0
    for device in devices:
        print('-'*100)
        print(table_generator(device_index, device.name, device.device_type))
        device_index += 1

    passage()
    chosen_device = input('Pleas select an device by "Device Index": ')
    
    passage()
    print('Start Training: ...')
    passage(False)

    # Start the training.
    for epoch in tqdm(range(epochs)):
        pass
