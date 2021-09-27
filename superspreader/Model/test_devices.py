import tensorflow as tf
from time import time

class TestNet(tf.keras.Model):

    def __init__ (self):

        super(TestNet, self).__init__()
        self.FC1 = tf.keras.layers.Dense(100,  activation = 'relu')
        self.FC2 = tf.keras.layers.Dense(1000, activation = 'relu')
        self.FC3 = tf.keras.layers.Dense(100,  activation = 'relu')
    
    def call(self, x):

        x = self.FC1(x) 
        x = self.FC2(x)
        return self.FC3(x)


if __name__ == '__main__':

    epochs = 15

    table_generator = lambda device_index, device_name, device_type: (  
                '{message: <{width}}'.format(message = device_index, width = 20) + ' | '
              + '{message: <{width}}'.format(message = device_name,  width = 40) + ' | '
              + '{message: <{width}}'.format(message = device_type,  width = 20))

    # Find all availabe devices.
    devices = tf.config.list_physical_devices()
    device_test = dict()
    for device in devices:

        # Store the current device information and 

        # Setup the model for testing the instance.
        test_model = TestNet()
        optimizer  = tf.keras.optimizers.Adam(learning_rate = 0.01)

        # Create a simple test batch.
        x = tf.random.normal((50,100), dtype = tf.float32)
        y = tf.random.normal((50,100), dtype = tf.float32)
        t = list()
        for epoch in range(epochs):

            start = time()
            with tf.device(device.name[-5:]):
                with tf.GradientTape() as tape:

                    y_hat = test_model(x)
                    loss  = tf.math.reduce_mean(tf.square(y_hat - y))
                
                gradients = tape.gradient(loss, test_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, test_model.trainable_variables))
            
            end = time()
            t.append(end - start)
        
        print('-'*100)
        print(' '*100)
        print(table_generator(sum(t)/len(t), device.name, device.device_type))