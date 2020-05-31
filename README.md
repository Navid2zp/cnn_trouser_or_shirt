# Convolutional neural network test
Trained on digikala images for trousers and shirts.

Accuracy of 0.9844. Trained on 7k samples and validated on 3k samples.


_________________________________________________________________
Layer (type)          |       Output Shape        |      Param #   
----------------------|---------------------------|--------------
conv2d_1 (Conv2D)     |       (None, 62, 62, 64)  |      640       
activation_1 (relu Activation) |   (None, 62, 62, 64)   |     0         
max_pooling2d_1 (MaxPooling2 | (None, 31, 31, 64)   |     0         
conv2d_2 (Conv2D)       |     (None, 29, 29, 64)     |   36928     
activation_2 (relu Activation)  |  (None, 29, 29, 64)    |    0         
max_pooling2d_2 (MaxPooling2) | (None, 14, 14, 64)   |     0         
flatten_1 (Flatten)     |     (None, 12544)      |       0         
dense_1 (Dense)          |    (None, 32)         |       401440    
dropout_1 (Dropout)      |    (None, 32)           |     0         
dense_2 (Dense)        |      (None, 1)            |     33        
activation_3 (sigmoid Activation) |   (None, 1)         |        0        


**Total params:** 439,041

**Trainable params:** 439,041

**Non-trainable params:** 0

