import numpy as np
import cv2
from CNN6_MNIST import CNN6_MNIST

cnn6 = CNN6_MNIST()
cnn6.build(train=True, training_steps=650)

# Test on trainingset

#image, classification = cnn6.predict_random_from_MNIST()
#image = np.reshape(image, (28,28))

# Test another hand drawn number
        
image = cv2.imread('test.jpg', 0)
image = 1-(np.ravel(image)/255.0)
classification = cnn6.predict(image)
image = np.reshape(image, (28,28))

cv2.namedWindow('classification: '+str(classification), cv2.WINDOW_NORMAL)
cv2.imshow('classification: '+str(classification), 1-image)
cv2.waitKey(0)
cv2.destroyAllWindows()