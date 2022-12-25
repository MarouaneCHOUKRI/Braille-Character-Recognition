from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

model = load_model('braille_train1.h5')

numero_lettre = 0
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

img_src_save = "image_test/c.PNG"

img = Image.open(img_src_save)
img = img.convert('L')
img.save(img_src_save)

img = image.load_img(img_src_save)
img = image.img_to_array(img)
img = cv2.resize(img, (28,28))


x = np.expand_dims(img, axis=0)
a = model.predict(x)
array_value = []
for j in range(len(a[0])):
    array_value.append(a[0][j])


a=np.argmax(model.predict(x), axis=1)
numero_lettre = a[0]
lettre = str(alphabet[numero_lettre])
acc = str(array_value[numero_lettre]*100)
print(lettre+" : "+ acc[0:6])


model = load_model('mnist.h5')


for i in range(0, 10) :
    #img = cv2.imread(f"{i}.png")
    img = Image.open(f"{i}.png")
    img = img.convert('L')
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    print(np.argmax(res))