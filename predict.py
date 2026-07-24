import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224,224)

model = tf.keras.models.load_model("object_model.keras")

with open("labels.txt") as f:
    labels = [line.strip() for line in f]

image = Image.open("test/mango.jpg").convert("RGB")
image = image.resize(IMG_SIZE)

img = np.array(image)

img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

index = np.argmax(prediction)

print("Prediction:", labels[index])

print("Confidence:", prediction[0][index] * 100)