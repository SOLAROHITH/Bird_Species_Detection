import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

# Set the path to your main folder containing class folders
main_folder = 'S:\My Files\VIT\VIT\SEM-5\Deep Learning\Project\Dataset\CUB_200_2011\images'
image_size = (224, 224)
batch_size = 32
num_classes = 200
epochs = 30

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of data will be used for validation
)

# Load the dataset and split into train and validation sets
train_generator = train_datagen.flow_from_directory(
    main_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # Specify training subset for train set
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    main_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Specify validation subset for validation set
    shuffle=False  # No need to shuffle validation data
)

# Load pre-trained VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Add a global average pooling layer and a dense layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine the base model and our classification layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_generator, epochs=epochs, verbose=1, validation_data=validation_generator)

# Plot the training accuracy and loss for analysis
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    main_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No need to shuffle test data
)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Testing Accuracy: {test_accuracy*100:.2f}%")

# Take a random image from a random class and predict its class
random_class_folder = random.choice(os.listdir(main_folder))
random_image_path = os.path.join(main_folder, random_class_folder, random.choice(os.listdir(os.path.join(main_folder, random_class_folder))))

# Load and preprocess the image
img = tf.keras.preprocessing.image.load_img(random_image_path, target_size=image_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print("Random Image Prediction:")
print(f"Predicted Class Index: {predicted_class}")
print(f"Predicted Class Name: {train_generator.class_indices}")

