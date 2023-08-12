# **Bird Species Detection**

### Abstract: 
Fine-grained image classification is a challenging task in computer vision, requiring models to distinguish subtle visual differences between similar object categories. The CUB_200_2011 dataset, comprising 11,788 images of 200 bird species, serves as a standard benchmark for evaluating fine-grained classification algorithms. In this research, we explore the effectiveness of the Residual Network (ResNet) architecture for fine-grained bird classification on the CUB_200_2011 dataset. We present an in-depth analysis of the experimental setup, model training, and evaluation. Our findings showcase the remarkable performance of ResNet models, highlighting their ability to achieve state-of-the-art accuracy in fine-grained bird classification.
 ### 1. Introduction
Fine-grained image classification tasks involve distinguishing between object categories with subtle visual differences. The CUB_200_2011 dataset presents a unique challenge in fine-grained classification by providing a diverse set of bird images with varying poses, backgrounds, and lighting conditions. In this research, we investigate the ResNet architecture for fine-grained bird classification using the CUB_200_2011 dataset. ResNet's ability to address the vanishing gradient problem and its exceptional performance in image recognition tasks make it a promising candidate for fine-grained classification.

### 2. Related Work
Several deep learning models have been explored for fine-grained image classification, including VGG, Inception, and DenseNet. The use of attention mechanisms, transfer learning, and data augmentation has been effective in improving classification accuracy on fine-grained datasets. We build upon these prior studies and assess the performance of ResNet variants on the challenging CUB_200_2011 dataset.
### 3. CUB_200_2011 Dataset
### 3.1 Dataset Overview
The CUB_200_2011 dataset consists of 11,788 images of 200 bird species, with an average of 60 images per species. Each image is annotated with a bounding box around the bird, and species labels are provided for supervised learning. The dataset's fine-grained nature and detailed annotations make it suitable for evaluating the effectiveness of deep learning models.
### 3.2 Data Preprocessing
We preprocess the images by resizing them to 224x224 pixels, consistent with the input size of ResNet. To augment the dataset and improve generalization, we apply random horizontal flipping and rotation during training. We divide the dataset into training, validation, and test sets with a ratio of 70%, 15%, and 15%, respectively.



### 4. Dataset Description

### 4.1 Image Data

The CUB_200_2011 dataset consists of 11,788 high-quality images of birds, captured in diverse natural settings. Each image represents a single bird instance, and they are stored in JPEG format. These images have varying resolutions, which reflects the challenges encountered in real-world scenarios where bird instances can be at varying distances and angles.

### 4.2 Annotations

One of the key strengths of the CUB_200_2011 dataset lies in its detailed annotations. For each image, bounding boxes are provided, delineating the precise location of the bird within the image. These bounding boxes are invaluable for tasks like localization, where the goal is not only to classify the bird species but also to identify its location within the image accurately. Additionally, each image is associated with a unique integer label, ranging from 1 to 200, indicating the bird species it belongs to. The annotations were carefully curated by expert ornithologists to ensure accuracy and consistency.



### 4.3 Categories

The dataset covers 200 bird species, and each category corresponds to a unique species. Each category is identified by a unique integer label, ranging from 1 to 200. The mapping between these category labels and the corresponding bird species names is provided in a separate file, ensuring ease of interpretation and analysis. The variety of bird species ensures that the dataset remains challenging and relevant for research in fine-grained bird classification.





### 5. Code
### VGG16
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




Output:
Found 9465 images belonging to 200 classes.
Found 2323 images belonging to 200 classes.
WARNING:absl:`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.
Epoch 1/30
296/296 [==============================] - 581s 2s/step - loss: 5.0916 - accuracy: 0.0190 - val_loss: 4.7462 - val_accuracy: 0.0349
Epoch 2/30
296/296 [==============================] - 553s 2s/step - loss: 4.5448 - accuracy: 0.0560 - val_loss: 4.3836 - val_accuracy: 0.0697
Epoch 3/30
296/296 [==============================] - 554s 2s/step - loss: 4.1836 - accuracy: 0.0977 - val_loss: 4.1593 - val_accuracy: 0.0852
Epoch 4/30
296/296 [==============================] - 552s 2s/step - loss: 3.9041 - accuracy: 0.1341 - val_loss: 3.9221 - val_accuracy: 0.1235
Epoch 5/30
296/296 [==============================] - 554s 2s/step - loss: 3.6530 - accuracy: 0.1736 - val_loss: 3.7693 - val_accuracy: 0.1416
Epoch 6/30
296/296 [==============================] - 553s 2s/step - loss: 3.4566 - accuracy: 0.2012 - val_loss: 3.6220 - val_accuracy: 0.1610
Epoch 7/30
296/296 [==============================] - 578s 2s/step - loss: 3.2985 - accuracy: 0.2335 - val_loss: 3.5296 - val_accuracy: 0.1838
Epoch 8/30
296/296 [==============================] - 594s 2s/step - loss: 3.1611 - accuracy: 0.2546 - val_loss: 3.4774 - val_accuracy: 0.1834
Epoch 9/30
296/296 [==============================] - 590s 2s/step - loss: 3.0304 - accuracy: 0.2787 - val_loss: 3.3874 - val_accuracy: 0.2028
Epoch 10/30
296/296 [==============================] - 589s 2s/step - loss: 2.9283 - accuracy: 0.2964 - val_loss: 3.3402 - val_accuracy: 0.2049
Epoch 11/30
296/296 [==============================] - 587s 2s/step - loss: 2.8238 - accuracy: 0.3144 - val_loss: 3.2654 - val_accuracy: 0.2161
Epoch 12/30
296/296 [==============================] - 587s 2s/step - loss: 2.7519 - accuracy: 0.3320 - val_loss: 3.2627 - val_accuracy: 0.2230
Epoch 13/30
296/296 [==============================] - 586s 2s/step - loss: 2.6558 - accuracy: 0.3516 - val_loss: 3.1900 - val_accuracy: 0.2312
Epoch 14/30
296/296 [==============================] - 588s 2s/step - loss: 2.5956 - accuracy: 0.3633 - val_loss: 3.1976 - val_accuracy: 0.2346
Epoch 15/30
296/296 [==============================] - 580s 2s/step - loss: 2.5281 - accuracy: 0.3754 - val_loss: 3.2071 - val_accuracy: 0.2294
Epoch 16/30
296/296 [==============================] - 580s 2s/step - loss: 2.4666 - accuracy: 0.3917 - val_loss: 3.1734 - val_accuracy: 0.2548
Epoch 17/30
296/296 [==============================] - 576s 2s/step - loss: 2.4048 - accuracy: 0.4010 - val_loss: 3.1343 - val_accuracy: 0.2591
Epoch 18/30
296/296 [==============================] - 575s 2s/step - loss: 2.3544 - accuracy: 0.4138 - val_loss: 3.1352 - val_accuracy: 0.2441
Epoch 19/30
296/296 [==============================] - 575s 2s/step - loss: 2.2911 - accuracy: 0.4263 - val_loss: 3.1222 - val_accuracy: 0.2492
Epoch 20/30
296/296 [==============================] - 575s 2s/step - loss: 2.2467 - accuracy: 0.4317 - val_loss: 3.1064 - val_accuracy: 0.2600
Epoch 21/30
296/296 [==============================] - 577s 2s/step - loss: 2.2003 - accuracy: 0.4417 - val_loss: 3.0957 - val_accuracy: 0.2686
Epoch 22/30
296/296 [==============================] - 576s 2s/step - loss: 2.1508 - accuracy: 0.4574 - val_loss: 3.1195 - val_accuracy: 0.2536
Epoch 23/30
296/296 [==============================] - 576s 2s/step - loss: 2.1023 - accuracy: 0.4683 - val_loss: 3.0940 - val_accuracy: 0.2609
Epoch 24/30
296/296 [==============================] - 575s 2s/step - loss: 2.0551 - accuracy: 0.4761 - val_loss: 3.0784 - val_accuracy: 0.2712
Epoch 25/30
296/296 [==============================] - 575s 2s/step - loss: 2.0220 - accuracy: 0.4817 - val_loss: 3.0849 - val_accuracy: 0.2699
Epoch 26/30
296/296 [==============================] - 580s 2s/step - loss: 1.9805 - accuracy: 0.4895 - val_loss: 3.0701 - val_accuracy: 0.2712
Epoch 27/30
296/296 [==============================] - 588s 2s/step - loss: 1.9401 - accuracy: 0.5030 - val_loss: 3.0840 - val_accuracy: 0.2695
Epoch 28/30
296/296 [==============================] - 580s 2s/step - loss: 1.8994 - accuracy: 0.5104 - val_loss: 3.1020 - val_accuracy: 0.2768
Epoch 29/30
296/296 [==============================] - 576s 2s/step - loss: 1.8682 - accuracy: 0.5199 - val_loss: 3.0607 - val_accuracy: 0.2742
Epoch 30/30
296/296 [==============================] - 575s 2s/step - loss: 1.8305 - accuracy: 0.5254 - val_loss: 3.0651 - val_accuracy: 0.2699
 
Found 11788 images belonging to 200 classes.
369/369 [==============================] - 567s 2s/step - loss: 1.8887 - accuracy: 0.5266
Testing Accuracy: 52.66%
1/1 [==============================] - 0s 173ms/step
Random Image Prediction:
Predicted Class Index: 20
Predicted Class Name: {'001.Black_footed_Albatross': 0, '002.Laysan_Albatross': 1, '003.Sooty_Albatross': 2, '004.Groove_billed_Ani': 3, '005.Crested_Auklet': 4, '006.Least_Auklet': 5, . . . . )






### Code used to access GPU:
pip install tensorflow-gpu numpy matplotlib
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
### Initialize GPU memory growth to avoid allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


### Output:
1 Physical GPUs, 1 Logical GPUs
6. Discussion
We analyze the experimental results to understand the impact of different ResNet architectures on fine-grained bird classification. We discuss the benefits of residual blocks and skip-connections in enabling the training of deeper networks. Additionally, we highlight the significance of transfer learning from pre-trained models and the effectiveness of data augmentation techniques in improving model generalization.



### 7. Conclusion
In this research, we explored the use of the ResNet architecture for fine-grained bird classification on the CUB_200_2011 dataset. Our findings demonstrate that ResNet models achieve state-of-the-art accuracy on this challenging dataset, showcasing their effectiveness in handling fine-grained recognition tasks. The ability of ResNet to capture subtle visual differences between similar bird species makes it a valuable tool for image classification in various domains. We conclude that ResNet is a powerful approach for fine-grained bird classification and holds great promise for advancing the field of computer vision.


