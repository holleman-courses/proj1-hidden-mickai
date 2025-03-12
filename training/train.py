import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

def mickey_mobilenet():
    input_shape = (96, 96, 3)
    num_classes = 2
    num_filters = 8

    inputs = Input(shape=input_shape)

    # Layer 1: Standard Convolution
    x = Conv2D(num_filters, kernel_size=3, strides=2, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 2: Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', 
                        depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 3: Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', 
                        depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 4: Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', 
                        depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 5: Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', 
                        depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 6: Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', 
                        depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 7: Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', 
                        depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Global Average Pooling & Output Layer
    x = AveragePooling2D(pool_size=x.shape[1:3])(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and summarize the model
model = mickey_mobilenet()
model.summary()


# Define image properties
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 150

# Define paths to image folders
positive_dir = "positive_images"  # Hidden Mickey
negative_dir = "negative_images"  # No Mickey

# Load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Supported formats
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(label)
    return images, labels

# Load positive (Hidden Mickey) images
pos_images, pos_labels = load_images_from_folder(positive_dir, label=1)

# Load negative (No Mickey) images
neg_images, neg_labels = load_images_from_folder(negative_dir, label=0)

# Combine datasets
all_images = np.array(pos_images + neg_images)
all_labels = np.array(pos_labels + neg_labels)

# Split into training and validation sets (80% train, 20% validation)
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Print dataset sizes
print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")

# Data augmentation
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator()  # No augmentation for validation

# Create generators
train_generator = train_datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=BATCH_SIZE)

# Modify the model architecture to include Dropout
def mickey_mobilenet_with_dropout():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout layer with a rate of 0.5
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    return model

# Import the updated model function
model = mickey_mobilenet_with_dropout()

# Define a learning rate for the Adam optimizer
learning_rate = 0.00005
optimizer = Adam(learning_rate=learning_rate)

# Compile model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.xlabel("Epochs", fontweight='bold', fontsize=12)
plt.ylabel("Accuracy", fontweight='bold', fontsize=12)
plt.title('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel("Epochs", fontweight='bold', fontsize=12)
plt.ylabel("Loss", fontweight='bold', fontsize=12)
plt.title('Loss')

plt.show()

# Save trained model
model.save("final_hidden_mickey_model.h5")
print("Model saved successfully!")
