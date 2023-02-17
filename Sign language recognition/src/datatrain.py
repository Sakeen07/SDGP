import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Define data directories
train_dir = 'Data/BSL'
val_dir = 'Data/BSL'

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Define ImageDataGenerator and rescale the pixel values to be in the range [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data and apply data augmentation
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Load validation data
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Define the number of classes in your dataset
num_classes = train_data.num_classes

# Load pre-trained VGG16 model without the top (fully connected) layers
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add your own classification layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Define the new model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with a low learning rate
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on your dataset
model.fit(train_data, epochs=10, validation_data=val_data)

# Save the model to a .h5 file
model.save('keras.h5')