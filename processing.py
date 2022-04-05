import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

print(keras.__version__)

train_data_path = r"D:\AI Dataset\Brain_tumor-dataset\Brain_Tumor_New\Training"
validation_data_path = r"D:\AI Dataset\Brain_tumor-dataset\Brain_Tumor_New\Testing"


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

training_data = training_datagen.flow_from_directory(train_data_path,  # this is the target directory
                                                     target_size=(224, 224),  # all images will be resized to 150x150
                                                     batch_size=32,
                                                     class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

training_data.class_indices

# this is the augmentation configuration we will use for validation:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='binary')

images = [training_data[0][0][0] for i in range(5)]
plotImages(images)

model_path = 'model/Brain_Tumor_New.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print('Checkpoint Ok Roger')
# Building cnn model
cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[224, 224, 3]),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=3),\
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),  # neural network beulding
    keras.layers.Dense(units=128, activation='relu'),  # input layers
    keras.layers.Dropout(0.1),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=4, activation='softmax')  # output layer
])
print('Summary')
cnn_model.summary()
# compile cnn model
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Roger that')
print('training cnn model.-.-.-.-.')
history = cnn_model.fit(training_data,
                        epochs=50,
                        verbose=1,
                        validation_data=valid_data,
                        callbacks=callbacks_list)  # time start 16.06

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

history.history

print('Roger that')
