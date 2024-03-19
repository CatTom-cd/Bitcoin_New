import shutil, os
from keras import layers
from keras import models
 
original_dataset_cat = 'dataset/cat'
original_dataset_dog= 'dataset/dog1'
 
base_dir = 'base'
os.mkdir(base_dir)
 
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
 
train_dir_cats = os.path.join(train_dir, 'cats')
os.mkdir(train_dir_cats)
train_dir_dogs = os.path.join(train_dir, 'dogs')
os.mkdir(train_dir_dogs)
 
validation_dir_cats = os.path.join(validation_dir, 'cats')
os.mkdir(validation_dir_cats)
validation_dir_dogs = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dir_dogs)
 
test_dir_cats = os.path.join(test_dir, 'cats')
os.mkdir(test_dir_cats)
test_dir_dogs = os.path.join(test_dir, 'dogs')
os.mkdir(test_dir_dogs)
 
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
  src = os.path.join(original_dataset_cat, fname)
  dst = os.path.join(train_dir_cats, fname)
  shutil.copyfile(src, dst)
 
fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
  src = os.path.join(original_dataset_cat, fname)
  dst = os.path.join(validation_dir_cats, fname)
  shutil.copyfile(src, dst)
 
 
fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
  src = os.path.join(original_dataset_cat, fname)
  dst = os.path.join(test_dir_cats, fname)
  shutil.copyfile(src, dst)
 
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
  src = os.path.join(original_dataset_dog, fname)
  dst = os.path.join(train_dir_dogs, fname)
  shutil.copyfile(src, dst)
 
fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
  src = os.path.join(original_dataset_dog, fname)
  dst = os.path.join(validation_dir_dogs, fname)
  shutil.copyfile(src, dst)
 
 
fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
  src = os.path.join(original_dataset_dog, fname)
  dst = os.path.join(test_dir_dogs, fname)
  shutil.copyfile(src, dst)

from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale=1./255, rotation = 40, width = 0.2, height = 0.2, shear = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'neaest')
test_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32,class_mode='binary')
 
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
 
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
 
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
 
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
 
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
 
history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=5, validation_data=validation_generator, validation_steps=10)
 
for data_batch, labels_batch in train_generator:
  print('data_batch_shape:', data_batch.shape)
  print('labels_batch_shape:', labels_batch.shape)
  break
 
model.save('cats_vs_dogs.h5')
 
import matplotlib.pyplot as plt
 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.figure()
 
 
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.show()
 
import keras
model = keras.models.load_model('cats_vs_dogs.h5')
pred = model.predict()
 
from keras.preprocessing import image
import numpy as np
 
test_image = image.load_img("3.jpg",target_size=(150,150))
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
 
i = 0
if(result >= 0.5):
    print("Dog")
else:
    print("Cat")