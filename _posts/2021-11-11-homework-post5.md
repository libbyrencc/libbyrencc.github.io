---
layout: post
title: Post 5
---

# Transfer learning and fine-tuning

First we need to import several important packages.


```python
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras
```


```python
from tensorflow.keras import utils 
```

Now download the dataset, which contains images of cats and dogs.  
Then create a `tf.data.Dataset` for training and a dataset for validation.  
Note here the batch size is 32 and img is 160x160x3.


```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
print(PATH)
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```

    C:\Users\libby\.keras\datasets\cats_and_dogs_filtered
    Found 2000 files belonging to 2 classes.
    Found 1000 files belonging to 2 classes.
    

And get our dataset class name here.


```python
class_names = train_dataset.class_names
```

### Improve the performance by setting `buffer_size =AUTOTUNE`


```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

Show some cats and dogs of our dataset.


```python
def show_simples(dataset)
    cats=[] 
    dogs=[]
    #new empty list to save images
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(32):
            #choose cats
            if labels[i]==0:
                cats.append(images[i].numpy().astype("uint8"))
            #choose dogs
            if labels[i]==0:
                dogs.append(images[i].numpy().astype("uint8"))
    for i in range(6):
        if i<3:
            #show 3 cats' images
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(cats[i])
            # Add title
            plt.title("cats")
            plt.axis("off")
        else:
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(dogs[i])
            plt.title("dogs")
            plt.axis("off")
show_simples(train_dataset)
```


    
![png](output_11_0.png)
    


## Check label frequencies


```python
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()
#Create an iterator 
```


```python
dogs_freq=0
cats_freq=0
for label in labels_iterator:
    if label ==0:
        cats_freq=cats_freq+1
        #Add 1 to cats counter
    else:
        dogs_freq=dogs_freq+1
        #Add 1 to dogs counter
print("dogs_freq = ",dogs_freq)
print("cats_freq = ",cats_freq)
```

    dogs_freq =  1000
    cats_freq =  1000
    

So the base line will be `50%`,since we have 1000 dogs images and 1000 cats images.

## Model 1
Simply use `tf.keras.Sequential` model with some layers to train.


```python
from tensorflow.keras import datasets, layers, models
```


```python
model1 = models.Sequential([
    layers.Conv2D(8, (9, 9), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((4, 4)),
    layers.Conv2D(128, (4, 4), activation='relu'),
    # 4 convolution blocks with a max pooling layer
    layers.Dropout(0.2),
    #drop 20% parameters randomly, left 80%
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    #fully-connected layer
    layers.Dense(2)
])
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#train the dataset
history = model1.fit(train_dataset, 
                     epochs=20,
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 20s 81ms/step - loss: 3.5748 - accuracy: 0.5255 - val_loss: 0.6932 - val_accuracy: 0.5235
    Epoch 2/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6838 - accuracy: 0.5445 - val_loss: 0.6898 - val_accuracy: 0.5470
    Epoch 3/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.6693 - accuracy: 0.5675 - val_loss: 0.7060 - val_accuracy: 0.5322
    Epoch 4/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.6503 - accuracy: 0.5755 - val_loss: 0.7102 - val_accuracy: 0.5210
    Epoch 5/20
    63/63 [==============================] - 3s 49ms/step - loss: 0.6214 - accuracy: 0.6195 - val_loss: 0.7434 - val_accuracy: 0.5260
    Epoch 6/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.5849 - accuracy: 0.6500 - val_loss: 0.7714 - val_accuracy: 0.5285
    Epoch 7/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.5561 - accuracy: 0.6625 - val_loss: 0.8597 - val_accuracy: 0.5384
    Epoch 8/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.5151 - accuracy: 0.6980 - val_loss: 0.8313 - val_accuracy: 0.5458
    Epoch 9/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.4860 - accuracy: 0.7220 - val_loss: 0.9209 - val_accuracy: 0.5186
    Epoch 10/20
    63/63 [==============================] - 4s 54ms/step - loss: 0.4384 - accuracy: 0.7470 - val_loss: 1.1168 - val_accuracy: 0.5272
    Epoch 11/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.4206 - accuracy: 0.7695 - val_loss: 1.0486 - val_accuracy: 0.5446
    Epoch 12/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.4296 - accuracy: 0.7620 - val_loss: 0.9362 - val_accuracy: 0.5235
    Epoch 13/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.3926 - accuracy: 0.7825 - val_loss: 1.0135 - val_accuracy: 0.5569
    Epoch 14/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.3602 - accuracy: 0.7920 - val_loss: 1.1131 - val_accuracy: 0.5408
    Epoch 15/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.3404 - accuracy: 0.8155 - val_loss: 1.1443 - val_accuracy: 0.5644
    Epoch 16/20
    63/63 [==============================] - 4s 54ms/step - loss: 0.3215 - accuracy: 0.8130 - val_loss: 1.2469 - val_accuracy: 0.5458
    Epoch 17/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.3195 - accuracy: 0.8275 - val_loss: 1.1903 - val_accuracy: 0.5483
    Epoch 18/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.2850 - accuracy: 0.8385 - val_loss: 1.2867 - val_accuracy: 0.5780
    Epoch 19/20
    63/63 [==============================] - 4s 53ms/step - loss: 0.3153 - accuracy: 0.8260 - val_loss: 1.2660 - val_accuracy: 0.5532
    Epoch 20/20
    63/63 [==============================] - 4s 53ms/step - loss: 0.2962 - accuracy: 0.8360 - val_loss: 1.3756 - val_accuracy: 0.5483
    

Try to improve or layers to see if we can achieve better `val_accuracy`


```python
model1 = models.Sequential([
    layers.Conv2D(8, (9, 9), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((4, 4)),
    layers.Conv2D(128, (4, 4), activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10) 
])
```


```python
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
history = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 4s 52ms/step - loss: 4.7760 - accuracy: 0.5000 - val_loss: 0.7078 - val_accuracy: 0.5037
    Epoch 2/20
    63/63 [==============================] - 3s 48ms/step - loss: 0.6965 - accuracy: 0.5235 - val_loss: 0.6777 - val_accuracy: 0.5359
    Epoch 3/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.6827 - accuracy: 0.5650 - val_loss: 0.6702 - val_accuracy: 0.5804
    Epoch 4/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.6480 - accuracy: 0.6250 - val_loss: 0.6373 - val_accuracy: 0.6176
    Epoch 5/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.5942 - accuracy: 0.6800 - val_loss: 0.6240 - val_accuracy: 0.6584
    Epoch 6/20
    63/63 [==============================] - 3s 49ms/step - loss: 0.5707 - accuracy: 0.6930 - val_loss: 0.6665 - val_accuracy: 0.6485
    Epoch 7/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.5399 - accuracy: 0.7330 - val_loss: 0.5893 - val_accuracy: 0.7178
    Epoch 8/20
    63/63 [==============================] - 3s 49ms/step - loss: 0.4958 - accuracy: 0.7660 - val_loss: 0.6227 - val_accuracy: 0.7067
    Epoch 9/20
    63/63 [==============================] - 4s 52ms/step - loss: 0.5237 - accuracy: 0.7410 - val_loss: 0.6850 - val_accuracy: 0.6275
    Epoch 10/20
    63/63 [==============================] - 3s 49ms/step - loss: 0.4454 - accuracy: 0.7970 - val_loss: 0.5919 - val_accuracy: 0.7215
    Epoch 11/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.4410 - accuracy: 0.7950 - val_loss: 0.5861 - val_accuracy: 0.7203
    Epoch 12/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.3770 - accuracy: 0.8305 - val_loss: 0.6028 - val_accuracy: 0.7129
    Epoch 13/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.3689 - accuracy: 0.8305 - val_loss: 0.6811 - val_accuracy: 0.7079
    Epoch 14/20
    63/63 [==============================] - 4s 54ms/step - loss: 0.3036 - accuracy: 0.8700 - val_loss: 0.6571 - val_accuracy: 0.7228
    Epoch 15/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.2525 - accuracy: 0.8890 - val_loss: 0.6710 - val_accuracy: 0.7376
    Epoch 16/20
    63/63 [==============================] - 3s 50ms/step - loss: 0.2140 - accuracy: 0.9110 - val_loss: 0.9246 - val_accuracy: 0.6733
    Epoch 17/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.1996 - accuracy: 0.9205 - val_loss: 0.9885 - val_accuracy: 0.6918
    Epoch 18/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.1659 - accuracy: 0.9330 - val_loss: 0.8779 - val_accuracy: 0.7178
    Epoch 19/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.2416 - accuracy: 0.8945 - val_loss: 1.0734 - val_accuracy: 0.6844
    Epoch 20/20
    63/63 [==============================] - 4s 55ms/step - loss: 0.2339 - accuracy: 0.9020 - val_loss: 0.9700 - val_accuracy: 0.7104
    

Plot the training data.


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.50, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x213a0b6ed30>




    
![png](output_24_1.png)
    


The accuracy of my model stabilized `between 65% and 70%` during training.  
My model did `15%` better than the baseline.
Yes! There's overfitting in my model, the training accuracy is much higher than the validation accuracy.

## Model 2 with Data Augmentation

Data augmentation is used to like slightly change the orginal data, like rotate or filp, which should help our training.

Test for tf.keras.layers.RandomFlip() layer


```python
filp_random=tf.keras.layers.RandomFlip('horizontal_and_vertical')
#RandomFilp either
```


```python
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = filp_random(tf.expand_dims(first_image, 0))
    #apply the layer
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```


    
![png](output_29_0.png)
    


Test for tf.keras.layers.RandomRotation() layer


```python
rt_random=tf.keras.layers.RandomRotation(0.8)
#randomly rotate between[-80% * 2pi, 80% * 2pi].
```


```python
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = rt_random(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```


    
![png](output_32_0.png)
    


Train our model 2


```python
model2 = models.Sequential([
    layers.RandomRotation(0.8),
    layers.RandomFlip('horizontal_and_vertical'),
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (4, 4), activation='relu'),
    layers.MaxPooling2D((4, 4)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10) 
])
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 5s 65ms/step - loss: 1.9141 - accuracy: 0.4940 - val_loss: 0.6953 - val_accuracy: 0.5347
    Epoch 2/20
    63/63 [==============================] - 4s 60ms/step - loss: 0.7090 - accuracy: 0.5310 - val_loss: 0.6906 - val_accuracy: 0.5384
    Epoch 3/20
    63/63 [==============================] - 4s 55ms/step - loss: 0.7005 - accuracy: 0.5210 - val_loss: 0.6830 - val_accuracy: 0.5644
    Epoch 4/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.6845 - accuracy: 0.5745 - val_loss: 0.6700 - val_accuracy: 0.5879
    Epoch 5/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.6778 - accuracy: 0.5825 - val_loss: 0.6542 - val_accuracy: 0.6300
    Epoch 6/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6708 - accuracy: 0.5985 - val_loss: 0.6751 - val_accuracy: 0.5916
    Epoch 7/20
    63/63 [==============================] - 3s 51ms/step - loss: 0.6478 - accuracy: 0.6315 - val_loss: 0.6552 - val_accuracy: 0.6114
    Epoch 8/20
    63/63 [==============================] - 4s 54ms/step - loss: 0.6424 - accuracy: 0.6300 - val_loss: 0.6275 - val_accuracy: 0.6423
    Epoch 9/20
    63/63 [==============================] - 4s 53ms/step - loss: 0.6490 - accuracy: 0.6340 - val_loss: 0.6035 - val_accuracy: 0.6819
    Epoch 10/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6136 - accuracy: 0.6520 - val_loss: 0.6014 - val_accuracy: 0.6634
    Epoch 11/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6212 - accuracy: 0.6385 - val_loss: 0.5923 - val_accuracy: 0.6658
    Epoch 12/20
    63/63 [==============================] - 4s 54ms/step - loss: 0.6139 - accuracy: 0.6625 - val_loss: 0.6357 - val_accuracy: 0.6262
    Epoch 13/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6174 - accuracy: 0.6550 - val_loss: 0.5868 - val_accuracy: 0.6844
    Epoch 14/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6044 - accuracy: 0.6700 - val_loss: 0.6243 - val_accuracy: 0.6436
    Epoch 15/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6141 - accuracy: 0.6595 - val_loss: 0.6037 - val_accuracy: 0.6696
    Epoch 16/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.6070 - accuracy: 0.6655 - val_loss: 0.5677 - val_accuracy: 0.7042
    Epoch 17/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.5936 - accuracy: 0.6680 - val_loss: 0.5793 - val_accuracy: 0.7030
    Epoch 18/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.5949 - accuracy: 0.6775 - val_loss: 0.6440 - val_accuracy: 0.6015
    Epoch 19/20
    63/63 [==============================] - 3s 52ms/step - loss: 0.5759 - accuracy: 0.6900 - val_loss: 0.5501 - val_accuracy: 0.7079
    Epoch 20/20
    63/63 [==============================] - 4s 53ms/step - loss: 0.5952 - accuracy: 0.6700 - val_loss: 0.5403 - val_accuracy: 0.7290
    


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.55, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x213a0b07460>




    
![png](output_35_1.png)
    


The accuracy of my model 2 stabilized `between 64% and 70%` during training.  
My model 2 did just as good as the model 1.  
There's no overfitting this time.

## Model 3 with preprocessor

The original data's RGB channel is between `0-255`, and if we normalize it between `0-1` or `-1 to 1`, our training model may have a better performance.


```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```


```python
model3 = models.Sequential([
    preprocessor,
    #put preprocessor before conv2D layers
    layers.RandomRotation(0.8),
    layers.RandomFlip('horizontal_and_vertical'),
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(2)
])
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 6s 68ms/step - loss: 0.7756 - accuracy: 0.5140 - val_loss: 0.6767 - val_accuracy: 0.5235
    Epoch 2/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.6752 - accuracy: 0.5710 - val_loss: 0.6357 - val_accuracy: 0.6423
    Epoch 3/20
    63/63 [==============================] - 4s 55ms/step - loss: 0.6312 - accuracy: 0.6270 - val_loss: 0.6154 - val_accuracy: 0.6399
    Epoch 4/20
    63/63 [==============================] - 4s 55ms/step - loss: 0.5967 - accuracy: 0.6680 - val_loss: 0.5852 - val_accuracy: 0.6720
    Epoch 5/20
    63/63 [==============================] - 4s 55ms/step - loss: 0.5795 - accuracy: 0.6830 - val_loss: 0.5766 - val_accuracy: 0.6993
    Epoch 6/20
    63/63 [==============================] - 4s 60ms/step - loss: 0.5815 - accuracy: 0.6865 - val_loss: 0.5668 - val_accuracy: 0.6980
    Epoch 7/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.5602 - accuracy: 0.7060 - val_loss: 0.6023 - val_accuracy: 0.6696
    Epoch 8/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5613 - accuracy: 0.7035 - val_loss: 0.5869 - val_accuracy: 0.6807
    Epoch 9/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5665 - accuracy: 0.6920 - val_loss: 0.5442 - val_accuracy: 0.7178
    Epoch 10/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5536 - accuracy: 0.7140 - val_loss: 0.5897 - val_accuracy: 0.6795
    Epoch 11/20
    63/63 [==============================] - 4s 60ms/step - loss: 0.5497 - accuracy: 0.7255 - val_loss: 0.5574 - val_accuracy: 0.7153
    Epoch 12/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5453 - accuracy: 0.7050 - val_loss: 0.5449 - val_accuracy: 0.7178
    Epoch 13/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.5482 - accuracy: 0.7160 - val_loss: 0.5526 - val_accuracy: 0.7191
    Epoch 14/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.5405 - accuracy: 0.7305 - val_loss: 0.5707 - val_accuracy: 0.7092
    Epoch 15/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.5308 - accuracy: 0.7270 - val_loss: 0.5687 - val_accuracy: 0.7141
    Epoch 16/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5222 - accuracy: 0.7365 - val_loss: 0.5410 - val_accuracy: 0.7450
    Epoch 17/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.5131 - accuracy: 0.7500 - val_loss: 0.5516 - val_accuracy: 0.7290
    Epoch 18/20
    63/63 [==============================] - 4s 61ms/step - loss: 0.5141 - accuracy: 0.7480 - val_loss: 0.5495 - val_accuracy: 0.7302
    Epoch 19/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5068 - accuracy: 0.7415 - val_loss: 0.5638 - val_accuracy: 0.7141
    Epoch 20/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.5100 - accuracy: 0.7450 - val_loss: 0.5271 - val_accuracy: 0.7475
    


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.7, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2138a3b3130>




    
![png](output_40_1.png)
    


The accuracy of my model 3 stabilized `between 71% and 73%` during training.  
My model 3 did  `16%` better than the model 1.  
There's no overfitting this time.

## Model 4 Transfer Learning

We can use the model someone alreay trained that does the similar job.  
Here we use MobileNet V2 developed at Google


```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
#Freeze the convolutional base

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
    9412608/9406464 [==============================] - 0s 0us/step
    9420800/9406464 [==============================] - 0s 0us/step
    


```python
model4 = models.Sequential([
    preprocessor,
    layers.RandomRotation(0.8),
    layers.RandomFlip('horizontal_and_vertical'),
    base_model_layer,
    layers.GlobalMaxPooling2D(),
    #average over the spatial 5x5 spatial locations
    layers.Dense(2)
    #number of classes
])

```


```python
model4.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model (Functional)          (None, 160, 160, 3)       0         
                                                                     
     random_rotation_3 (RandomRo  (None, 160, 160, 3)      0         
     tation)                                                         
                                                                     
     random_flip_3 (RandomFlip)  (None, 160, 160, 3)       0         
                                                                     
     model_1 (Functional)        (None, 5, 5, 1280)        2257984   
                                                                     
     global_max_pooling2d (Globa  (None, 1280)             0         
     lMaxPooling2D)                                                  
                                                                     
     dense_8 (Dense)             (None, 2)                 2562      
                                                                     
    =================================================================
    Total params: 2,260,546
    Trainable params: 2,562
    Non-trainable params: 2,257,984
    _________________________________________________________________
    

Wow we have so much parameters!


```python
model4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 8s 89ms/step - loss: 0.9729 - accuracy: 0.7445 - val_loss: 0.1867 - val_accuracy: 0.9369
    Epoch 2/20
    63/63 [==============================] - 5s 70ms/step - loss: 0.4320 - accuracy: 0.8695 - val_loss: 0.1620 - val_accuracy: 0.9554
    Epoch 3/20
    63/63 [==============================] - 5s 72ms/step - loss: 0.3565 - accuracy: 0.8955 - val_loss: 0.1464 - val_accuracy: 0.9554
    Epoch 4/20
    63/63 [==============================] - 5s 72ms/step - loss: 0.3512 - accuracy: 0.8875 - val_loss: 0.1295 - val_accuracy: 0.9604
    Epoch 5/20
    63/63 [==============================] - 5s 70ms/step - loss: 0.3712 - accuracy: 0.8900 - val_loss: 0.1382 - val_accuracy: 0.9592
    Epoch 6/20
    63/63 [==============================] - 5s 69ms/step - loss: 0.2985 - accuracy: 0.9115 - val_loss: 0.1076 - val_accuracy: 0.9666
    Epoch 7/20
    63/63 [==============================] - 5s 70ms/step - loss: 0.3283 - accuracy: 0.8965 - val_loss: 0.1529 - val_accuracy: 0.9480
    Epoch 8/20
    63/63 [==============================] - 5s 71ms/step - loss: 0.3086 - accuracy: 0.9075 - val_loss: 0.1041 - val_accuracy: 0.9728
    Epoch 9/20
    63/63 [==============================] - 5s 71ms/step - loss: 0.2334 - accuracy: 0.9195 - val_loss: 0.0961 - val_accuracy: 0.9715
    Epoch 10/20
    63/63 [==============================] - 5s 70ms/step - loss: 0.2530 - accuracy: 0.9215 - val_loss: 0.0865 - val_accuracy: 0.9728
    Epoch 11/20
    63/63 [==============================] - 5s 70ms/step - loss: 0.2788 - accuracy: 0.9155 - val_loss: 0.1147 - val_accuracy: 0.9678
    Epoch 12/20
    63/63 [==============================] - 5s 72ms/step - loss: 0.2369 - accuracy: 0.9230 - val_loss: 0.0877 - val_accuracy: 0.9715
    Epoch 13/20
    63/63 [==============================] - 5s 71ms/step - loss: 0.2502 - accuracy: 0.9255 - val_loss: 0.0916 - val_accuracy: 0.9703
    Epoch 14/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.2153 - accuracy: 0.9225 - val_loss: 0.1126 - val_accuracy: 0.9653
    Epoch 15/20
    63/63 [==============================] - 5s 71ms/step - loss: 0.1962 - accuracy: 0.9335 - val_loss: 0.0932 - val_accuracy: 0.9641
    Epoch 16/20
    63/63 [==============================] - 5s 70ms/step - loss: 0.2190 - accuracy: 0.9230 - val_loss: 0.1007 - val_accuracy: 0.9641
    Epoch 17/20
    63/63 [==============================] - 5s 73ms/step - loss: 0.2108 - accuracy: 0.9210 - val_loss: 0.0961 - val_accuracy: 0.9666
    Epoch 18/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.2157 - accuracy: 0.9265 - val_loss: 0.0899 - val_accuracy: 0.9703
    Epoch 19/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.2075 - accuracy: 0.9295 - val_loss: 0.0869 - val_accuracy: 0.9678
    Epoch 20/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.2190 - accuracy: 0.9230 - val_loss: 0.1151 - val_accuracy: 0.9641
    


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.95, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2138a3ef4f0>




    
![png](output_48_1.png)
    


The accuracy of my model 4 stabilized `between 95% and 98%` during training.  
My model 4 did  `48%` better than the model 1.  
There's no overfitting this time.

##  Score on Test Data


```python
model4.evaluate(test_dataset)
```

    6/6 [==============================] - 1s 65ms/step - loss: 0.1258 - accuracy: 0.9688
    




    [0.12578193843364716, 0.96875]



The accuracy is 0.9688, which is extremely high!  
The model others write is great!
