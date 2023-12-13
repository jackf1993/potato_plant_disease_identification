#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[46]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from experta import *


# In[4]:


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3


# # Reading Dataset and Inspection

# In[5]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle= True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)


# In[6]:


class_names = dataset.class_names
print(class_names)


# In[7]:


len(dataset)


# In[8]:


# inspect 2 batches of 32
for image_batch, label_batch in dataset.take(2).as_numpy_iterator():
    print(image_batch.shape)
    print(label_batch)


# In[9]:


# Inspecting first image as a numpy iterator
for image_batch, label_batch in dataset.take(1).as_numpy_iterator():
    print(image_batch[0])


# In[10]:


# Show image example with plt
plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8')) # expects 3D array
        plt.title(class_names[label_batch[i]])
        plt.axis('off')


# # Preprocessing

# ## Split Dataset

# In[11]:


def get_dataset_partition(ds, train_split = 0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 10000):
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed =12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[12]:


train_ds, val_ds, test_ds = get_dataset_partition(dataset)


# In[13]:


# Optimizing image pipeline
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)


# ## Scaling

# In[14]:


resize_and_scale = tf.keras.Sequential([
                       layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
                       layers.experimental.preprocessing.Rescaling(1.0/255)
                    ])


# ## Data Augmentation

# In[15]:


data_augmentation = tf.keras.Sequential([
                        layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                        layers.experimental.preprocessing.RandomRotation(0.2)

                    ])


# ## Build Model

# In[16]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE,CHANNELS)
n_classes = 3

model = tf.keras.Sequential([
    resize_and_scale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation ='softmax')
])

model.build(input_shape = input_shape)


# In[17]:


model.summary()


# In[18]:


model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)


# In[19]:


EPOCHS = 30
history = model.fit(
            train_ds,
            epochs = EPOCHS,
            batch_size = BATCH_SIZE,
            verbose = 1,
            validation_data = val_ds
)


# In[25]:


score = model.evaluate(test_ds)


# In[24]:


score


# In[23]:


history


# In[27]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


# In[159]:


plt.figure(figsize=(8, 8))  # Adjust figure size as needed

# Get a random index from the test dataset
random_index = np.random.choice(len(test_ds))

# Take a batch of images starting from the random index
for images, labels in test_ds.skip(random_index).take(1):
    ax = plt.subplot(1, 1, 1)  # Only one image, so one subplot
    plt.imshow(images[0].numpy().astype("uint8"))

    predicted_class, confidence = predict(model, images[0].numpy())
    actual_class = class_names[labels[0]]

    plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}. \n Confidence: {confidence}")

    plt.axis('off')

plt.show()


# ## Knowledge Engine

# In[99]:


from experta import *

class Pred(Fact):
    """ ask about temperature """
    pass

class Solution(KnowledgeEngine):
    
    @Rule(Pred(pred='Potato___Early_blight'))
    def Potato___Early_blight(self):
        print('Potato Early Blight')
        print('Treatment: ')
        print('  1. Prune or stake plants to improve air circulation and reduce fungal problems.')
        print('  2. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut.')
        print('  3. Keep the soil under plants clean and free of garden debris. Add a layer of organic \n     compost to prevent the spores from splashing back up onto vegetation.')
        print('  4. Drip irrigation and soaker hoses can be used to help keep the foliage dry.')
        print('  5. For best control, apply copper-based fungicides early, two weeks before disease \n     normally appears or when weather forecasts predict a long period of wet weather. \n     Alternatively, begin treatment when disease first appears, and repeat every 7-10 days \n     for as long as needed.')
        print('  6. Containing copper and pyrethrins, Bonide® Garden Dust is a safe, one-step control for \n     many insect attacks and fungal problems. For best results, cover both the tops and \n     undersides of leaves with a thin uniform film or dust. Depending on foliage density, 10 oz \n     will cover 625 sq ft. Repeat applications every 7-10 days, as needed.')
        print('  7. SERENADE Garden is a broad spectrum, preventative bio-fungicide recommended for \n     the control or suppression of many important plant diseases. For best results, treat prior \n     to foliar disease development or at the first sign of infection. Repeat at 7-day intervals or \n     as needed.')
        print('  8. Remove and destroy all garden debris after harvest and practice crop rotation the following year.')
        print('  9. Burn or bag infected plant parts. Do NOT compost.')
        print('Reference:')
        print('  Vinje, E. (2023, August 6). Early blight treatment and control. Planet Natural. \n  https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/#:\n  ~:text=For%20best%20control%2C%20apply%20copper,for%20as%20long%20as%20needed.')
        
    @Rule(Pred(pred='Potato___Late_blight'))
    def Potato___Late_blight(self):
        print('Potato Late Blight')
        print('With late blighted potatoes, there are three phases of treatment: during plant growth, After harvest, and resistance:')
        print('During Plant Growth: ')
        print('  1. Examine potato plants daily for foliar symptoms.')
        print('  2. Save plants if few symptoms, hot and dry weather, and no nearby late blight outbreaks.')
        print('  3. Remove affected tissue regularly and apply fungicides to slow late blight development.')
        print('  4. Avoid wetting leaves with overhead irrigation and eliminate weeds and extra branches \n     for better air circulation.')
        print('  5. Destroy severely infected plants.')
        print('  6. Commercial growers use fungicides early and often, while homeowners use \n     chlorothalonil-based products before disease appears.')
        print('After Harvest: ')
        print('  NOTE: Late blight pathogen survives only in living plant tissue.')
        print('  NOTE: Treating the soil is not effective against late blight.')
        print('  1. Destroy potato cull piles. Freeze infected potatoes over winter or dispose of them.\n     Prevent the growth of volunteer potatoes.')
        print('  2. Inspect transplants by carefully inspect purchased transplants for disease symptoms. \n     Select healthy and vigorously growing transplants.')
        print('Resistance: ')
        print('  NOTE: No potato varieties have complete resistance to late blight.')
        print('  1. Some varieties show tolerance; disease develops slowly or not at all. Recommended Potato \n     varieties: Kennebec, Sebago, Allegany, Jacqueline Lee.')
        print('Reference: ')
        print('  Gugino, B. (n.d.). Tomato-potato late blight in the Home Garden. Penn State Extension. \n   https://extension.psu.edu/tomato-potato-late-blight-in-the-home-garden ')
        
        
    @Rule(Pred(pred='Potato Healthy'))
    def Potato___healthy(self):
        print('Potato Healthy')
        print('No Recommendation')


# In[160]:


engine = Solution()
engine.reset()

try:
    if predicted_class != actual_class:
        raise ValueError("Incorrect Prediction")
        
    engine.declare(Pred(pred=predicted_class))
    engine.run()

except ValueError as e:
    print(f"Error: {e}")

