#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[3]:


train_directory=train_directory = 'C:/Users/ACER/OneDrive/Desktop/dataset/train'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',      
    label_mode='categorical', 
    color_mode='rgb',        
    batch_size=32,           
    image_size=(224, 224), 
    shuffle=True,           
    seed=None,                
    validation_split=None,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)


# In[4]:


validation_directory=train_directory = 'C:/Users/ACER/OneDrive/Desktop/dataset/validation'

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',      
    label_mode='categorical',
    #class_mode='categorical',
    color_mode='rgb',        
    batch_size=32,           
    image_size=(224, 224), 
    shuffle=True,           
    seed=None,                
    validation_split=None,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)


# In[5]:


import tensorflow as tf

class_names = train_dataset.class_names

print("Inferred Class Names:", class_names)


# In[6]:


for images, labels in train_dataset.take(1): 
    image = images[0].numpy()  
    label = tf.argmax(labels[0]).numpy()  
    image = image.astype(float) / 255.0
    plt.imshow(image, interpolation='bilinear') 
    plt.title(f"Label: {label}")
    plt.show()


# In[7]:


#main model of CNN
model = Sequential([
    Conv2D(32 ,kernel_size=3, activation = 'relu' , input_shape=(224,224,3)),
    Conv2D(32 ,kernel_size=3, activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(64 ,kernel_size=3, activation = 'relu' ),
    Conv2D(64 ,(3,3) , activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    
    Dense(512,activation='relu'),
     Dense(512,activation='relu'),
    Dense(36,activation='softmax')
])


# In[8]:


#minimizing the errors by optimizer such as adam
model.compile(loss='categorical_crossentropy' ,optimizer='adam' ,metrics=['accuracy'])


# In[9]:


history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset
    )


# In[11]:


test_directory=train_directory = 'C:/Users/ACER/OneDrive/Desktop/dataset/test'
test_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',      
    label_mode='categorical', 
    #class_mode='categorical',
    color_mode='rgb',        
    batch_size=32,           
    image_size=(224, 224), 
    shuffle=True,           
    seed=None,                
    validation_split=None,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)


# In[12]:


from tensorflow.keras.preprocessing import image
image_path="C:/Users/ACER/OneDrive/Desktop/dataset/test/banana/Image_3.JPG"
img = cv2.imread(image_path)
img = image.load_img(image_path, target_size=(224, 224))  
# Convert the image to a NumPy array
img_array = image.img_to_array(img)
img_array = np.array([img_array])
prediction = model.predict(img_array)


# In[13]:


print(prediction)


# In[14]:


result_index = np.where(prediction[0]==max(prediction[0]))


# In[15]:


img=cv2.imread(image_path)
plt.imshow(img)
plt.title("Test Image")
plt.show()
print("Prediction is {}".format(test_dataset.class_names[result_index[0][0]]))


# In[21]:


model.save('my_model.keras')


# In[23]:


#One more test
image_path="C:/Users/ACER/OneDrive/Desktop/dataset/test/corn/Image_4.jpg"
img = cv2.imread(image_path)
img = image.load_img(image_path, target_size=(224,224))  
# Convert the image to a NumPy array
img_array = image.img_to_array(img)
img_array = np.array([img_array])
prediction = model.predict(img_array)


# In[24]:


result_index = np.where(prediction[0]==max(prediction[0]))


# In[25]:


img=cv2.imread(image_path)
plt.imshow(img)
plt.title("Test Image")
plt.show()
print("Prediction is {}".format(test_dataset.class_names[result_index[0][0]]))


# In[26]:


print("Validation Set Accuracy: {} %" .format(history.history['val_accuracy'][-1]*100))

print("Training Set Accuracy: {} %" .format(history.history['accuracy'][-1]*100))


# In[18]:


from tensorflow.keras.preprocessing import image
image_path="C:/Users/ACER/OneDrive/Desktop/dataset/test/garlic/Image_1.jpg"
img = cv2.imread(image_path)
img = image.load_img(image_path, target_size=(224, 224))  
# Convert the image to a NumPy array
img_array = image.img_to_array(img)
img_array = np.array([img_array])
prediction = model.predict(img_array)


# In[19]:


result_index = np.where(prediction[0]==max(prediction[0]))


# In[20]:


test_directory=train_directory = 'C:/Users/ACER/OneDrive/Desktop/dataset/test'
test_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',      
    label_mode='categorical', 
    #class_mode='categorical',
    color_mode='rgb',        
    batch_size=32,           
    image_size=(224, 224), 
    shuffle=True,           
    seed=None,                
    validation_split=None,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)
img=cv2.imread(image_path)
plt.imshow(img)
plt.title("Test Image")

plt.show()
print("Prediction is {}".format(test_dataset.class_names[result_index[0][0]]))


# In[27]:


epochs=[i for i in range (1,25)]
plt.plot(history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No of Epochs')
plt.ylabel('Training Accuracy & Validation Accuracy')
plt.title('Visualization of Training Accuracy & Validation Accuracy Result')
plt.legend()
plt.savefig('C:/Users/ACER/OneDrive/Desktop/pic2.svg')
plt.show()


# In[28]:


plt.plot(history.history['loss'],color='red',label='Training Loss')
plt.plot(history.history['val_loss'],color='blue',label='Validation Loss')
plt.xlabel('No of Epochs')
plt.ylabel('Training Loss & Validation Loss')
plt.title('Visualization of Training Loss & Validation Loss Result')
plt.legend()
plt.savefig('C:/Users/ACER/OneDrive/Desktop/pic3.svg')
plt.show()


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools



true_labels = []
predicted_labels = []


for images, labels in validation_dataset:
    true_labels.extend(np.argmax(labels, axis=1)) 
    predictions = saved_model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))  

# Create a classification report
class_names = validation_dataset.class_names
report = classification_report(true_labels, predicted_labels, target_names=class_names)


print("Classification Report:\n", report)


confusion = confusion_matrix(true_labels, predicted_labels)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
plot_confusion_matrix(confusion, class_names)
plt.savefig('C:/Users/aryan/OneDrive/Desktop/conf.svg')
plt.show()


# In[32]:


metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [0.96, 0.96, 0.95, 0.95]

plt.figure(figsize=(8, 6))
plt.plot(metrics, scores, marker='o', linestyle='-')
plt.title('Model Evaluation Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('C:/Users/ACER/OneDrive/Desktop/pic1.svg')
plt.show()


# In[ ]:




