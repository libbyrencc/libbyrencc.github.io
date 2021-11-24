---
layout: post
title: Post 6
---

# Blog Post 6--Fake news classifier


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

import plotly.express as px 
import plotly.io as pio
import pandas as pd
from matplotlib import pyplot as plt
pio.templates.default = "plotly_white"
```

### Acquire Training Data


```python
# Read csv from url
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
train_org=pd.read_csv(train_url)
```


```python
train_org.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Drop first column here.


```python
train_org= train_org.iloc[: , 1:]
```


```python
train_org.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Make a Dataset

First we need to remove stopwords(such as 'and','but', 'the') from `text` and `title`.  

Then we construct the `tf.data.Dataset` with input `(title,text)` and output `fake`.


```python
def make_dataset(data):
    #Import stopwords from package
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    #Remove stopwords from text and title
    data['title']=data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    data['text']=data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #construct the Dataset from dictionaries (title:,text:)
    tfdata=tf.data.Dataset.from_tensor_slices(
        ({'title': data['title'],'text':data['text']},
         data['fake']
        )
    )
    tfdata=tfdata.shuffle(buffer_size = len(tfdata))
    #return as batch to imporve the performance
    tfdata=tfdata.batch(100)
    return tfdata
```

Call the function to make the Dataset from our raw data


```python
tfdata=make_dataset(train_org)
```

Split 20% of data for validation


```python
train_size=int(0.8*len(tfdata))
val_size=int(0.2*len(tfdata))
train = tfdata.take(train_size)
val   = tfdata.skip(train_size).take(val_size)
```


```python
len(train),len(val)
```




    (180, 45)



Check the base rate


```python
#The labels_iterator will return 0 or 1 to indicate whether the news is fake
labels_iterator= train.unbatch().map(lambda imput,fake: fake).as_numpy_iterator()
true__freq=0
false_freq=0
for label in labels_iterator:
    if label==0:
        true__freq=true__freq+1
    else:
        false_freq=false_freq+1
print("true__freq = ",true__freq)
print("false_freq = ",false_freq)
```

    true__freq =  8536
    false_freq =  9464
    

Base line is 9511/18000=0.5284

### Create Models

#### Model 1--Title only Model

Below is code for **Text Standardization and Vectorization**.  


Standardization step removes some text like capitals, punctuation, HTML elements or other non-semantic content.  
Vectorization step will reperesent text as a vector


```python
size_vocabulary = 2000
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation
```


```python
vectorize_layer_title = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary,
    output_mode='int',
    output_sequence_length=500)
```


```python
#vectorize layer
vectorize_layer_title.adapt(train.map(lambda x, y: x['title']))
```


```python
#Setting title input
title_input=keras.Input(
    shape =(1,),
    name='title',
    dtype='string'
)
```

Add layers same as lecture notes.

ref: https://nbviewer.org/github/PhilChodrow/PIC16B/blob/master/lectures/tf/tf-4.ipynb


```python
title_features = vectorize_layer_title(title_input)
title_features = layers.Embedding(size_vocabulary, 3, name = "embedding")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)
```


```python
title_output = layers.Dense(2,name='fake')(title_features)
```

Specify the inputs so the model_1 only take `title` column as inputs


```python
model_1 = keras.Model(inputs=title_input,
                      outputs=title_output,
                      name="title_model")
```


```python
model_1.summary()
```

    Model: "title_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 500)              0         
     torization)                                                     
                                                                     
     embedding (Embedding)       (None, 500, 3)            6000      
                                                                     
     dropout (Dropout)           (None, 500, 3)            0         
                                                                     
     global_average_pooling1d (G  (None, 3)                0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 3)                 0         
                                                                     
     dense (Dense)               (None, 32)                128       
                                                                     
     fake (Dense)                (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 6,194
    Trainable params: 6,194
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model_1.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```

Then we train the model_1


```python
history = model_1.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```


```python
plt.plot(history.history["accuracy"],label = "training")
plt.plot(history.history["val_accuracy"],label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.97, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1cb97fb8340>




    
![png](output_34_1.png)
    


The accuracy of my model 1 stabilized above 97% during training.  
And there's no overfitting.

#### Model_2 -- Text only Model  

Details same as model_1


```python
# Setting for text_input_only
text_input=keras.Input(
    shape =(1,),
    name='text',
    dtype='string'
)
vectorize_layer_text = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, 
    output_mode='int',
    output_sequence_length=250)
vectorize_layer_text.adapt(train.map(lambda x, y: x['text']))
text_features = vectorize_layer_title(text_input)
text_features = layers.Embedding(size_vocabulary, 3, name = "embedding")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
#A little difference here in order to speed up, since text's size is bigger
text_features = layers.Dense(16, activation='relu')(text_features)
text_output = layers.Dense(2,name='fake')(text_features)
```


```python
model_2= keras.Model(inputs=text_input,outputs=text_output,name="text_model")
```


```python
model_2.summary()
```

    Model: "text_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text (InputLayer)           [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 500)              0         
     torization)                                                     
                                                                     
     embedding (Embedding)       (None, 500, 3)            6000      
                                                                     
     dropout_2 (Dropout)         (None, 500, 3)            0         
                                                                     
     global_average_pooling1d_1   (None, 3)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_3 (Dropout)         (None, 3)                 0         
                                                                     
     dense_1 (Dense)             (None, 16)                64        
                                                                     
     fake (Dense)                (None, 2)                 34        
                                                                     
    =================================================================
    Total params: 6,098
    Trainable params: 6,098
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model_2.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
history = model_2.fit(train, 
                    validation_data=val,
                    epochs = 20
                      #less epoches for speed
                     )
```


```python
plt.plot(history.history["accuracy"],label = "training")
plt.plot(history.history["val_accuracy"],label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.97, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1cb97fa1640>




    
![png](output_42_1.png)
    


The accuracy of my model 2 stabilized above 97% during training, a little bit worse than model 1.  
And there's no overfitting.

#### Model 3 with both `title` and `text` inputs

Add layers to both inputs


```python
#Same as model 1
title_features = vectorize_layer_title(title_input)
title_features = layers.Embedding(size_vocabulary, 3, name = "embedding_title")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)

#Same as model 2
text_features = vectorize_layer_title(text_input)
text_features = layers.Embedding(size_vocabulary, 3, name = "embedding_text")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(16, activation='relu')(text_features)
```

`Concatenate` the output of `title` pipline with `text` pipline


```python
main = layers.concatenate([title_features, text_features], axis = 1)
```


```python
main = layers.Dense(32, activation='relu')(main)
```


```python
#Setting output
output = layers.Dense(2, name = "fake")(main)
```


```python
model_3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = output
)
```


```python
model_3.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     title (InputLayer)             [(None, 1)]          0           []                               
                                                                                                      
     text (InputLayer)              [(None, 1)]          0           []                               
                                                                                                      
     text_vectorization (TextVector  (None, 500)         0           ['title[0][0]',                  
     ization)                                                         'text[0][0]']                   
                                                                                                      
     embedding_title (Embedding)    (None, 500, 3)       6000        ['text_vectorization[4][0]']     
                                                                                                      
     embedding_text (Embedding)     (None, 500, 3)       6000        ['text_vectorization[5][0]']     
                                                                                                      
     dropout_8 (Dropout)            (None, 500, 3)       0           ['embedding_title[0][0]']        
                                                                                                      
     dropout_10 (Dropout)           (None, 500, 3)       0           ['embedding_text[0][0]']         
                                                                                                      
     global_average_pooling1d_4 (Gl  (None, 3)           0           ['dropout_8[0][0]']              
     obalAveragePooling1D)                                                                            
                                                                                                      
     global_average_pooling1d_5 (Gl  (None, 3)           0           ['dropout_10[0][0]']             
     obalAveragePooling1D)                                                                            
                                                                                                      
     dropout_9 (Dropout)            (None, 3)            0           ['global_average_pooling1d_4[0][0
                                                                     ]']                              
                                                                                                      
     dropout_11 (Dropout)           (None, 3)            0           ['global_average_pooling1d_5[0][0
                                                                     ]']                              
                                                                                                      
     dense_5 (Dense)                (None, 32)           128         ['dropout_9[0][0]']              
                                                                                                      
     dense_6 (Dense)                (None, 16)           64          ['dropout_11[0][0]']             
                                                                                                      
     concatenate_1 (Concatenate)    (None, 48)           0           ['dense_5[0][0]',                
                                                                      'dense_6[0][0]']                
                                                                                                      
     dense_7 (Dense)                (None, 32)           1568        ['concatenate_1[0][0]']          
                                                                                                      
     fake (Dense)                   (None, 2)            66          ['dense_7[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 13,826
    Trainable params: 13,826
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

Use the `plot_model` function to have a better visualization for layers


```python
keras.utils.plot_model(model_3)
```




    
![png](output_54_0.png)
    




```python
model_3.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
history = model_3.fit(train, 
                    validation_data=val,
                    epochs = 30)
```


```python
plt.plot(history.history["accuracy"],label = "training")
plt.plot(history.history["val_accuracy"],label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.axhline(y=0.97, color='r', linestyle='-')
plt.legend()
```




    <matplotlib.legend.Legend at 0x20b7a740ac0>




    
![png](output_57_1.png)
    


Good news!
The accuracy of my model 3 stabilized very close to 100% during training.  
And there's no overfitting.

## Model Evaluation


```python
#Read the data for testing
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test_org=pd.read_csv(test_url)
test_org=test_org.iloc[: , 1:]
```

Use the function `make_dataset` to process our test data


```python
test_data=make_dataset(test_org)
```


```python
model_3.evaluate(test_data)
```

    225/225 [==============================] - 3s 11ms/step - loss: 0.0171 - accuracy: 0.9955
    




    [0.0171230249106884, 0.9954563975334167]



The accuracy is 0.9954, which is extremely close to 1!  
GOOD job!

### Embedding Visualization  

Get the weights from both `embedding_title` layer and `embedding_text` layer.  
Since the accuracy of model_1 is higher than model_2, we give `embedding_title` more weight as 0.6, and `embedding_text` with 0.4.


```python
weights = model_3.get_layer('embedding_title').get_weights()[0]*0.6+model_3.get_layer('embedding_text').get_weights()[0]*0.4
weights
```




    array([[ 4.4220760e-03, -3.7515955e-03,  2.9505624e-03],
           [-1.0663554e-01,  2.0389837e-01, -1.8373589e-01],
           [ 1.0753224e-01, -1.6184452e-01, -7.7932082e-02],
           ...,
           [-7.4841455e-02,  1.1953704e-01, -1.2076619e-01],
           [ 4.2736549e-02, -4.0900894e-05,  1.1663854e-02],
           [ 1.8610361e-01, -1.7711309e-01,  1.2361175e-01]], dtype=float32)




```python
# get the vocabulary from our data prep
vocab = vectorize_layer_text.get_vocabulary()
```

Reduce our data from 3d to a 2d using `PCA`


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)
```


```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
embedding_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>x0</th>
      <th>x1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>-0.095760</td>
      <td>0.007419</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[UNK]</td>
      <td>0.196556</td>
      <td>0.034882</td>
    </tr>
    <tr>
      <th>2</th>
      <td>said</td>
      <td>-0.170584</td>
      <td>-0.180881</td>
    </tr>
    <tr>
      <th>3</th>
      <td>trump</td>
      <td>2.154822</td>
      <td>0.275418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>the</td>
      <td>1.979405</td>
      <td>0.249041</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>christians</td>
      <td>-0.253563</td>
      <td>0.038703</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>st</td>
      <td>-0.023181</td>
      <td>0.108869</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>quoted</td>
      <td>0.093434</td>
      <td>0.020968</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>products</td>
      <td>-0.118910</td>
      <td>-0.002864</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>outlets</td>
      <td>-0.358968</td>
      <td>-0.083210</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 3 columns</p>
</div>




```python
import plotly.express as px 
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

fig.show()
```
![png](1.png)

Embedding finished! 
This embedding is really interesting.   
In the right part **trump, president,obama,election,republican** is close to each other when detecting "fake news".   
And **korea** is close to **missiles**.  
More than that, **women** is close to **power**.    

These words somekind reflect the ralations between each other, the closer group may share some same topics.
