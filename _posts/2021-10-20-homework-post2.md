---
layout: post
title: Post 2
---

# Webscraping

In this blog post I will discuss the following question:  
    *What movie or TV shows share actors with your favorite movie or show?*

And the link to my project repository:  
    https://github.com/libbyrencc/post/tree/main/post2

### 1.Setup

First, we need to setup our webscraping environment, remember to install scrapy package in your conda environment.

And run the following command in your terminal to initialize the project:


```python
conda activate PIC16B
scrapy startproject IMDB_scraper
cd IMDB_scraper

```

Now the file of scraper should be constructed correctly. Go to /IMDB_scraper/spiders/, and create a file "imdb_spider.py". And add following code to the start of the file


```python
import scrapy

class ImdbSpider(scrapy.Spider):
    name = 'imdb_spider'
    
    start_urls = ['https://www.imdb.com/title/tt1533117/']
                #You can change your favorite film's url here
                #Here I choose my favorite film: Let the Bullets Fly
```

## Implement three parsing methods for the ImdbSpider class

### 1. prase(self,response)

First we start writing the prase method. It should start on a movie page, and then navigate to the Cast&Crew page.  
And call the next method prase_full_credits(self,response).

Here I use the CSS selector to find the URL of the Cast&Crew page
and yeild scrapy.Request


```python
 def parse(self, response):
        next_page = response.css("a.ipc-metadata-list-item__icon-link").attrib["href"]
        #Find the partial url by located tag <a> with
        #class= ipc-metadata-list-item__icon-link
        # and url is in "herf"
        if next_page:           
            next_page=response.urljoin(next_page) 
            #Here we make the URL complete
            yield scrapy.Request(next_page,callback= self.prase_full_credits)
            #Call next function
```

### 2. prase_full_credits(self, response)  
Now we are at the Cast&Crew page, next we should look at each actors' page, and call the next method prase_actor_page(self,response)  

We will use the css selector and the list comprehension to mimic the process of clicking on the headshots on this page.

Here we select `<a>` tag, which is under `<td>` tag with `class ='primary_photo'`, and url is still in `"herf"`.
Then we iterate each Cast&Crews' URL, use it as a parameter to call next method.


```python
def prase_full_credits(self,response):
        for next_page in [a.attrib["href"] for a in response.css("td.primary_photo a")]:
            next_page=response.urljoin(next_page)
            yield scrapy.Request(next_page,callback= self.prase_actor_page)
```

### 3. prase_actor_page(self,response)  
In each actors' page, we should first look up the actor's name.  
Then we find each movie or TV name below.  
Finally we yield a dictionary with two key-value pairs, of the form {"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}.


```python
def prase_actor_page(self,response):
        actor_name=response.css("h1.header span.itemprop::text").get() 
        #get actor's name
        #the name is in <h1 class="header"><span class+"itemprop">name<\span><\h1>
        for quote in response.css("div.filmo-row"): 
            #get a list of  his/her works
            movie_or_TV_name=quote.css("b a::text").get()
            # the name is in <b> <a>work name<\a><\b>, under tag <div> with class ="filmo-row"
            yield {"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name} 
            #yeild an dictionary
```

Now we are done! The finally project looks like below:


```python
import scrapy

class ImdbSpider(scrapy.Spider):
    name = 'imdb_spider'
    
    start_urls = ['https://www.imdb.com/title/tt1533117/']

    def parse(self, response):
        next_page = response.css("a.ipc-metadata-list-item__icon-link").attrib["href"]
        if next_page:           
            next_page=response.urljoin(next_page)
            yield scrapy.Request(next_page,callback= self.prase_full_credits)

    
    
    def prase_full_credits(self,response):
        for next_page in [a.attrib["href"] for a in response.css("td.primary_photo a")]:
            next_page=response.urljoin(next_page)
            yield scrapy.Request(next_page,callback= self.prase_actor_page)

    def prase_actor_page(self,response):
        actor_name=response.css("h1.header span.itemprop::text").get()                             
        for quote in response.css("div.filmo-row"):
            movie_or_TV_name=quote.css("b a::text").get()
            yield {"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}
```

And to run this web scraper we build, we need to run the command below in this repository.  
It will save all the results to results.csv


```python
scrapy crawl imdb_spider -o results.csv
```

## Data analysis  

Import some useful package first


```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
```

read reasults.csv as pandas dataframe


```python
results=pd.read_csv("results.csv")
results
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
      <th>actor</th>
      <th>movie_or_TV_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Marco Ma</td>
      <td>Gone with the Bullets</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marco Ma</td>
      <td>Let the Bullets Fly</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Marco Ma</td>
      <td>Pk.com.cn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marco Ma</td>
      <td>The Blossom of Roses</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Marco Ma</td>
      <td>Let the Bullets Fly</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>885</th>
      <td>You Ge</td>
      <td>The Troubleshooters</td>
    </tr>
    <tr>
      <th>886</th>
      <td>You Ge</td>
      <td>Shan de nu er</td>
    </tr>
    <tr>
      <th>887</th>
      <td>You Ge</td>
      <td>Sheng xia he ta de wei hun fu</td>
    </tr>
    <tr>
      <th>888</th>
      <td>You Ge</td>
      <td>The 11th China Movie Awards</td>
    </tr>
    <tr>
      <th>889</th>
      <td>You Ge</td>
      <td>I Love My Family</td>
    </tr>
  </tbody>
</table>
<p>890 rows × 2 columns</p>
</div>



Now we use aggregation to count each movie or TV contians how many shared actors


```python
k=results.groupby(["movie_or_TV_name"])["actor"].aggregate(len)
k=k.reset_index()
k
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
      <th>movie_or_TV_name</th>
      <th>actor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"Swordsmen of the Passes"</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100 Ways to Murder Your Wife</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1911</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1921</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000 Blockbuster Entertainment Awards</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>663</th>
      <td>Zhi Ming Yuan Wang</td>
      <td>1</td>
    </tr>
    <tr>
      <th>664</th>
      <td>Zhongkui: Snow Girl and the Dark Crystal</td>
      <td>1</td>
    </tr>
    <tr>
      <th>665</th>
      <td>Zou dao di</td>
      <td>1</td>
    </tr>
    <tr>
      <th>666</th>
      <td>Zou xi kou</td>
      <td>1</td>
    </tr>
    <tr>
      <th>667</th>
      <td>Ôsama no buranchi</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>668 rows × 2 columns</p>
</div>



Then sort it by actors number


```python
k=k.sort_values(by=["actor"],ascending=False).reset_index(drop=True)
k.head(10)
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
      <th>movie_or_TV_name</th>
      <th>actor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Let the Bullets Fly</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gone with the Bullets</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Sun Also Rises</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Founding of a Republic</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hidden Man</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>If You Are the One 2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I Love My Family</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Robbers</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Assembly</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Beginning of the Great Revival</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Then we can try to visulaize it.


```python
sns.barplot(x='movie_or_TV_name', y='actor', data=k.head(5))
sns.set(rc={'figure.figsize':(15,8.27)})
```


    
![png](output_27_0.png)
    


It shows that Gone with the Bullets may be a good choice for me since it shares most same actors.

