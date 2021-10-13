---
layout: post
title: Post 1
---

# Data analysis of Climate change

## I. Create a Database


```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# First import useful tools for data analysis
```

Then we need to access and read three tables.(temperatures stations and countries)  
Using pd.read_csv()


```python
temps = pd.read_csv('temps.csv')
#Since I already download the file, just import is enougha
```


```python
temps.head(5)
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
      <th>ID</th>
      <th>Year</th>
      <th>VALUE1</th>
      <th>VALUE2</th>
      <th>VALUE3</th>
      <th>VALUE4</th>
      <th>VALUE5</th>
      <th>VALUE6</th>
      <th>VALUE7</th>
      <th>VALUE8</th>
      <th>VALUE9</th>
      <th>VALUE10</th>
      <th>VALUE11</th>
      <th>VALUE12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>-89.0</td>
      <td>236.0</td>
      <td>472.0</td>
      <td>773.0</td>
      <td>1128.0</td>
      <td>1599.0</td>
      <td>1570.0</td>
      <td>1481.0</td>
      <td>1413.0</td>
      <td>1174.0</td>
      <td>510.0</td>
      <td>-39.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1962</td>
      <td>113.0</td>
      <td>85.0</td>
      <td>-154.0</td>
      <td>635.0</td>
      <td>908.0</td>
      <td>1381.0</td>
      <td>1510.0</td>
      <td>1393.0</td>
      <td>1163.0</td>
      <td>994.0</td>
      <td>323.0</td>
      <td>-126.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1963</td>
      <td>-713.0</td>
      <td>-553.0</td>
      <td>-99.0</td>
      <td>541.0</td>
      <td>1224.0</td>
      <td>1627.0</td>
      <td>1620.0</td>
      <td>1596.0</td>
      <td>1332.0</td>
      <td>940.0</td>
      <td>566.0</td>
      <td>-108.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1964</td>
      <td>62.0</td>
      <td>-85.0</td>
      <td>55.0</td>
      <td>738.0</td>
      <td>1219.0</td>
      <td>1442.0</td>
      <td>1506.0</td>
      <td>1557.0</td>
      <td>1221.0</td>
      <td>788.0</td>
      <td>546.0</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1965</td>
      <td>44.0</td>
      <td>-105.0</td>
      <td>38.0</td>
      <td>590.0</td>
      <td>987.0</td>
      <td>1500.0</td>
      <td>1487.0</td>
      <td>1477.0</td>
      <td>1377.0</td>
      <td>974.0</td>
      <td>31.0</td>
      <td>-178.0</td>
    </tr>
  </tbody>
</table>
</div>



As provided, the data set contains the following columns: 

- `ID`: the ID number of the station. We can use this to figure out which country the station is in, as well as the spatial location of the station. 
- `Year`: the year of the measurement. 
- `VALUE1`-`VALUE12`: the temperature measurements themselves. `VALUE1` contains the temperature measurements for January, `VALUE2` for February, and so on. 
- The measurements are in hundredths of a degree, Celsius. 

And such data is hard to deal with, we may need to do some changes so that each months' data is in a single column. 

First, we could convert all the columns into a mulit-index for the data frame. Here we use ID and Year.


```python
temps=temps.set_index(keys=["ID","Year"])
temps.head()
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
      <th></th>
      <th>VALUE1</th>
      <th>VALUE2</th>
      <th>VALUE3</th>
      <th>VALUE4</th>
      <th>VALUE5</th>
      <th>VALUE6</th>
      <th>VALUE7</th>
      <th>VALUE8</th>
      <th>VALUE9</th>
      <th>VALUE10</th>
      <th>VALUE11</th>
      <th>VALUE12</th>
    </tr>
    <tr>
      <th>ID</th>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">ACW00011604</th>
      <th>1961</th>
      <td>-89.0</td>
      <td>236.0</td>
      <td>472.0</td>
      <td>773.0</td>
      <td>1128.0</td>
      <td>1599.0</td>
      <td>1570.0</td>
      <td>1481.0</td>
      <td>1413.0</td>
      <td>1174.0</td>
      <td>510.0</td>
      <td>-39.0</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>113.0</td>
      <td>85.0</td>
      <td>-154.0</td>
      <td>635.0</td>
      <td>908.0</td>
      <td>1381.0</td>
      <td>1510.0</td>
      <td>1393.0</td>
      <td>1163.0</td>
      <td>994.0</td>
      <td>323.0</td>
      <td>-126.0</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>-713.0</td>
      <td>-553.0</td>
      <td>-99.0</td>
      <td>541.0</td>
      <td>1224.0</td>
      <td>1627.0</td>
      <td>1620.0</td>
      <td>1596.0</td>
      <td>1332.0</td>
      <td>940.0</td>
      <td>566.0</td>
      <td>-108.0</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>62.0</td>
      <td>-85.0</td>
      <td>55.0</td>
      <td>738.0</td>
      <td>1219.0</td>
      <td>1442.0</td>
      <td>1506.0</td>
      <td>1557.0</td>
      <td>1221.0</td>
      <td>788.0</td>
      <td>546.0</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>44.0</td>
      <td>-105.0</td>
      <td>38.0</td>
      <td>590.0</td>
      <td>987.0</td>
      <td>1500.0</td>
      <td>1487.0</td>
      <td>1477.0</td>
      <td>1377.0</td>
      <td>974.0</td>
      <td>31.0</td>
      <td>-178.0</td>
    </tr>
  </tbody>
</table>
</div>



Then, we use stack() method to rotate the axis of "value". A new column will be creadted. 


```python
temps = temps.stack() 
#note here we must have"=", stack() method does not change the origin
temps.head()
```




    ID           Year        
    ACW00011604  1961  VALUE1     -89.0
                       VALUE2     236.0
                       VALUE3     472.0
                       VALUE4     773.0
                       VALUE5    1128.0
    dtype: float64



And now we can use reset_index() method.


```python
temps= temps.reset_index()
temps
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
      <th>ID</th>
      <th>Year</th>
      <th>level_2</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE1</td>
      <td>-89.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE2</td>
      <td>236.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE3</td>
      <td>472.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE4</td>
      <td>773.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE5</td>
      <td>1128.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13992657</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE8</td>
      <td>1540.0</td>
    </tr>
    <tr>
      <th>13992658</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE9</td>
      <td>2040.0</td>
    </tr>
    <tr>
      <th>13992659</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE10</td>
      <td>2030.0</td>
    </tr>
    <tr>
      <th>13992660</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE11</td>
      <td>2130.0</td>
    </tr>
    <tr>
      <th>13992661</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE12</td>
      <td>2150.0</td>
    </tr>
  </tbody>
</table>
<p>13992662 rows × 4 columns</p>
</div>



Now it's time to relabel the ugly "lebel_0" and "0"


```python
temps=temps.rename(columns={"level_2":"Month",0:"temp"})
temps
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
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE1</td>
      <td>-89.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE2</td>
      <td>236.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE3</td>
      <td>472.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE4</td>
      <td>773.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE5</td>
      <td>1128.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13992657</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE8</td>
      <td>1540.0</td>
    </tr>
    <tr>
      <th>13992658</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE9</td>
      <td>2040.0</td>
    </tr>
    <tr>
      <th>13992659</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE10</td>
      <td>2030.0</td>
    </tr>
    <tr>
      <th>13992660</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE11</td>
      <td>2130.0</td>
    </tr>
    <tr>
      <th>13992661</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>VALUE12</td>
      <td>2150.0</td>
    </tr>
  </tbody>
</table>
<p>13992662 rows × 4 columns</p>
</div>



It looks much better. And we need to replace value12345... by 123456, which is true month 


```python
temps["Month"] = temps["Month"].str[5:].astype(int)
```


```python
temps
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
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-89.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>236.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>472.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>773.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>1128.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13992657</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>8</td>
      <td>1540.0</td>
    </tr>
    <tr>
      <th>13992658</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>9</td>
      <td>2040.0</td>
    </tr>
    <tr>
      <th>13992659</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>10</td>
      <td>2030.0</td>
    </tr>
    <tr>
      <th>13992660</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>11</td>
      <td>2130.0</td>
    </tr>
    <tr>
      <th>13992661</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>12</td>
      <td>2150.0</td>
    </tr>
  </tbody>
</table>
<p>13992662 rows × 4 columns</p>
</div>



And we need to make our temp more familiar by divide 100


```python
temps["temp"]  =temps["temp"] / 100
temps
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
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13992657</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>8</td>
      <td>15.40</td>
    </tr>
    <tr>
      <th>13992658</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>9</td>
      <td>20.40</td>
    </tr>
    <tr>
      <th>13992659</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>10</td>
      <td>20.30</td>
    </tr>
    <tr>
      <th>13992660</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>11</td>
      <td>21.30</td>
    </tr>
    <tr>
      <th>13992661</th>
      <td>ZIXLT622116</td>
      <td>1970</td>
      <td>12</td>
      <td>21.50</td>
    </tr>
  </tbody>
</table>
<p>13992662 rows × 4 columns</p>
</div>



Now we can import this table to our data base.


```python
import sqlite3 
# use module sqlite3 to help to create ,edit and query databases
```


```python
db=sqlite3.connect("climate-data.db")
#create an empty database called climate-data
```

Use .to_sql() method to write this table in the database (db here).  
About .to_sql, check https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.to_sql.html


```python
temps.to_sql("temps",db,if_exists="replace",index=False)
#Here we use if_exists="replace" since we write the whole table at once
```

Now we should import stations and countries


```python
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(url)
stations.head(5)
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
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



The data here is good enough to use, next we just write it into our database


```python
stations.to_sql("stations", db, if_exists = "replace", index = False)
```

Import countries, same as above.


```python
countries_url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries = pd.read_csv(countries_url)
countries.head(5)
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
      <th>FIPS 10-4</th>
      <th>ISO 3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries.to_sql("country", db, if_exists = "replace", index = False)
```

    C:\conda\lib\site-packages\pandas\core\generic.py:2779: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      sql.to_sql(
    

Now the first part is done, lets check what our database looks like.


```python
cursor = db.cursor()# cursor is used to interact woth the database and 
#will execute SQL commands.
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
for result in cursor.fetchall():
    print(result[0])
```

    CREATE TABLE "temps" (
    "ID" TEXT,
      "Year" INTEGER,
      "Month" INTEGER,
      "temp" REAL
    )
    CREATE TABLE "stations" (
    "ID" TEXT,
      "LATITUDE" REAL,
      "LONGITUDE" REAL,
      "STNELEV" REAL,
      "NAME" TEXT
    )
    CREATE TABLE "country" (
    "FIPS 10-4" TEXT,
      "ISO 3166" TEXT,
      "Name" TEXT
    )
    

After finishing construsting our database. It's needed to close the connection.


```python
db.close()
```

## II. Query Function

We could build a function that make it much easier to query some data we need, instead of using complex SQL commands each time.

The function accept 4 arguments as inputs (Country, year_begin, year_end, month)  
And should return a Pandas dataframe contains as follows:  
###### The station name.  
###### The latitude of the station.  
###### The longitude of the station.  
###### The name of the country in which the station is located.  
###### The year in which the reading was taken.  
###### The month in which the reading was taken.  
###### The average temperature at the specified station during the specified year and month.   

Let's try how we should get our ideal results in an example that input is country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1


```python
db=sqlite3.connect("climate-data.db") 
# First we need connect the database
cursor = db.cursor()
cmd = \
"""
SELECT "FIPS 10-4" FROM country where Name = "India";
"""
#quary what FIP 10-4 code is for India
cursor.execute(cmd)
result=cursor.fetchone()
result[0]

```




    'IN'



Next we should query the specific data from table temps and stations with condtions.


```python
cmd = \
"""
SELECT S.name,S.LATITUDE,S.LONGITUDE,T.year, T.month, T.temp
FROM temps T
LEFT JOIN stations S ON T.id = S.id
WHERE S.id LIKE 'IN%' AND T.year >=1980 AND T.year<=2020 AND T.month=1
"""
#Here we use LIKE operator in the WHERE sentance, it will match the pattern 
#that S.id begins with IN, which is the FIPS code of India
```

And we use .read_sql_query() to read the result in pandas


```python
result=pd.read_sql_query(cmd, db)
```


```python
result
db.close()
# Always close the connection when we finish.
```

Here we missed one column "Country", we could add it manually.


```python
result.insert(3,'Country',"India")
```


```python
result
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



Now this is what we want, writing the related function should be easy.


```python
def query_climate_database(country,year_begin,year_end,month):
    db=sqlite3.connect("climate-data.db")
    cursor = db.cursor()
    country_check=\
    f"""SELECT "FIPS 10-4" FROM country where Name ="{country}" """
    cursor.execute(country_check)
    result=cursor.fetchone()
    country_code=result[0]
    main_query= \
    f"""
    SELECT S.name,S.LATITUDE,S.LONGITUDE,T.year, T.month, T.temp
    FROM temps T
    LEFT JOIN stations S ON T.id = S.id
    WHERE S.id LIKE '{country_code}%' AND T.year >={year_begin} AND T.year<={year_end} AND T.month={month}
    """
    result=pd.read_sql_query(main_query, db)
    result.insert(3,'Country',country)
    db.close()
    return result
```

Done! We can try the same input to see if we can get the same output above


```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



## Data Visualization

First we need to import a package plotly


```python
from plotly import express as px
```

Again first try to figure it out without function. We can use query_climate_database() we write above to find data required


```python
temp_vs_data=query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
temp_vs_data
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



Now we filter out data that not satisfy the minimum required number of years of data.  
Assume the minimum required number=10


```python
min_obs=10
#Note here +1 since we need how many years, not the difference
def count_year(x):
    return len(x)
temp_vs_data["year_count"]=temp_vs_data.groupby(["NAME","LATITUDE"])["Year"].transform(count_year)
temp_vs_data
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
      <th>year_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
      <td>34</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 8 columns</p>
</div>



Now we filter out years_of_data<10


```python
temp_vs_data_2=temp_vs_data[temp_vs_data["year_count"]>=10]
```


```python
temp_vs_data_2
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
      <th>year_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
      <td>34</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>SHILONG</td>
      <td>25.600</td>
      <td>91.890</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>10.40</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>SHILONG</td>
      <td>25.600</td>
      <td>91.890</td>
      <td>India</td>
      <td>1990</td>
      <td>1</td>
      <td>11.20</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>SHILONG</td>
      <td>25.600</td>
      <td>91.890</td>
      <td>India</td>
      <td>2010</td>
      <td>1</td>
      <td>11.99</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>SHILONG</td>
      <td>25.600</td>
      <td>91.890</td>
      <td>India</td>
      <td>2011</td>
      <td>1</td>
      <td>9.93</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>SHILONG</td>
      <td>25.600</td>
      <td>91.890</td>
      <td>India</td>
      <td>2012</td>
      <td>1</td>
      <td>9.68</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>3106 rows × 8 columns</p>
</div>



Now we compute the year-over-year average change in temperature by finding the cofficient of linear regression


```python
from sklearn.linear_model import LinearRegression
```


```python
def coef(data_group):
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]
```


```python
coefs = temp_vs_data_2.groupby(["NAME"]).apply(coef)
```


```python
coefs=coefs.reset_index()
```


```python
coefs
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
      <th>NAME</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGARTALA</td>
      <td>-0.006184</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AHMADABAD</td>
      <td>0.006731</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AKOLA</td>
      <td>-0.008063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ALLAHABAD</td>
      <td>-0.029375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ALLAHABAD_BAMHRAULI</td>
      <td>-0.015457</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>TRIVANDRUM</td>
      <td>0.022892</td>
    </tr>
    <tr>
      <th>93</th>
      <td>UDAIPUR_DABOK</td>
      <td>0.072424</td>
    </tr>
    <tr>
      <th>94</th>
      <td>VARANASI_BABATPUR</td>
      <td>-0.012996</td>
    </tr>
    <tr>
      <th>95</th>
      <td>VERAVAL</td>
      <td>0.024848</td>
    </tr>
    <tr>
      <th>96</th>
      <td>VISHAKHAPATNAM</td>
      <td>-0.034050</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 2 columns</p>
</div>



Note here we lose LATITUDE and LONGITUDE, and we can use merge to restore them


```python
final=pd.merge(coefs,temp_vs_data_2,how='inner',on = ["NAME"]).drop_duplicates('NAME')
```


```python
final=final.rename(columns={ 0 :"change"})
final["change"]=round(final["change"],5)
final
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
      <th>NAME</th>
      <th>change</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
      <th>year_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGARTALA</td>
      <td>-0.00618</td>
      <td>23.883</td>
      <td>91.250</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>18.21</td>
      <td>33</td>
    </tr>
    <tr>
      <th>33</th>
      <td>AHMADABAD</td>
      <td>0.00673</td>
      <td>23.067</td>
      <td>72.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>20.39</td>
      <td>38</td>
    </tr>
    <tr>
      <th>71</th>
      <td>AKOLA</td>
      <td>-0.00806</td>
      <td>20.700</td>
      <td>77.033</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>22.47</td>
      <td>60</td>
    </tr>
    <tr>
      <th>131</th>
      <td>ALLAHABAD</td>
      <td>-0.02938</td>
      <td>25.441</td>
      <td>81.735</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>17.47</td>
      <td>27</td>
    </tr>
    <tr>
      <th>158</th>
      <td>ALLAHABAD_BAMHRAULI</td>
      <td>-0.01546</td>
      <td>25.500</td>
      <td>81.900</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>15.42</td>
      <td>26</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2981</th>
      <td>TRIVANDRUM</td>
      <td>0.02289</td>
      <td>8.500</td>
      <td>77.000</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>27.10</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3006</th>
      <td>UDAIPUR_DABOK</td>
      <td>0.07242</td>
      <td>24.617</td>
      <td>73.883</td>
      <td>India</td>
      <td>2010</td>
      <td>1</td>
      <td>16.84</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>VARANASI_BABATPUR</td>
      <td>-0.01300</td>
      <td>25.450</td>
      <td>82.867</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>15.83</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3038</th>
      <td>VERAVAL</td>
      <td>0.02485</td>
      <td>20.900</td>
      <td>70.367</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>21.45</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3079</th>
      <td>VISHAKHAPATNAM</td>
      <td>-0.03405</td>
      <td>17.717</td>
      <td>83.233</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>25.44</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 9 columns</p>
</div>



Now trying to visualize our data.


```python
color_map = px.colors.diverging.RdGy_r
fig = px.scatter_mapbox(final, 
                        lat = "LATITUDE",
                        lon = "LONGITUDE", 
                        hover_name = "NAME", 
                        color = "change",
                        zoom = 2,
                        #opacity = 0.2,
                        height = 300,
                        mapbox_style="carto-positron",
                        color_continuous_scale=color_map)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_coloraxes(cmid=0)
fig.update_coloraxes(colorbar_title_text="Avg Yearly Increase(C)",colorbar_title_font_size=10)
fig.show()
```

Now we should able to finish the function.

We need month_name to replace the month number, so we write a function


```python
def month_name (number):
    if number == 1:
        return "January"
    elif number == 2:
        return "February"
    elif number == 3:
        return "March"
    elif number == 4:
        return "April"
    elif number == 5:
        return "May"
    elif number == 6:
        return "June"
    elif number == 7:
        return "July"
    elif number == 8:
        return "August"
    elif number == 9:
        return "September"
    elif number == 10:
        return "October"
    elif number == 11:
        return "November"
    elif number == 12:
        return "December"
```


```python
def temperature_coefficient_plot(country,year_begin,year_end,month,min_obs,**kwargs):
    temp_vs_data=query_climate_database(country,year_begin,year_end,month)
    temp_vs_data["year_count"]=temp_vs_data.groupby(["NAME"])["Month"].transform(count_year)
    temp_vs_data_2=temp_vs_data[temp_vs_data["year_count"]>=min_obs]
    coefs = temp_vs_data_2.groupby(["NAME"]).apply(coef)
    coefs=coefs.reset_index()
    final=pd.merge(coefs,temp_vs_data_2,how='inner',on = ["NAME"]).drop_duplicates('NAME')
    final=final.rename(columns={ 0 :"change"})
    final["change"]=round(final["change"],5)
    #Here round to 5 digits
    fig=px.scatter_mapbox(final, 
                        lat = "LATITUDE",
                        lon = "LONGITUDE", 
                        hover_name = "NAME", 
                        color = "change",
                        height = 300,
                        title="Estimates of yearly increase in temperature in " +month_name(month)+" for stations in "+str(country) +", years "+str(year_begin)+"-"+str(year_end),
                        **kwargs)
    fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
    fig.update_coloraxes(cmid=0)
    fig.update_coloraxes(colorbar_title_text="Avg Yearly Increase(C)",colorbar_title_font_size=10)
    
    return fig
```


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()
```


## extra Question 1: Is there any relations about climate change and LATITUDE given month and year range?

First we write a function to query data with input latitude range (min, low)


```python
def query_climate_database_by_lat(latmin,latmax,year_begin,year_end,month):
    db=sqlite3.connect("climate-data.db")
    cursor = db.cursor()
    main_query= \
    f"""
    SELECT S.name,S.LATITUDE,T.year, T.month, T.temp
    FROM temps T
    LEFT JOIN stations S ON T.id = S.id
    WHERE S.LATITUDE >={latmin} AND S.LATITUDE<{latmax} AND T.year >={year_begin} AND T.year<={year_end} AND T.month={month}
    """
    result=pd.read_sql_query(main_query, db)
    db.close()
    return result
```


```python
result=query_climate_database_by_lat(0,10,2000,2010,8)
#return the dataframe of given latitude range
```


```python
result
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VICTORIA_POINT</td>
      <td>9.967</td>
      <td>2001</td>
      <td>8</td>
      <td>24.42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VICTORIA_POINT</td>
      <td>9.967</td>
      <td>2002</td>
      <td>8</td>
      <td>23.82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2000</td>
      <td>8</td>
      <td>24.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2001</td>
      <td>8</td>
      <td>24.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2002</td>
      <td>8</td>
      <td>25.10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2001</td>
      <td>8</td>
      <td>28.26</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2003</td>
      <td>8</td>
      <td>28.54</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2004</td>
      <td>8</td>
      <td>28.12</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2008</td>
      <td>8</td>
      <td>28.16</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2010</td>
      <td>8</td>
      <td>28.04</td>
    </tr>
  </tbody>
</table>
<p>1926 rows × 5 columns</p>
</div>



We also want to filter out with min year condition


```python
result["years_of_data"]=result.groupby(["NAME","LATITUDE"])['Month'].transform(count_year)
```


```python
result=result[result["years_of_data"]>=5]
result
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
      <th>years_of_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2000</td>
      <td>8</td>
      <td>24.50</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2001</td>
      <td>8</td>
      <td>24.40</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2002</td>
      <td>8</td>
      <td>25.10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2003</td>
      <td>8</td>
      <td>25.20</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SAVE</td>
      <td>7.980</td>
      <td>2004</td>
      <td>8</td>
      <td>24.90</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2001</td>
      <td>8</td>
      <td>28.26</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2003</td>
      <td>8</td>
      <td>28.54</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2004</td>
      <td>8</td>
      <td>28.12</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2008</td>
      <td>8</td>
      <td>28.16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>CON_SON</td>
      <td>8.683</td>
      <td>2010</td>
      <td>8</td>
      <td>28.04</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>1790 rows × 6 columns</p>
</div>



And we calculte the yearly change within this latitute range by stations


```python
coefs2 = result.groupby(["NAME"]).apply(coef)
```


```python
coefs2=coefs2.reset_index()
```


```python
final=pd.merge(coefs2,result,how='inner',on = ["NAME"]).drop_duplicates('NAME')
final=final.rename(columns={ 0 :"change"})
final
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
      <th>NAME</th>
      <th>change</th>
      <th>LATITUDE</th>
      <th>Year</th>
      <th>Month</th>
      <th>temp</th>
      <th>years_of_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ABIDJAN_FELIX_HOUPHOUET_BOIGN</td>
      <td>0.084636</td>
      <td>5.2610</td>
      <td>2000</td>
      <td>8</td>
      <td>24.48</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ACCRA</td>
      <td>0.072757</td>
      <td>5.5500</td>
      <td>2003</td>
      <td>8</td>
      <td>25.60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ADDIS_ABABA</td>
      <td>0.026324</td>
      <td>9.0000</td>
      <td>2000</td>
      <td>8</td>
      <td>15.60</td>
      <td>9</td>
    </tr>
    <tr>
      <th>35</th>
      <td>ADDIS_ABABA_BOLE</td>
      <td>-0.019069</td>
      <td>9.0330</td>
      <td>2000</td>
      <td>8</td>
      <td>15.65</td>
      <td>10</td>
    </tr>
    <tr>
      <th>45</th>
      <td>ADIAKE</td>
      <td>0.117674</td>
      <td>5.3000</td>
      <td>2005</td>
      <td>8</td>
      <td>24.44</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1746</th>
      <td>WAU</td>
      <td>0.146216</td>
      <td>7.7000</td>
      <td>2000</td>
      <td>8</td>
      <td>25.75</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1752</th>
      <td>WEATHER_SERVICE_OFFICEPOHNPE</td>
      <td>0.047500</td>
      <td>6.9500</td>
      <td>2000</td>
      <td>8</td>
      <td>26.60</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1761</th>
      <td>YAP_ISLAND_WSO_AP</td>
      <td>0.030390</td>
      <td>9.4833</td>
      <td>2000</td>
      <td>8</td>
      <td>27.23</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1771</th>
      <td>YARIGUIES</td>
      <td>-0.040322</td>
      <td>7.0240</td>
      <td>2000</td>
      <td>8</td>
      <td>27.65</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1781</th>
      <td>ZAMBOANGA</td>
      <td>0.017667</td>
      <td>6.9000</td>
      <td>2000</td>
      <td>8</td>
      <td>27.71</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 7 columns</p>
</div>



Then we get the average temperature change over year bewtween latitude 0-10, year 2000-2010


```python
final["change"].mean()
```




    0.025535073308215226



Now lets write the function.  
The function below will return the average temperature change given the latitude range and year range in a certain month


```python
def avg_temp_change_latitude(latmin,latmax,year_begin,year_end,month,min_obs):
    result=query_climate_database_by_lat(latmin,latmax,year_begin,year_end,month)
    result["years_of_data"]=result.groupby(["NAME","LATITUDE"])['Year'].transform(count_year)
    result=result[result["years_of_data"]>=min_obs]
    if len(result)==0:
        return np.nan
    coefs2 = result.groupby(["NAME"]).apply(coef)
    coefs2=coefs2.reset_index()
    final=pd.merge(coefs2,result,how='inner',on = ["NAME"]).drop_duplicates('NAME')
    final=final.rename(columns={ 0 :"change"})
    return final["change"].mean()
```


```python
a=avg_temp_change_latitude(84,90,2000,2010,8,5)
a
#This shows that temperature in latitude(-10,10) is moving 0.02 C higher in average during year 2000-2010 in Aug
```




    nan



And we write a function that will return a dataframe that contains two column [latitude] and [temps_change] over the earth(-90,90) and we can choose how many intevals it have. And the latitude will be the mid point of the latitude 


```python
def latitude_temps_change(year_begin,year_end,month,min_obs,inteval_number):
    k=pd.DataFrame(columns=['latitude', 'temps_change'])
    for i in range(inteval_number):
        inteval=180/inteval_number
        l_min=-90+inteval*(i)
        l_max=l_min+inteval
        l_mid=(l_min+l_max)/2
        avg_change=avg_temp_change_latitude(l_min,l_max,year_begin,year_end,month,min_obs)
        k.loc[i]={'latitude':l_mid,'temps_change': avg_change}
    return k
```


```python
m=latitude_temps_change(2000,2010,8,5,30)
m
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
      <th>latitude</th>
      <th>temps_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-87.0</td>
      <td>0.242526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-81.0</td>
      <td>0.206937</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-75.0</td>
      <td>0.117477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-69.0</td>
      <td>-0.012480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-63.0</td>
      <td>-0.121091</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-57.0</td>
      <td>-0.011684</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-51.0</td>
      <td>-0.027366</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-45.0</td>
      <td>-0.027148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-39.0</td>
      <td>-0.000294</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-33.0</td>
      <td>0.003428</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-27.0</td>
      <td>0.016539</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-21.0</td>
      <td>0.008471</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-15.0</td>
      <td>0.036815</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-9.0</td>
      <td>0.012858</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-3.0</td>
      <td>0.037993</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.0</td>
      <td>0.023535</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9.0</td>
      <td>0.025337</td>
    </tr>
    <tr>
      <th>17</th>
      <td>15.0</td>
      <td>0.010469</td>
    </tr>
    <tr>
      <th>18</th>
      <td>21.0</td>
      <td>0.003607</td>
    </tr>
    <tr>
      <th>19</th>
      <td>27.0</td>
      <td>0.066296</td>
    </tr>
    <tr>
      <th>20</th>
      <td>33.0</td>
      <td>0.039407</td>
    </tr>
    <tr>
      <th>21</th>
      <td>39.0</td>
      <td>-0.032260</td>
    </tr>
    <tr>
      <th>22</th>
      <td>45.0</td>
      <td>-0.073466</td>
    </tr>
    <tr>
      <th>23</th>
      <td>51.0</td>
      <td>-0.078757</td>
    </tr>
    <tr>
      <th>24</th>
      <td>57.0</td>
      <td>0.007513</td>
    </tr>
    <tr>
      <th>25</th>
      <td>63.0</td>
      <td>0.034609</td>
    </tr>
    <tr>
      <th>26</th>
      <td>69.0</td>
      <td>0.017199</td>
    </tr>
    <tr>
      <th>27</th>
      <td>75.0</td>
      <td>0.171940</td>
    </tr>
    <tr>
      <th>28</th>
      <td>81.0</td>
      <td>0.013791</td>
    </tr>
    <tr>
      <th>29</th>
      <td>87.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.relplot(data = m, 
            x = "temps_change", 
            y = "latitude",
            alpha = 1, 
            height = 4,
            aspect = 1.7)
plt.plot([0,0], [-90,90], color = "lightgray", zorder = 0)
plt.gca().set(title = "Yearly change in temperature by Latitude, 2010-2020 in Aug")
sns.despine()
```


    
![png](output_100_0.png)
    


#### As we see in this plot, the temperature's change is bigger when in high latitude, which is bad for the polar regions.  
#### And most people are living within the latitude (-60,60), we may underestimating such climate change.
#### But the glaciers are melting faster than we think.

Now let me write the function for auto plotting


```python
def latitude_temp_change_plot(year_begin,year_end,month,min_obs,inteval_number):
    data_set=latitude_temps_change(year_begin,year_end,month,min_obs,inteval_number)
    sns.relplot(data = data_set, 
            x = "temps_change", 
            y = "latitude",
            alpha = 1, 
            height = 4,
            aspect = 1.7)
    plt.plot([0,0], [-90,90], color = "lightgray", zorder = 0)
    plt.gca().set(title = f"Yearly change in temperature by Latitude between {year_begin}-{year_end} in {month_name(month)}")
    sns.despine()
```

Try if the function is good.


```python
latitude_temp_change_plot(2010,2020,8,5,30)
```


    
![png](output_105_0.png)
    


### extra Question 2:  How does climate change differ between countries?


```python
from urllib.request import urlopen
import json

countries_gj_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/countries.geojson"

with urlopen(countries_gj_url) as response:
    countries_gj = json.load(response)
```

First we write a function return the yearly change by temprature given country, time duration, and month


```python
def yearly_change_by_country(country,year_begin,year_end,month):
    db=sqlite3.connect("climate-data.db")
    cursor = db.cursor()
    country_check=\
    f"""SELECT "FIPS 10-4" FROM country where Name ="{country}" """
    cursor.execute(country_check)
    result=cursor.fetchone()
    country_code=result[0]
    main_query= \
    f"""
    SELECT S.name,S.LATITUDE,S.LONGITUDE,T.year, T.month, T.temp
    FROM temps T
    LEFT JOIN stations S ON T.id = S.id
    WHERE S.id LIKE '{country_code}%' AND T.year >={year_begin} AND T.year<={year_end} AND T.month={month}
    """
    result=pd.read_sql_query(main_query, db)
    result.insert(3,'Country',country)
    coefs = result.groupby(["Country"]).apply(coef)
    coefs=coefs.reset_index()
    coefs=coefs.rename(columns={ 0 :"change"})
    return coefs
```


```python
p=yearly_change_by_country(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
k=yearly_change_by_country(country = "China", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```


```python
s=pd.DataFrame(columns=['Country', 'change'])
s
s=s.append(k)
print(s)
```

      Country    change
    0   China -0.007466
    


```python
def global_yearly_change(year_begin,year_end,month):
    k=pd.DataFrame(columns=['Country', 'change'])
    for Name in countries["Name"]:
        k=k.append(yearly_change_by_country(Name,year_begin,year_end,month))
    return k
    
```


```python
change_2015_2020_Jan=global_yearly_change(2015,2020,1)
change_2015_2020_Jan
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
      <th>Country</th>
      <th>change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>-0.358605</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Albania</td>
      <td>-0.127429</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Algeria</td>
      <td>-0.026098</td>
    </tr>
    <tr>
      <th>0</th>
      <td>American Samoa</td>
      <td>-0.025714</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Angola</td>
      <td>-0.100000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Wake Island</td>
      <td>0.047297</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Wallis and Futuna</td>
      <td>-0.125516</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Western Sahara</td>
      <td>0.164940</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Zambia</td>
      <td>-1.018750</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Zimbabwe</td>
      <td>-0.205380</td>
    </tr>
  </tbody>
</table>
<p>202 rows × 2 columns</p>
</div>




```python
fig = px.choropleth(change_2015_2020_Jan, 
                    geojson=countries_gj,
                    locations = "Country",
                    locationmode = "country names",
                    color = "change", 
                    height = 300)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
```


This plot is showing that not all country are getting warmer,some even getting much colder in a very fast speed.  
And the situation is consistent with the first extra question, the climate change in polar area are stronger than equatorial region.

And below is the function for the plot easy for use


```python
def change_by_country_plot(year_begin,year_end,month):
    result=global_yearly_change(year_begin,year_end,month)
    fig = px.choropleth(result, 
                    geojson=countries_gj,
                    locations = "Country",
                    locationmode = "country names",
                    color = "change", 
                    height = 300)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
```
