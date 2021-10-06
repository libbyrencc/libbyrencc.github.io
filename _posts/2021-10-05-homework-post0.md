---
layout: post
title: Post 0
---

## Trying to the find the relationship between body mass and the area of Culmen by  visualization

Use **pandas** to deal with data

```python
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```
Read 5 lines

```python
penguins.head(5)
```

###### Notice that there are penguins with **NaN** in this set, we need to drop that by the dropna() method.
```python
penguins=penguins.dropna(subset=['Culmen Length (mm)','Culmen Depth (mm)','Body Mass (g)'])
penguins.head(5)
```
Calculate the area of penguins' culmen  
    Assume penguins' culmens is similar to triangles, ***A=Length\*Depth/2***

```python
penguins['Culmen Area (mm^2)']=penguins['Culmen Length (mm)']*penguins['Culmen Depth (mm)']/2
penguins.head(5)
```
### Plotting

Then we need to visualize our data by using ***seaborn*** and ***matplotlib***, which are great libraries.
```python
import seaborn as sns
from matplotlib import pyplot as plt
```

Plot data

```python
sns.scatterplot(data = penguins, #
                x = "Culmen Area (mm^2)", #
                y = "Body Mass (g)", 
                hue = "Species",
                )
plt.legend(bbox_to_anchor=(1.05, 1),loc=2)
sns.despine()
```

It seems that the Culmen Area and Body Mass are linear dependence in each specie.