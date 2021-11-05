---
layout: post
title: Post 4
---

# Spectral Clustering


## Introduction

Spectral clustering is an algorithm used to deal with some complex datas.   
It is usefull when sometimes the K-means method does not work properly.


```python
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
```

For example in the following data, which shape is two curves. The Euclidean coordinates of data points is in the matrix `X`, and the labels of points is in the `y`.


```python
np.random.seed(1234)
n = 200
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x22ea1b070d0>




    
![png](output_3_1.png)
    


And if we use K-means method: 


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x22ea17851f0>




    
![png](output_5_1.png)
    


We notice that this K-means method cannot handle this kind of data very well.  
We expect the two curve can be seperated into different groups like following:

![q2](q2.png)

How we can achieve this? You can read the following instructions:

## Part A

First we need to construct the *similarity matrix*  $$\mathbf{A}$$ with shape `(n, n)` by using numpy 2d array.

To construct the similarity matrix, first we need to compute the *distance matrix* $$\mathbf{dis M}$$ .Then we will use a parameter `epsilon` to give the weight to each distance.`A[i,j]` should be equal to `1` if `X[i]` is within distance `epsilon` of `X[j]`,and `0` otherwise. It means that when elements in $$\mathbf{dis M}$$ < `epsilon`, then = 1, otherwise = 1

For this part, we use `epsilon = 0.4`. 


```python
import sklearn
dis_M=sklearn.metrics.pairwise_distances(X) 
dis_M # compute the distance matrix for X
```




    array([[0.        , 1.27292462, 1.33315598, ..., 1.9812102 , 1.68337039,
            1.94073324],
           [1.27292462, 0.        , 1.46325112, ..., 1.93729167, 1.68543003,
            1.91287315],
           [1.33315598, 1.46325112, 0.        , ..., 0.64857172, 0.35035968,
            0.60860868],
           ...,
           [1.9812102 , 1.93729167, 0.64857172, ..., 0.        , 0.30070415,
            0.04219636],
           [1.68337039, 1.68543003, 0.35035968, ..., 0.30070415, 0.        ,
            0.26255757],
           [1.94073324, 1.91287315, 0.60860868, ..., 0.04219636, 0.26255757,
            0.        ]])




```python
A = np.zeros_like(dis_M) #create a matrix where all 0s
musk=dis_M<0.4 #create a musk to find in the dis_M where dis<0.4
A[musk]=1 #let number = 1 where dis <0.4
np.fill_diagonal(A,0) # diagonal entries should = 0
A
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.],
           ...,
           [0., 0., 0., ..., 0., 1., 1.],
           [0., 0., 1., ..., 1., 0., 1.],
           [0., 0., 0., ..., 1., 1., 0.]])



## Part B

And then we need to calculate the *binary norm cut objective* 

$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;.$$

Such that, 
- $$\mathbf{cut}(C_0, C_1) \equiv \sum_{i \in C_0, j \in C_1} a_{ij}$$ is the *cut* of the clusters $$C_0$$ and $$C_1$$. 
- $$\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$$, where $$d_i = \sum_{j = 1}^n a_{ij}$$ is the *degree* of row $$i$$ ($$d_i = \sum_{j = 1}^n a_{ij}$$). The *volume* of cluster $$C_0$$ is a measure of the size of the cluster. 

#### B.1 The Cut Term

The $$\mathbf{cut}$$ function is used to measure the similarity between clusters.  
And the following function is to calculate the cut if given how to seperate datas using `y` array.


```python
def cut(A,y):
    cut_sum=0
    for i in range(200):
        for j in range(200):
            if y[i]==0 and y[j]==1:
                cut_sum=cut_sum+A[i][j]
    return cut_sum
```

The following code calculate the $$\mathbf{cut}$$ number using `y` we "assumed above" (This actually is aleady an optimal solution).  
And we compare it with the random selected cluster labels `randomLabel`.
Notice that the cut we use `y` is much smaller than the randomly selected one.


```python
cut_term=cut(A,y)
print("cut term _true = ",cut_term)
randomLabel = np.random.randint(2, size=200)
cut_term_rand=cut(A,randomLabel)
print("cut term _rand = ",cut_term_rand)
```

    cut term _true =  13.0
    cut term _rand =  1150.0
    

#### B.2 The Volume Term 

The $$\mathbf{vol}$$ is used to measure the similarity within a cluster.   

$$\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$$, where $$d_i = \sum_{j = 1}^n a_{ij}$$ is the *degree* of row $$i$$ ($$d_i = \sum_{j = 1}^n a_{ij}$$)  



```python
def vols(A,y):
    deg=A.sum(axis=1)
    v0=deg[y==0].sum()
    v1=deg[y==1].sum()
    return (v0,v1)
```

And now we can write the final normcut function :


```python
def normcut(A,y):
    v0,v1=vols(A,y)
    cut_C=cut(A,y)
    return cut_C*(1/v0+1/v1)
```

If we compare the normcut using `y` (optimal) and using `randomLabel`. We could see a huge difference!


```python
normcut_true=normcut(A,y)
print("normcut_true = ", normcut_true)
normcut_rand=normcut(A,randomLabel)
print("normcut_rand = ", normcut_rand)

#Huge difference! Where normcut_rand is about 1
```

    normcut_true =  0.011518412331615225
    normcut_rand =  1.0240023597759158
    

## Part C

And there's another way to calculate the `normcut` by using the following formula

We define a new vector $$\mathbf{z}$$ : 
$$
z_i = 
\begin{cases}
    \frac{1}{\mathbf{vol}(C_0)} &\quad \text{if } y_i = 0 \\ 
    -\frac{1}{\mathbf{vol}(C_1)} &\quad \text{if } y_i = 1 \\ 
\end{cases}
$$

And the `new normcut` is :

$$\mathbf{N}_{\mathbf{A}}(C_0, C_1) = \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}}\;,$$

where $$\mathbf{D}$$ is the diagonal matrix with nonzero entries $$d_{ii} = d_i$$, and  where $$d_i = \sum_{j = 1}^n a_i$$ is the degree (row-sum) from before. 

And we write the following code to form the new vector z


```python
def transform(A,y):
    z=np.zeros(len(y))
    v0,v1=vols(A,y)
    z[y==0]=1/v0
    z[y==1]=-1/v1
    return z
```

And we check if the `new normcut` is equal `normcut` we did before


```python
row_sum=A.sum(axis=1)
D=np.zeros_like(A)
np.fill_diagonal(D,row_sum)
z=transform(A,y)
Na=(z.T@(D-A)@z)/(z.T@D@z)
np.isclose(Na,normcut_true)
```




    True



## Part D

All the above is when we have the solution `y`.      


However, in real world, we need to calculate the "y", which is meant to minimize the following function, and the solution related to such Min is what we want. 

$$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$

such that $$\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$


```python
def orth(u, v):
    return (u @ v) / (v @ v) * v

e = np.ones(n) 

d = D @ e

def orth_obj(z):
    z_o = z - orth(z, d)
    return (z_o.T @ (D - A) @ z_o)/(z_o.T @ D @ z_o)
```

And we use `scipy.optimize.minimize` function to minimize our function above, note that we need to speicify `method = 'Nelder-Mead'`, or it will not success.

And we will have the "`y`", here is z_min


```python
import scipy
res=scipy.optimize.minimize(orth_obj,z,method = 'Nelder-Mead')
z_min=res.x
```


```python
res.success
```




    True



## Part E

The sign of `z_min[k]` indicates the point is belong to which cluster, and if we plot the data. We expect the correct graph.


```python
z_min[z_min>=0]=1
z_min[z_min<0]=0
plt.scatter(X[:,0], X[:,1], c = z_min)
```




    <matplotlib.collections.PathCollection at 0x22ea1c5ff40>




    
![png](output_33_1.png)
    


## Part F

And the above method could be slow when the condition is bad, and we have a better method to minimize!

According to the math in the written assignment, minimize the function above is equivalent to find the eigenvector of second-smallest eigenvalue of 

matrix $$\mathbf{L} = \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A})$$, which is called the (normalized) *Laplacian* matrix of the similarity matrix $$\mathbf{A}$$


```python
L=np.linalg.inv(D)@(D-A)  #Laplacian matrix
Lam, U = np.linalg.eig(L) #eig value and eig vector
ix = Lam.argsort()
Lam, U = Lam[ix], U[:,ix]
z_eig=U[:,1] #2nd eig vector
```

And if we plot the 


```python
res=np.zeros_like(z_eig)
res[z_eig>=0]=1
plt.scatter(X[:,0], X[:,1], c = res)
```




    <matplotlib.collections.PathCollection at 0x22ea2c880d0>




    
![png](output_37_1.png)
    


Another great graph!

## Part G
 
And now we should able to write the whole function to find the solution to `spectral clustering`, which will give labels.


```python
def sml_m(X,epsilon):
    """
    Return the similarity matrix.
    
    Keyword arguments:
    
    X -- the Euclidean coordinates of data points
    
    epsilon -- the distance threshold to perform spectral clustering
    
    """
    dis_M=sklearn.metrics.pairwise_distances(X)
    A = np.zeros_like(dis_M) 
    A[dis_M<epsilon]=1
    np.fill_diagonal(A,0)
    return A

def L_m(A):
    """
    Return the Laplacian matrix.
    
    Keyword arguments:
    
    A -- the similarity matrix
    
    """
    row_sum=A.sum(axis=1)
    D=np.zeros_like(A)
    np.fill_diagonal(D,row_sum)
    L=np.linalg.inv(D)@(D-A)
    return L

def eig(L):
    """
    Return the eigenvector with second-smallest eigenvalue of the Laplacian matrix
    
    Keyword arguments:
    
    L -- the Laplacian matrix
    
    """
    Lam, U = np.linalg.eig(L)
    ix = Lam.argsort()
    Lam, U = Lam[ix], U[:,ix]
    z_eig=U[:,1]
    return z_eig
```

Final function :


```python
def spectral_clustering(X, epsilon):
    """
    Return the binary label of each point, the point will either in group 0 or 1
    
    Keyword arguments:
    
    X -- the Euclidean coordinates of data points
    
    epsilon -- the distance threshold to perform spectral clustering
    
    """
    A=sml_m(X,epsilon)
    L=L_m(A)
    z_eig=eig(L)
    #Return labels based on this eigenvector.
    res=np.zeros_like(z_eig)
    res[z_eig>=0]=1
    return res
```

## Part H

Try different data sets using `make_moons`.  
Notice that if we increase the `noice`, the points are getting more dispersed.   
So we need to narrow our `epsilon` to get a better result.


```python
X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X,0.4))

```




    <matplotlib.collections.PathCollection at 0x22ea2f7e6d0>




    
![png](output_44_1.png)
    



```python
X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X,0.4))
```




    <matplotlib.collections.PathCollection at 0x22ea2f18ac0>




    
![png](output_45_1.png)
    



```python
X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X,0.28))
```




    <matplotlib.collections.PathCollection at 0x22ea309e460>




    
![png](output_46_1.png)
    


## Part I

Try our spectral clustering function on other graph


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x22ea2e7e2e0>




    
![png](output_48_1.png)
    


K-means method fails again here.


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x22ea2ec4d90>




    
![png](output_50_1.png)
    


Spectral clustering does a very good job!


```python
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X,0.5))
```




    <matplotlib.collections.PathCollection at 0x22ea16cad60>




    
![png](output_52_1.png)
    