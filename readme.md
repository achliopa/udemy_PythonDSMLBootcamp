# Udemy Course: Python for Data Science and Machine Learning Bootcamp

* [Course Link](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview)

## Section 1 - Course Introduction

### Lecture 1 - Introduction to the Course

* Course Includes:
	* NumPy
	* SciPy
	* Pandas
	* SeaBorn
	* SciKit-Learn
	* MatPlotLib
	* Plotly
	* PySpark
* Tensorflow Intro

## Section 2 - Environment Setup

### Lecture 4 - Environment Setup and Installation

* we use Jupyter Notebooks (have it in Anaconda)
* we use Python3 (have it with Anaconda)
* install jupyter with python3 `pip3 install jupyter`
* [Jupyter](www.jupyter.org) to install. jupyter works for >40 pls
* we can use an online version
* we install Anaconda

## Section 3 - Jupyter Overview

### Lecture 5 - Jupyter Notebooks

* start notebooks from terminal . `jupyter notebook` in the folder where you want it to run
* we navigate to the course repo
* For tutorial see the PythonBootcamp for how to
* we can download jupyter files in other formats

### Lecture 6 - Optional: Virtual Environments

* virtual environments allow setting up virtual installations of python and libraries on our computer
* we can have multiple versions of python or libs and activate or deactivate these environments
* handy if we want to program in different versions of the library
	* e.g we develop with SciKit v0.17. v0.18 is out. we want to experiment with it but dont want to cause code pbreaks in old code
* sometimes we want our library installations in the correct location (py2.7 py3.5)
* there is *virtualenv* lib for nomal python distributions
* anaconda has inbuilt virtual environment manager [link](https://conda.pydata.org/docs/using/envs.html)
* we create a virtual environment with conda named fluffy setting the lib we want to use in it `conda create --name fluffy numpy`
* to activate thsi environment with `activate fluffy` and deactivate it with `deactivate fluffy`
* if we write `python` we get the normal version installed with conda or directly
* in the python cli we can import the libs manually `import numpy as np` `import pandas as pd`
* we quit cli with `quit()`
* we activate our vitrualenv with `activate fluffy` env is activated. if we invloke python `python` we get an old version as per lib we speced. this env has no pandas so we cannot import them we have to install them (while the env is activated) in the given environment
* we can set an env for a gibven python version (for testing) `conda create --name mypython3version python=3.5 numpy`
* we can list our envs with `conda info --envs`

## Section 4 - Python Crash Course

Going to skim through this as Python Bootcamp is a complete Python Course

## Section 5 - Python for Data Analysis - NymPy

### Lecture 16 - Introduction to NumPy

* NumPy is a Linear Algebra Lib for Python
* It is very important for Data Science wwith Python
* Almost all of the libraries in the PyData Ecosystem rely on NumPy as one of their main builing blocks
* NumPy is blazing fast as it binds to C libraries
* It si recommended to install Python with Anaconda to make sure all dependencies are in sync with the conda install
* If we have Anaconda to install NumPy we use `conda install numpy` or `pip install numpy`
* NumPy arrays are the main way we use Numpy in the course
* Numpy arrays come in two flavors: vectors and matrices
	* Vectors are 1-D arrays
	* Matrices are 2-D (also can be 1 D with 1 row or ! column)

### Lecture 17 - NumPy Arrays

* we have a list `my_list = [1,2,3]`
* we import numpy `import numpy as np`
* we use numpy array wrapper method to cast the list as an array `np.array(my_list)` we get an array back `array([1,2,3])` this is an 1D array
* if i have a python nested list *list-of-lists* and use the numpy array wrapper it gets cast to a Matrix or 2D array
```
np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> 	array([[1,2,3],
			   [4,5,6],
			   [7,8,9]])
```
* we can use *np.arange(start,stop)* to generate arrays from a range. `np.arange(0,10)` => `array([0,1,2,3,4,5,6,7,8,9])`
* we can pass a 3rd argument the stpsize *np.arange(start,stop,stepsize)* to jump numbers in the range
* if we want to generate fixed num arrays we can use `np.zeros(3)` passing a single num to get 1-D arrays `array([0,0,0])` or passing tuples to get multidimension matrix arrays `np.zeros((5,5))` where the tuple element is the size (1st number rows 2nd number columns)
* the number is writen in descriptive form . e.g ones twos, threes
* the method *linspace* makes an array (1D vector) distributing evenly the range based on the speced num of points `np.linspace(start,end,numofpoints)` e.g `np.linspace(0,5,10)` => `array([0. , 0.555556, 1.1111111, 1.6666667, 2.22222222, 2,77777778, 3.33333333, 3.88888889, 4.44444444, 5.])`. so it produces an array of evenly spaced points between the range
* the number of dimensions in an array is determined by the num of brackets
* with nympy we can make an identity matrix I with `np.eye(size)` passing the size (num of rows or columns). its 2D and its square, diagonal is ones and rest is zero
* with nympy we have a large selection of random methods in `np.random.` (hit TAB to see them)
* we can create arrays of random number (uniform distribution) passing the size (one argument per dimension size ) `np.random.rand(size)` or `np.random.rand(sizeX,sizeY)`
* if we want to return an array of samples of the standar normal or Gaussian distribution use `np.random.randn(size)` again here a number per dimension
* `np.random.randint(low,high,size)` returns random integer numbers from a low to a high number
* Attributes of Arrays:
	* we can reshape them passing the new dimensions and sizes `arr = np.arange(25)` 1 D size25 vector becomes a 5 * 5 matrix with `arr.reshape(5,5)`. if we cant fill the new array completely we get ewrror. the total size must remain Unchanged
	* we can find min and max value in an array with `my_array.max()` or `.min()` 
	* with `.argmax()` and `.argmin()` we get the location of the min or max
	* `.shape()` returns the shape of the vector eg. (25,)
	* we can use `.dtype()` to find the datatype
* we can reduze the size of method calls by importing the methods from the libs in the namespace `from numpy.random import randint`

### Lecture 19 - Numpy Array Indexing

* we set a range 0-10 array with `arr = np.arange(0,11)`
* to pick an element in an array is similar to picking from a list `arr[9]`
* we can use slicing like in lists `arr[1:5]` or arr[:5] or `arr[::2]` to perform jump
* an array can broadcast or set a value to a slice `arr[0:2] = 100` => array([100,100,2,3,4,5,6,7,])
* a slice of an array points to the original array. if i get a slice and the set it a fixed var (broadcast) if i go back to the array the broadcast value takes effect there as well.
* if we want a new copy we have to use `arr.copy()` and do the changes there
* we create a 2d array `arr_2d = np.array([[5,10,15],[20,25,30],[35.40,45]])`
* we can select an element of a 2d matrix `arr_2d[0][0]` => 5 row-column `arr_2d[0]` returns the whole first row
* we can use comma single bracket notation `arr_2d[2,1]`=> 40
* we gan get slices or submatrices with `arr_2d[:2,1:]` => array([[10,15],[25,30]]) aggain if we omit a dimension we get the rows slice
* we can use conditional selection `arr = np.arange(1,11)` we can take this and use conditional selection to turn it to a boolean array. e.g `bool_arr = arr > 4` is `array([false,false,false,false,true,true,true.., dtype=bool])`
* we can use the boolean array for conditional selection `arr[bppl_arr]` => `array([5,6,7,8,9,10])` so i pass the bool array to the array and filter it out, sekect what is true or `arr[arr>5]`

### Lecture 20 - Numpy Operations
