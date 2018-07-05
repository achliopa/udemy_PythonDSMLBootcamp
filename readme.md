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

* we setup an array `arr = np.arange(0,11)`
* we can add two arrays eloement by element with `arr + arr` subtract them or multiply them
* we can use a scalar broadcasting its effect to all array elements `arr+100` => `array([100,101,102,103,104,105,106...]) same we can doo  - / * **
* numpy throws warnings instead of errors. e.g zero division
* numpy has special word *inf* for infinity and *nan* for not a number
* we can use universal operations on arrays `np.sqrt(arr)` squareroots every element in the array `np.exp(arr)` raises array to exp and `np.max(arr)` finds the maximum value like `arr.max()` 
* [univ oper list](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)

### Lecture 21 - Numpy Exercise Overview

* to sum all columns in a matrix `mat.sum(axis=0`
* see docstring of function in jupyter (shift+tab)

## Section 6 - Python for Data Analysis - Pandas

### Lecture 24 - Introduction to Pandas

* Pandas is an open source library built on top of NumPy
* It allows for fast analysis and data cleaning and preparation
* It excels in performance and productivity
* It also has buil-in  visualization features
* It can work with data from a variety of sources
* we can install pandas by going to your command line or terminal and using either `conda install pandas` or `pip install pandas`
* we will see:
	* series
	* dataframes
	* missing data
	* groupBy
	* merging,joining, concat
	* operations
	* data IO

### Lecture 25 - Series

* panda datatype like a numpy array
* the difference between the two is that pandas series can be accessed by label (or indexed by label)
* to use pandas we need to import numpy first
```
import numpy as np
inport pandas as pd
```
* we create various series from various datatypes.
* first we create standard python lists
```
labels = ['a','b','c']
my_data = [10,20,30]
```
* then we cast the values to numpy array `arr = np.array(my_data)`
* and we set the relationship between the two as a standard python dictionary `d = {'a':10,'b':20,'c':30}`
* we end up with 4 objects, 2 lists, 1 array, 1 dictionary
* we create the panda series with `pd.Series(data = my_data)` setting only the actual data not the labels. this auto indexes with numbers 0,1,2,...
* we can specify the index with `pd.Series(data=my_data, index=labels)`. now the indexes have labels 'a'.'b','c'
* so unlike numpy arrays we have labeled indexes. so we can call the data points using the labels
* we can use `pd.Series(my_data,labels)` taking care to keep the order
* we can create a series passing a numpy array . is like passign a data list `pd.Series(arr,label)` 
* we can create a series passing a python dictionary. this auto creates the labels using the key value relationship. keys become index labels and values the data.   `pd.Series(d)` <=> `pd.Series(my_data,labels)`
* a series can have any type of data as datapoints. `pd.Series(data=labels)` => `dtype: pbject`
* it can even hold functions as datapoints `pd.Series(data=[sum,print,len])`
* grabbing data from a series is similar to grabbing data from a dictionary `ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])` `ser1['USA']` => 1. i should now the label type thow. if there are no labels i grab them like python lists or numpy arrays
* adding series is done based on index. `series1+series2` if an index is not common ehat happens is that the result series interpolates non common label datapoints but the val is NaN
* when we perform operations on integer datapoints they are converted to floats

### Lecture 26 - DataFrames: Part 1

* expose the true power of pandas
* we import randn `from numpy.random import randn` and seed the generator `np.random.seed(101)`
* DataFram builds on series and has similar function signature but has a columns argument  `df = pd.DataFrame(randn(5,4),['A','B','C','D','E'], ['W','X','Y','Z'] )` columnd arg is the label index for columns. so dataframe is for numpy matrices what series is for numpy 1-D vectors. `df` prints out the table with axis labels
* DataFrame in essence is a set  of series sharing an index. we can use indexing and selection  to extract series out of a DataFrame e.g `df['W']` an alternate way is `df.W`
* if we pass multiple keys `df['W','Z']` we get a sub DataFrame
* we can set a new column. not as `df['new']` but as `df['new'] = df['W'] + df['Y']` we need to assign a defined column to it
* we can delete a column by `df.drop('W', axis=1)` if i dont specify an axis i can delete a row passing the correct label. drop does not affect the original dataframe. to alter the orginal dataframe we must pass as argument *inplace=True* `df.drop('W', axis=1,inplace=True)`
* to drop a row we can omit axis od set it to 0 `df.drop('E',axis=0)`
* rows have axis = 0 and columns have axis =1 this comes from numpy this is shown in `df.spape()` with gives a tuple `(5,4)` 0 index is rows and 1 is columns
* to select rows we use `df.loc['A']` this returns a series so rows are series like columns. to select locs we canuse index instead of lables with iloc `df.iloc[1]`
* to select subsets of rows and columns we use loc and numpy notation with comma `df.loc['B','Y']` or `df.loc[['A','B'],['X','Y']]`

### Lecture 27 - DataFrames: Part 2

* we see conditional selection and multyindex selection
* we can use operands and broadcast selectors in all datapoints `booldf = df > 0` gives the df where all datapoints are boolean
* if we pass bool_df in df `df[bool_df]` we the df where all datapoints where booldf is false are NaN (null). another way is `df[df>0]`
* we spec `df['W']>0` and get a series of the column with booleans. if I pass it as a conditional selector in df `df[df['W']>0]` this filter is applied in all columns filtering out 3 row
* we want to get all the rows of the datafram where Z is <0. `df[df['Z']<0]`. we know Z is <0 only in 3rd row so that is what we get back
* we can do selction on the fly on the filtered dataframe `df[df['W']>0]['X']` or `df[df['W']>0][['Y','X']]`
* one liners are generaly faster
* we can use multiple conditions, `(df['W']>0) and (df[Y]>1)` throuws an error as normal python boolean operators cannot handle series of data, only single bools. he have to use & instead `(df['W']>0) & (df[Y]>1)` is valid and can be passed ans condition selector `df[(df['W']>0) & (df[Y]>1)]` for or we use |
* we can reset the index or set it to something else. we use `df.reset_index()` to reset the index (now they are not labels but index numbers) and set the labels to a separate column named *index*. this opeartion does not alter the original dataframe
* we can set the index. we set a new list of labels `newind = 'CA NY WY OR CO'.split()`. we then set it as column `df['States'] = newind` and then st this column as index with `df.set_inex('States')` . the effect of that is to create a new row for the index column title. 

### Lecture 28 - Dataframes: Part 3

* we look into multindex dataframes and multilevel index.
* we do multiIndex as follows:
	* we create two lists
	```
	outside = ['G1','G1','G1','G2','G2','G2']
	inside = [1,2,3,1,2,3]
	```
	* we zip them to a list of tuples
	```
	hier_index = list(zip(outside,inside))
	``` 
	* we create a dataframe multiindex  using the special pandas method *MultiIndex* specifing we will use tuples
	```
	hier_index = pd.MultiIndex.from_tuples(hier_index)
	```
	* we get a MultiIndex object with levels and labels
	* we create a dataframe with 6by2 random data, the hier_index and  lables for the 2 columns
	```
	df = pd.DataFrame(randn(6,2),hier_index, ['A','B'])
	```
	* the result is a 6 by 2 dataframe with 2 columns for indexes, the first w/ lables G1 and G2 and the second with labels 1,2,3 all grouped according to their appearance on the lists we declared them. the two columns are called levels
* we can call data from multiindexed dataframes `df.loc['G1']` and we can chain multiple levels `df.loc['G1'].loc[1]`
* we can get and set the names of the index level columns: get `df.index.names` retrieves a FrozenList with None. to set we use `df.index.names = ['Groups','Num']`. the index labels are in a new extra row
* we can select a specific element with `df.loc['G2'].loc[[2]['B']`
* we can get a crossSection or rows/columns with xs which we can use when we have m,ultilevel index. `df.xs('G1')` is like `df.loc['G1']` but xs has the ability to go insiude the multilevel index
* say we want all values with num index = 1 `df.xs(1,level='Num')`

### Lecture 29 - Missing Data

* 