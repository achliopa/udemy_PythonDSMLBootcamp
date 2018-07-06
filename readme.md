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
* we can use a scalar broadcasting its effect to all array elements `arr+100` => `array([100,101,102,103,104,105,106...])` same we can doo  - / * **
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
* we can set the index. we set a new list of labels `newind = 'CA NY WY OR CO'.split()`. we then set it as column `df['States'] = newind` and then st this column as index with `df.set_index('States')` . the effect of that is to create a new row for the index column title. 

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

* we define a dictionary to pupulate a dataFrame `d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}` filling in as values lists with null values `np.nan`
* we use it to create a datafram `df = pd.dataFrame(d)`
* we drop the missing values with `df.dropna()`. this drops any row that has missing value(s). if we pass axis=1 de drop the columns with missing values
* we can pass a threshold argument to dropna to set the minimum number of null required to drop the row or column `df.dropna(thresh=2)`
* we can fill in missing values with *fillna* `df.fillna(value='FILL VALUE')` will pass the FILL VALUE string to any null datapoint
* we can fill the mean of a column to a missing datapoint `df['A'].fillna(value=df['A'].mean())`

### Lecture 30 - Groupby

* groupby is used to produce aggregates, it allows to group together rows based off of a column and perform an aggregate function on them. e.g we can groupby id
* we use a dictionary to produce a dataframe
```
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
```
* we have a column based on which we can group sales. Company `byComp = df.groupby('Company')`. this produces a groupby object not a df. 
* the way to use groupby objects is to apply aggregate functions on them `byComp.mean()`
* we can use many aggregate functions like `.sum() .std()`
* the result of the aggregate function is a dataframe. we can use df methods on it `byComp.sum().loc['FB']`
* usually we produce oneliners by chaining methods
* usually aggregate methods operate an return only number columns to the newdataframe. some can operate on strings and return result columns for them e.g .count() .max() .min()
* if insteade of an aggregate method we chain a `.descripe()` method to a groupby object, we get many aggregate results in the resulting df. 
we can chain after the aggregate or desgribe `.transpose()` to view them as a row rather than column

### Lecture 31 - Merging, Joining and Concatenating

* we create 3 dataframes (df1,df2,df3) with same labeled columns ('A'.'B','C','D'), continuous indexes and datapoints strings that represend the combination of column and index e.g 'B5'
* we concatenate them (glue them together, stacking up rows) with `pd.concat([df1,df2,df3])`. the dimensions should match on the axis we are concatenating on. here we dont spec axis so we concatenate along the row (axis=0) which has size 4 for all
* we can concatenate along the column with `pd.concat([df1,df2,df3], axis=1)`. the rule about size applies. missing values are NaN. the result is a dataframe
* we create two new dataframes with same indexes, columns that have continuous letters labels and a key column with the same datapoints in each
```
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})   
```
* we can use merge to merge dataframes in the same way we merge SQL tables. 
* we do it with `pd.merge(left,right,how='inner,on='key')`. the default how is 'inner' like innerjoin, on is where the merge takes place
* we can do inner merge on multiple keys where the merge take place where the compination of keys is the same for both dataframes
* an outer merge stacks up all combinations of keys
* there is also right or left merge
* joining is a conveninet method of combiningthe columns of two potentially different-indexed DataFrames into a single result DataFrame
* id our dataframes are df1 and df2 we join them `d1.join(df2)` innerjoin. the outer join is `df1.join(df2,how='outer')`. inner join between a range of keys that are common, outer hjoin join all.

### Lecture 32 - Operations

 * we create dataframe `df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})` with 3 columns and unlabeled index
 * to find unique values in dataframes in a column we use `df['col2'].unique()` which returns a numpy array
 * to count the unique values `len(df['col2'].unique())` or `df['col2'].nunique()`
 * to get counts for the occurence of a value in a column we use `df['col2'].value_counts()` which returns a 2-d numpy matrix with the value and counter
 * to select data from a df we can use conditional selection or the apply method
 * we define a methodL
 ```
 def times2(x):
 	return x*2
 ```
 * to broadcast a function to a dataframe we use apply. e.g `df['col1'].apply(times2)` broadcasts its to the col1 values outputing a matrix (numpy)
 * we can apply built in methods or even lambdas `df['col2'].apply(lambda x: x*2)`
 * we can get the column label with `df.columns` which gives an Index object with column labels or for index `df.index`
 * we can sort values in a column with `df.sort_values('col2')`. index follows the sort
 * to find the nulls in the datafram we can use `df.isnull()`
 * we create dataframe with repeating values
 * we can create pivot tables out of a dataframe with `df.pivot_table(values='D',index=['A','B'],columns=['C'])`. this will be a multiindex dataframe.

 ### Lecture 33 - Data Input and Output

 * to input data from external sources we need to install extra libraries.
 ```
 conda install sqlalchemy
 conda install lxml
 conda install html5lib
 conda install BautifulSoup4
 ```
 * we will io data from CSV,Excel,HTML,SQL
 * `import pandas as pd`
 * we find our current dir in Jupyter with 'pwd'
 * we have a csv file named *example* with the following content
 ```
a,b,c,d
0,1,2,3
4,5,6,7
8,9,10,11
12,13,14,15
 ```
 * we read a csv named *example* in the current dir. `df = pd.read_csv('example`)` which gets imported and stored as a dataframe
 * pandas can read from a wide variety of file formats with `.read_<filetype>(<filename>)`
 * we can store to a file e.g csv with `df.to_csv('my_output', index=False)` we set index to false as we dont eant to save the index as a separate column
 * pandas can read from excel but read only the data. not formulas  or macros. reading excels with formulas will make it to crash
 * to read from excel we install `conda install xlrd` (installed with anaconda) or with pip
* we have an excel file *Excel_sample.xlsx* with same data as the csv in Sheet1
* pandas treats excel as a set of dataframes (a dataframe per sheet)
* the expression to read is `df = pd.read_excel('Excel_Sample.xlsx',sheetname='Sheet1')` which stores the input in a dataframe
* we can store a dtaframe to an excel sheet `df.to_excel('Excel_Sample2.xlsx',sheet_name='NewSheet')`
* to read from HTML we need to install some libraries and restart jupyter notebook
```
 conda install lxml
 conda install html5lib
 conda install BautifulSoup4
```
* we will read from the page *http://www.fdic.gov/bank/individual/failed/banklist.html* whoch contains a table
* we read with `df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')` which stores data in a list where df[0] is a DataFrame
* panda is not the best way to read sql as there are many libs for each SQL DB flavor
* we create a sqlengine `from sqlalchemy import create_engine`
* then we create an imemory sqlite db with the engine to test our code `engine = create_engine('sqlite:///:memory:')`
* we store a dataframe as a sql table in our test db `df.to_sql('data', engine)` data is the name of the table
* we readthe table and store it as a dataframe `sql_df = pd.read_sql('data',con=engine)`

## Section 7 - Python for Data Analysis - Pandas Exercises

### Lecture 34 - SF Salaries Exercise 

* we get datasets from kaggle
* Check the head of the DataFrame `sal.head()`
* Use the .info() method to find out how many entries there are. `sal.info()`
* What is the average BasePay ? `sal['BasePay'].mean()`
* What is the highest amount of OvertimePay in the dataset ? `sal['OvertimePay'].max()`
* What is the job title of JOSEPH DRISCOLL ? `sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']`
* How much does JOSEPH DRISCOLL make (including benefits)? sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']`
* What is the name of highest paid person (including benefits)? sal[sal['TotalPayBenefits'] == max(sal['TotalPayBenefits'])]`
* What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid? `sal[sal['TotalPayBenefits'] == min(sal['TotalPayBenefits'])]`