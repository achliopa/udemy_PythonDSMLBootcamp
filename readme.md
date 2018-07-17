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
* What is the name of highest paid person (including benefits)? sal[sal['TotalPayBenefits'] == max(sal['TotalPayBenefits'])]['EmployeeName']`
* alternative mehod to get max `sal.lov[sal['TotalPayBenefits'].idxmax()]` or `sal.iloc[sal['TotalPayBenefits'].argmax()]`
* What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid? `sal[sal['TotalPayBenefits'] == min(sal['TotalPayBenefits'])]`
* What was the average (mean) BasePay of all employees per year? (2011-2014) ? `sal.groupby('Year')['BasePay'].mean()`
* How many unique job titles are there? `len(sal['JobTitle'].unique())` or `sal['JobTitle'].nunique()`
* What are the top 5 most common jobs? `sal.groupby('JobTitle')['JobTitle'].count().sort_values(ascending=False)[:5]` alternatively `sal['JobTitle'].value_counts().head(5)`
* How many Job Titles were represented by only one person in 2013? 
```
usal = sal[(sal['Year']==2013)].groupby('JobTitle')['JobTitle'].count()
len(usal[usal == 1])
```
alternatively `sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)`
* How many people have the word Chief in their job title? (This is pretty tricky)
	* my solution	
	```
	ysal = sal.drop_duplicates(["EmployeeName", "Year"])
	ysal[ysal['JobTitle'].str.lower().str.contains('chief')].shape[0]
	```
	* teacher solution
	```
	def chief_string(title):
		if 'chief' in title.lower().split():
			return True
		else:
			return False
	sum(sal['JobTitle'].apply(lambda x: chief_string(x)))
	```
* Bonus: Is there a correlation between length of the Job Title string and Salary?
```
# we make a new column for title length
# we use method corr() for correlation
sal['title_len'] = sal['JobTitle'].apply(len)
sal['TotalPayBenefits','title_len'].corr()
```

### Lecture 36 - Ecommerce Purchase Exercises

* What is the average Purchase Price? `ecom['Purchase Price'].mean()`
* What were the highest and lowest purchase prices?
```
ecom['Purchase Price'].max()
ecom['Purchase Price'].min()
```
* How many people have English 'en' as their Language of choice on the website? `ecom[ecom['Language'] == 'en'].shape[0]` or `ecom['Language'] == 'en']['Language'].count()`
* How many people have the job title of "Lawyer" ? `ecom[ecom['Job'] == 'Lawyer'].shape[0]`
* How many people made the purchase during the AM and how many people made the purchase during PM ? `ecom['AM or PM'].value_counts()`
* What are the 5 most common Job Titles? `ecom['Job'].value_counts().head(5)`
* Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction? `ecom[ecom['Lot'] == '90 WT']['Purchase Price']`
* What is the email of the person with the following Credit Card Number: 4926535242672853 `ecom[ecom['Credit Card'] == 4926535242672853]['Email']`
* How many people have American Express as their Credit Card Provider and made a purchase above $95 ? `ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95.0)].shape[0]`
* Hard: How many people have a credit card that expires in 2025? ` sum(ecom['CC Exp Date'].apply(lambda x: x[3:]=='25))`
* Hard: What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...) `ecom['Email'].str.split('@',1,expand=True)[1].value_counts().head(5)` other solution `ecom['Email].apply(lambda email: email.split('@')[1]).value_counts().head(5)`
* to get column names `df.columns`

## Section 8 - Python for Data Visualization - Matplotlib

### Lecture 40 - Introduction to [Matplotlib](https://matplotlib.org/)

* most popular plotting library for Python
* it gives vontrol ofer every aspect of a figure
* designed to give Matlab plotting look and feel
* works well with pandas and numpy
* to install it `conda install matplotlib` or `pip install matplotlib`
* in project site => examples we can see examples of the plots with code examples

### Lecture 41 - Matplotlib Part 1

* we import matplotlib `import matplotlib.pyplot as plt`
* to use it efficiently in jupyter we need to enter `%matplotlib inline`. this will allow us to see our plots as we write them in notebook
* if we dont use jupyter after our code we need to enter `plt.show()` before executing it
* we will use numpy for our first example 
```
import numpy as np
x = np.linspace(0,5,11)
y = x**2
```
* there are 2 ways to create plots, functional and object oriented
```
# Functional
plt.plot(x,y) # shows the plot
```
* `plt.show()` in jupyter prints the plot
* we can add matlab style arguments like color and style `plt.plot(x,y,'r--')` for red and dashed
* if we want to add labels we use `plt.xlabel('my labbel')` or `plt.ylabel('my labbel')`
* for plot title `plt.title('My title')`
* we can create multiplots in the same canvas using the subplot method subplot takes 3 arguments: num of rows, num of columns, and the plot num we refer to
```
plt.subplot(1,2,1)
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(y,x,'b')
```
* we will now have alook at the OO method: we create figure objects and call methods on them
```
fig = plt.figure() # creates afigure object like a cnavas
axes = fig.add_axes([0.1,0.1,0.8,0.8])
```
* we add axis with a method passing a list of axis. the list takes 4 args: left, bottom, width and height arg, which take a val from 0 - 1 (represent percentage of screen) left and bottom are coordinates and wight and height size
* we then plot on the axes to see the plot in our set of axes `axes.plot(x,y)`. the plot is the same as before but in an OO approach
* we can put labels like before on the set of axes
```
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')
```
* we will put 2 sets of figures on the same canvas using OO
```
fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8]) # for first plot
axes2 = fig.add_axes([0.2,0.4,0.4,0.3]) # for second plot
```
* the plots are overlayed one over other as the axes are overlapping
* we plot on the axes and the advantage of OO approach becomes apparent
```
axes1.plot(x,y)
axes2.plot(y,x)
```

### Lecture 42 - Matplotlib Part 2

* we will create subplots using OO `fig, axes = plt.subplots()`
* by using `axes.plot(x,y` we get a plot like before
* with subplots we can specify rows and columns `fig, axes = plt.subplots(nroows=1,ncols=2)`, now we have two empty canavases to apply our plots as 2 columns. 
* so subplot manages the axes automaticaly. the fig,axes syntax is tuple unpacking
* axes is actually an array of two (1 per subplot). we can iterate through it or index it
```
for current_ax in axes:
	current_ax.plot(x,y)
# OR INDEX EQUIVALENT
axes[0].plot(x,y)
axes[1].plot(x,y)
```
* each axes element is an object where we can call the methods we have seen before
* at the end of the plot statements (especially when we use subplots) we should use `plt.tight_layout()`
* We will now see figure size, aspect ratio and DPI. matplotlib allows control over them
* we can control figure size, dpi `fig = plt.figure(figsize=(3,2),dpi=100)` usually we dont set dpi (analysis) in jupyter. the tuple we put in figsize is (inches_width,inches_height). we can set the figsize to subplots
* to save a figure to a file we use savefig `fig.savefig('mu_pickture.jpg`,dpi=200) . the filename defines the file format
* we can add legends to our plots to clarify the plot. we do it by adding `axes.legend()` as last statement wehre axes are defined
* for this to work we need to add a label argument to our plots `axes.plot(x,y,label='X in power of 2')`
* we can position the legend passing a param in legend() e.g axes.legend(loc=0) aka best position or passing a tuple for bottom left postiion in % (0 to 1)

### Lecture 43 - Matplotlib Part 3

* seting colors in plot. we pass them as argument colors in plot as strings . they are cpecked as literals or RGB hex e.g `ax.plot(x,y,color="green")` or `ax.plot(x,y,color="#FF00FF')`
* we can set plot line width (in px) or line style `ax.plot(x,y,linewidth=20)`  or `ax.plot(x,y,lw=20)`. we can also control the a (transparency) `ax.plot(x,y,alpha=0.5)`
* linestyle is again passed as param `linestyle="--"` or `ls="--"` or ":" or "steps" or ...
*  markers are used when we have a few datapoints. we set them as argumetns in plot `ax.plot(x,y,marker="*")` or any other mark we want. we can set their marakersize with the plot argument`marakersize=13` or any int
* we can customize markers further with `markerfacecolor="yellow", markeredgewidth=3,markeredgecolor="green"`
* we can limit the x and y in our plots `axes.set_xlim([0,1])` passing lower bound and upper bound or ylim limiting the axis in real values
* matplotlib supports a lot of plot types (see links and [tutorial](http://www.labri.fr/perso/nrougier/teaching/matplotlib/))
* advanced topics avaialble as a bonus notebook

### Lecture 44 - Matplotlib Exercises

* all commands on a plot must be in one cell

## Section 9 - Python for Data Visualization - Seaborn

### Lecture 46 - Introduction to [Seaborn](https://seaborn.pydata.org/)

* seaborn is a statistical ploting library with beautiful default styles
* works very well with panda dataframe objects
* we install it with `conda install seaborn` or `pip install seaborn`
* there are many examples in docs site

### Lecture 47 - Distribution Plots

* we will analyze distribyution of a dataset
* we import seaborn `import seaborn as sns`
* we set matplotlib as inline (seaborn use it) `%matplotlib inline`
* seaborn comes with inbuilt datasets for testing `tips = sns.load_dataset('tips')` tips is one of them a simple dataframe about tips
* for distplot we pass a single column of our dataframe. it shows uniform distribution `sns.distplot(tips['total_bill'])` we get a histogram and a KDE (kernel density estimation or probability density function of a random variable)
* we can remove the kde and keep only th histogram by setting it to false `sns.distplot(tips['total_bill'],kde=False)`
* the histogram is a distribution chart which shows where our values lay on the value min max range. the x range is split into areas or bins and the y axis is a count, we can change the number of bins `sns.distplot(tips['total_bill'],kde=False,bins=30)` . if our bins value is too high we plot each value
* jointplot allow us to match two distplots of bivarial data, we pass an x variable and ay variable and our dataset(dataframe), usually x and y are column labels `sns.jointplot(x='total_bill',y='tip',data=tips)`
* what we get is 2 distplots on each axis and a scaterplot to show the correlation, when bill is higher tip is higher also
* in jointplot we can set an addional paramter called kind `sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')` the default is scatter. we  can use hex instead to get a hexagon distribution representation (more points, darker color)
* we can also set *kind="reg"* which gives a scatterplot with a regression line overlaid (linear regression) like a linear fit. we can also put *kind="kde"* to get a 2D KDE plot showing the density
* pairplot shows pairlike relationships within an entire dataset(dataframe). it also supports a color hue param for categorical data. what pairplot does is essential every possible combination of jointplots in a datasets numerical values `sns.pairplot(tips)`. 
* when doing a plot of the same column it shows a histogram instead the rest is scatterplot
* if we add the param hue="sex" we get colored plot showing the 2 different categories of sex with different color. this is a great way to add in the mix non numeral categorical data (passing their column label)
* we can choose a palette for hue `sns.pairplot(tips,hue='sex',palette='coolwarm')`
* next plot we look at is rugplot. in rugplot we pass a single column `sns.rugplot(tips["total_bill"])`. rugplot plot 1 dash for every datapoint in the column with the x axis being the value range.dash stack on each other. the distplot counts the values adding them to bins. the rugplot shows a density like representation of the datapoints
* so how we build the kde plot from rugplot? kde stands for Kernel Density Estimation plot. from rugplot and datapoints gaussian (normal) distributions are calculated. their sum is the KDE plot
* in an example we make a random data dataset =>  make a rugplot of them => set axis for the plot => use linspace to split the axis => plot a normal distribution for all the rugplot points => we sum them up to get the kde plot

### Lecture 48 - Categorical Plots

* we again import seaborn and set matplotlib inline
* in these plots we are interested in the distribution of categorical data
* we start with barplot, where we aggregate categorical data based on some function (default is mean) `sns.barplot(x=,y=.data=tips)` we again set the x axis the y axis and the dataset. usually x is the categorical column label e.g "sex" and for y we choose a column that is numeric e.g "total_bill". what we get a sdefault is 2 bars with the mean or average total bill value for male and female customers.
* we can add an other aggregate function instead of the default with specifying the estimator parameter `sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)` for standard deviation. we can use a custom function of our own
* a countplot in seaborn is a barplot where in the y axis we have the counter of occurences of each category in the dataset `sns.countplot(x='sex',data=tips)`
* boxplot (or box and wisker plot) is used to show the distribution of categorical data. again we set x, y and data param like boxplot `sns.boxplot(x="day", y="total_bill", data=tips)`. boxplot shows boxes for quartile of distribution. the wiskers show the rest of the distribution while the dots outside outliers. if we add a hue as parameter passing another categorical column label our boxplot is split showing boxplots for each category of hue. hue is excellent ofn adding insight in analysis
* violiplot also shows distribution of categorical data, arguments are exactly like boxplot `sns.violinplot(x="day", y="total_bill", data=tips)`. violinplot allows plotting of all datapoints in the dataset in a continuous diagram showing the kde of the underline distribution. its harder to read, violinplot also accepts hue param. one cool effect is that with hue and with split=True in violi plot we get per violit plot both hues one per side (as the original is symmetrical on y axis) `sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True)`
* a stripplot is a scatterplot of categorical data. the basic params are same as the other ones `sns.stripplot(x="day", y="total_bill", data=tips)`. with this plot we cannot tell how many points stack on each other. we can use jitter param for this *jitter=True* which makes the line thicker to show all points. again hue is supported and split
* the idea of stripplot and violinplot is combined in swarmplot. its as triplot where poitns are stackedvertically  so that they dont overlap giving the violin shape. params are the same `sns.swarmplot(x="day", y="total_bill", data=tips)`
* swarmplots do not scale well for large numbers(too wide). 
* we can stack a swarmplot on a violin plot to show where the kdes came from
* factorplots take x and y argument and dataset and the kind of plot we want. eg. kind="bar" for barplot `sns.factorplot(x='sex',y='total_bill',data=tips,kind='bar')`

### Lecture 49 - Matrix Plots

* we load 2 built-in datasets from seaborn tips and flights
```
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')
```
* we will look into heatmap plot. in order to work properly our data should be in matrix form (index name and column name match up) so that the cell value shows somthing relevant to both names. 
* so in tips dataset our columns are labeled but the rows not. in order to make it in matrix form our index should have label relevant to the cell. this is done by applying pivot or correlation transformation to the dataset. `tc = tips.corr()` now both rows and columns are labeled
* with data in matrix form i just have to call `sns.heatmap(tc)`. heatmap color values based on gradient scale. we can set the scale with cmap param *cmap="coolwarm"* we can annotate the plot with param *annot=True* this overlays the values on the tiles
* we want to transform the flights dataset  so that index is the month, column the year and datavalues the passengers. we use pivot `flights.pivot_table(index="month",columns='year',values="passengers")`. we get it in matrix form so we can show it in heatmap plot. we can also set linecolor and linewidth params to seperate cells in heatmap
* we also have clustermap matrix plot which is also applied on matrix data sets `snsclustermap(tc)`
* cluster map has hierarchical clusters clustering rows and columns together based on their similarity. this breaks the original order of data but still contains valid representations. Clustermap searches Similarities
* we can change the representation by normalizing data changing the scale `sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)` adding a standard_scale param (1 is 0-1 stale)

### Lecture 50 - Grids

* we again import seaborn and se matplotlib inline
* we use the inbuilt iris dataset `iris = sns.load_dataset('iris')`
* it contains measurements of different flowers and species has categorical data of 3 distinct values
* we plot the pairplot with `sns.pairplot(iris)` (grid of scatterplots of numericalvalues seen before)
* another plot is PairGrid. with `g = sns.PairGrid(iris)` we print only the grid
* pairplot is a simplyfied version of PairGrid. PairGrid needs tweaking but gives more flexibility. to mat a scatterplot on the pairgrid we need to import matplotlib, set the grid (passing the dataset) and the on it map the scatter
```
import matplotlib.pyplot as plt
g = sns.PairGrid(iris)
g.map(plt.scatter)
```
* this produces a grid of scatterplots. its like pairplot. only that its missing the histograms along the axes (it has scatters)
* with PairGrid we have control on the plots we will show on the grid
```
g.map_diag(sns.distplot) # distplot (histogram) on thediagonal
g.map_upper(plt.scatter) # scatterplot on upper half of grid
g.map_lower(sns.kdeplot) # kde plot on the lower half
```
* for facet grid plot we will use the tips dataset `tips = sns.load_dataset('tips`)
* for facet grid we specify the data the column and the row like sublplots in matplotlib `g = sns.FacetGrid(data=tips,col='time',row='smoker')`. the expression produces an empty grid (2 by 2)
* on the grid we do `g.map(sns.distplot,'total_bill')`
* this shows a distplot on each grid cell with the distribution of total_bill on the 4 cases defined by the combination of the 2 category types of each dataset column we chose (time and smoker)
* the data we pass is subplot dependent. so for scatter we need two arrays `g = g.map(plt.scatter, "total_bill", "tip")`

### Lecture 51 - Regression Plots

* we import seaborn, set matplotlib inline and load the tips inbuilt dataset
* we use lmplot for linear regression. we set the feat we want on the x axis and the y axis  `sns.lmplot(x='total_bill',y='tip',data=tips)` what we get is a scatterplot witha linear fit on top. we can spec hue based on sex (we get 2 scatterplots and 2 linear fits on top of each other). we can even set different markers for our hue categories using the parameter *markers=['o','v']* to control the markers size we use *scatter_kws={'s':100})*
* so seaborn uses matlibplot under the hood and with scatter_kws we control the scatterplot of matplotlib passing the dictionary s is size (see documentation)
* instead of separating categories by hue we can use grid `sns.lmplot(x='total_bill',y='tip',data=tips,col='sex')` by passing the col parameter. we can add a row param for an other category (like facet grid but simpler)
* we can combine grid and hue
* size and aspect ratio is adjustable in seaborn with *aspect=0.6,size=8* params

### Lecture 52 - Style and Color

* we import seaborn, se matplotlib inline and load the tips dataset from seaborn
* we do a simple plot `sns.countplot(x='sex',tips)`
* seaborn has a set_style method to set the style for all our plots `sns.set_style('white')` darkgrid, whitegrid, ticks etc are acceptable values
* we can remove spines `sns.despine()` the top and right. we can remove bottom and left by specifying it `sns.despine(left=True)`
* we can control the size and aspect ration by specifying it
* in a non grid plot we can pec it in the matplotlib figure like we have seen `plt.figure(figsize=(12,3))` before our actual plot
* in grid pplots we do it in seaborn `sns.lmplot(x='total_bill',y='tip',size=2,aspect=4,data=tips)`
* we can set the context with `sns.set_context('poster'.font_size=4)` with poster we get a much largez size suitable to be put on a poster, default is notebook
* coloring can be controled with palette parameter. [possible values](https://matplotlib.org/examples/color/colormaps_reference.html)

### Lecture 53 - Seaborn Exercise

* we work with the titanic dataset

## Section 10 - Python for Data Visualization - Pandas Built-in Data Visualization

### Lecture 55 - Pandas Built-In Visualization

* we import numpy and pandas in our notebook and set natplotlib as inline
* we upload 2 csv files using pandas 
```
df1 = pd.read_csv('df1',index_col=0) # remove index column (the imported dataframe has as index dates)
df2 = pd.read_csv('df2')
```
* say we want a *histogram* of all the values in a column of df1. pandas can do it with `df1['A'].hist()`. it calls matplotlib under the hood. like seaborn we can set the number of bins in the histogram as a param *bins=*
* if we dont like the style we can import seaborn in our notebook and the pandas plots are styled like seaborn
* pandas have several inbuilt plots. all of them are called as methods on the dataframe
* we can use the general method plot specifying the kind of plot + params `df1['A'].plot(kind='hist',bins=30)`
* another way is to chain plot ype as method `df['A'].plot.hist(bins=30)`
* we can have an *areaplot* of multiple num columns in a dataframe, even the entire dataframe `df2.plot.area()`, we add transpoarency specifying alpha as a param (0 -1 value) *alpha=0.4*
* also we can have a *barplot* `df2.plot.bar()` of all rows (indexes). we can hacve a *stacked* barplot passing the parameter *stacked=True*
* histograms are most use in pandas builtin
* *lineplot* is also available. we need to specify the x and y axis for this `df1.plot.line(x=df1.index,y='B')`
* we can use matplotlib arguments to control the appearance *figsize=(12,3)* or lw markers etc
* we can do *scatterplots* with pandas. `df1.plot.scatter(x='A',y='B')` we can set color dependent on another column with *c='C'*. we can set cmap as well like in seaborn. instead of color we can set dor size dependent on a COlumns values with *s=df1['C']*200* passing a factor to multiply size
* we can do a *boxplot* of a dataframe with `df2.plot.box()`
* he can ddo hexbin plots. passing x and y columns. we use a bigdtaset dfor this, we can multiply the hex size fy a factor
```
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges')
```
* we can also do *kdeplots* with `df['a'].plot.kde()` od density plots `df2.plot.density()` which is the same thing

### Lecture 56 - Pandas Data Visualization Exercise

* we can limit the ampount of indexed we feed in a plot with ix[:max] `df3.ix[:30].plot.area()`
* we can print legend outside of plot with `plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))`

## Section 11 - Python for Data Visualization - Plotly and Cufflinks

### Lecture 58 - Introduction to Plotly and Cufflinks

* plotly and cufflinks allow us to create interactive visualizations
* [Plotly](https://plot.ly/) is an interactive visualization library
* [Cufflinks](https://github.com/santosjorge/cufflinks) connects plotly with pandas
* we need to install the libs `pip install plotly` and `pip install cufflinks` not available in conda
* we create a virtual env `conda create -n plotly plotly`
* we activate it `source activate plotly`
* we install cufflinks in it `pip install cufflinks`
* we install numpy `conda install -n plotly numpy`
* we install pandas `conda install -n plotly pandas`
* we install matplotlib `conda install -n plotly  matplotlib`
* we install seaborn `conda install -n plotly seaborn`
* we run `anaconda-navigator` from it
* we select our new environment in navigator
* we install notebook in it

### Lecture 59 - Plotly and Cufflinks

* we import numpy and pandas and set matplotlib inline
* we check the plotly version
```
from plotly import __version__
print(__version__)
```
* we need a version > 1/9
* we import cufflinks `import cufflinks as cf`
* we import plotly libs `from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot`
* `init_notebook_mode(connected=True)` connects javascript to our notebook, as plotly connects pandas and python to an interactive JS library
* `cf.go_offline()` to tell cufflinks to work offline
* we get some data for our plots `df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())`
* we createa sceond dataframe using a dictionary `df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})`
* if we write `df.plot()` pandas plots a matplotlib plot (a line plot with hue for each column)
* if insteat we use iplot `df.iplot()` we get the same plot but from plotly. and is interactive, we can zoom pan, use tools
* we can use iplot for a scatterplot specifing kind="scatter" and defining the 2 axes by col name `df.iplot(kind="scatter,x="A",y="B")` this is a line scatterplot!?! .. we fix that by adding param *mode="markers"*, we can also affect size of marks with *size=*
* we do a barplot on df wher x is a categorical column (Category) and y anumerical column (Values) `df2.iplot(kind="bar", x="Category",y="Values")`
* we can use grpup by or perform aggregate funtions on our dataframe to bring it to a form where we can plot it. `df.sum().iplot(kind="bar")`
* for boxplots `df.iplot(kind="box")`
* we can do 3d surface plot using a new dataframe `df3=pd.DataFrame('x':[1,2,3,4,5],'y':[10,20,30,40,50],'z':[100,200,300,400,500])`
* we plot the surface plot with `df3.iplot(kind="surface")` ce can set a colorscale with *colorscale="rdylby"*
* we can do histogram plots `df['A'].iplot(kind='hist',bins=345)` . if we pass al the dataframe we get overlapping histogram of all columns
* we can use spread type visualization (for stock analysis) df['A','B'].iplot(kind="spread"). we get a linechart and a spread (area PLOT) beneath
* we can use bubblieplot (like a spreadplot but size of marks show a value) so we specify the column for size `df.iplot(kind="bubble",x="A",y="B",size="C")` popular in UN reports
* scatter matrix is like a seaborn pairplot. is is applied on a datafram `df.scatter_matix()`
* cufflinks can do technical analysis plots for financial (beta)

## Section 12 - Python for Data Visualization - Geographical Plotting

### Lecture 60 - Introduction to Geographical Plotting

* geo plotting is difficult due to the various formats the data can come in
* we will focus on using plotly for plotting
* matplotlib has also a basemap extension
* usualy the dataset determines the library we will use

### Lecture 61 - Chloropleth Maps: Part 1 USA

* we should keep the notebook of the course as refrence as chloropleths are difficult to master
* we import plotly `import plotly.plotly as py`
* we import plotly libs `from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot`
* we set noteboom mode `init_notebook_mode(connected=True)`
* we start by setting our configuration object as a dictionary `data = dict(type='cloropleth`,locations=['AZ','CA','NY'], locationmode='USA-states',colorscale="Portland",text=['text 1','text 2','text 3'], z = [1.0 ,2.0 ,3.0],colorbar={'title':'Colorbar Title'})`. text is the text shown when we hover over the area , z are the actual values, colorscale is the colorscale we use, colorbar sets configfor colorscale legend (title), locations are a way to identify the areas on the colorpleth (standardized) and the locationmode is the type of map
* next we create the layout as a dictionary passing the map scope. `layout = dict(geo = { 'scope':'usa'})`
* we neer to import the geo  library from plotly to plot our cloropleth `import plotly.graph_objs as go`
* we use the FIgure method (constructor) from go to instantiate our choropleth `choromap = go.Figure(data=[data],layout = layout)` note we pass data dictionary in an array
* we plot with `iplot(choromap)` we can plot a non-interactive map for printing with `plot(choromap)`
* for choropleth we plot a Figure that uses two objects . data and layout both dictionaries
* in data type sets the type of plor. locations and location mode are interlinked to identify areas. text z and locations have 1:1 relationship within them
* we will use real data this time `df = pd.read_csv('2011_US_AGRI_Exports')`
* we set our data dict using columns from dataframe
* we also add a new argument. a marker as a nested dictionary seting its line as a nested dicxtionary
```
data = dict(type='choropleth',
            colorscale = 'YIOrRd',
            locations = df['code'],
            z = df['total exports'],
            locationmode = 'USA-states',
            text = df['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Millions USD"}
            ) 
```
* we set the layout passing a title we also set the lakes?!
```
layout = dict(title = '2011 US Agriculture Exports by State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )
```
* we then instantiate the figure and plot it
* the marker sets a line between states

### Lecture 62 - Choropleth Maps: Part 2 World

* we load a new dataset `df = pd.read_csv('2014_World_GDP')`
* we set the data config dictioanary
```
data = dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['GDP (BILLIONS)'],
        text = df['COUNTRY'],
        colorbar = {'title' : 'GDP Billions US'},
      ) 
```
* there is no location mode (only countries) using international CODE. we again use columns from df
* we set the layout setting the geo in a new way
```
layout = dict(
    title = '2014 Global GDP',
    geo = dict(
        showframe = False,
        projection = {'type':'Mercator'}
    )
)
```
* we set the figure and plot
* see [docs](https://plot.ly/python/choropleth-maps/) for more 
* with choropleth we need our dataset to contain the countrcodes or statecodes it understands

### lecture 63 - Choropleth Maps Excercises

* we can set `locationmode="country names",` to use full country names
* we can set reversescale=True to reverse color scale

## Section 13 - Data Capstone Project

### Lecture 66 - 911 Calls Project

* convert column from string to timestamp `df['timeStamp'] = pd.to_datetime(df['timeStamp'])`
* What are the top 5 zipcodes for 911 calls?  `df['zip'].value_counts().head(5)`
* What are the top 5 townships (twp) for 911 calls? `df['twp'].value_counts().head(5)`
* Take a look at the 'title' column, how many unique title codes are there? `df['title'].unique().shape[0]`
* In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value. `df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])`
* What is the most common Reason for a 911 call based off of this new column? `df['Reason'].value_counts()`
* Now use seaborn to create a countplot of 911 calls by Reason. `sns.countplot(x='Reason',data=df)`
* You should have seen that these timestamps are still strings. Use pd.to_datetime to convert the column from strings to DateTime objects. `df['timeStamp'] = pd.to_datetime(df['timeStamp'])`
* You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.
```
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
```
* Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:
```
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].apply(lambda x: dmap[x])
```
* Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. ```
sns.countplot(x='Day of Week',data=df,hue='Reason')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
```
* Now do the same for Month:
```
sns.countplot(x='Month',data=df,hue='Reason')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
```
* Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.
```
byMonth = df.groupby('Month').count()
byMonth.reset_index(level=0, inplace=True)
byMonth.head()
```
* Now create a simple plot off of the dataframe indicating the count of calls per month.
```
plt.xlabel('Month')
plt.ylabel('Num of Calls')
plt.title('Number of 911 calls per Month')
plt.grid(True)
plt.plot(byMonth['Month'],byMonth['e'].values)
```
* Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. `sns.lmplot(x='Month',y='e',data=byMonth)`
* Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. `df['Date'] = df['timeStamp'].apply(lambda x: x.date())`
* Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
```
byDate = df.groupby('Date').count()
plt.figure(figsize=(12,3))
plt.xlabel('Date')
plt.ylabel('Num of Calls')
plt.title('Total Calls per Day')
plt.grid(True)
plt.plot(byDate.index,byDate['e'])
```
* Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call
```
traffic = df[df['Reason']=='Traffic'].groupby('Date').count()
plt.figure(figsize=(12,3))
plt.xlabel('Date')
plt.ylabel('Num of Calls')
plt.title('Traffic')
plt.grid(True)
plt.plot(traffic.index, traffic['e'])
```
* Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an unstack method. Reference the solutions if you get stuck on this!
```
df1 = df.copy()
df1.set_index(['Day of Week','Hour'], inplace=True)
dfh = df1.groupby(level=['Day of Week','Hour']).sum()
pivot = dfh.pivot_table(values='e',index=dfh.index.get_level_values('Day of Week'),columns=dfh.index.get_level_values('Hour'))
pivot
```
* Now create a HeatMap using this new DataFrame.
```
plt.figure(figsize=(12,6))
sns.heatmap(pivot,cmap="viridis")
```
* Now create a clustermap using this DataFrame.
```
sns.clustermap(pivot,cmap="viridis")
```
### Lecture 69 - Finance Data Project

* we need to install pandas-datareader to fetcj the data  `pip install pandas-datareader`
* we can download data from [link](https://www.dropbox.com/s/s9uq4qvls4rghm7/all_banks?dl=0) and import them `df = pd.read_pickle('all_banks')`

## Section 14 - Introduction to Machine Learning

### Lecture 76 - Introduction to Machine Learning

* we will use the [ISLR book](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf) from Gareth James as reading assignment
* Statistical Learning is another name for Machine Learning
* this book will be used as mathematical reference. the code in the book is in R
* THe book is a reference if we want to dive into the mathematics behind the theory
* Reading is optional
* machine learning is a method of data analysis that automates analytical model building
* using algorithms that iteratively learn from data machine learning allows computers to find hidden insights without beign explicitly programmed where to look
* Machine Learning is used in:
	* Fraud detection
	* Web search results
	* Real-time ads on web-pages
	* Credit scoring and next-best offers
	* Prediction of equipment failures
	* New pricing models
	* Network intrusion detection
	* Recommendation Engines
	* Customer Segmentation
	* Pattern and Image Recognition
	* Financial Modeling
* The Machine Learning Process is the Following: Data Acquisition -> Data Cleaning -> [ Model Training & Building <-> Model Testing] || [Test Data -> Model Testing] -> Model Deployment
*  To clean the data we can use Pandas.
*  Clean Data are Split into Training Data and Test Data
*  we train our machine learing model on the training data
* we test our model on the test data
* we tune our model until testing on test data gives good results
* then we deploy our model
* There are 3 main types of Machine Learning algorithms
	* *Supervised Learning:* we have labeled data and try to predict a label based on known features
	* *Unsupervised Learning:* We have unlabeled data and we are trying to group together similar data points based off of features
	* *Reinforcement Learning:* Algorithm learn to perform an action from experience
* Supervised learning:
	* algorithms are trained using *labeled* examples, like an input where the desired output is known e.g a piece of equipment could have datapoints labeled either F (failed) or R (runs). 
	* The learning algorithm receives a set of inputs along with the corresponding correct outputs. 
	* The algorithm learns by comparing its actual output with correct outputs to find errors. it then modifies the model accordingly
	* through methods like classification, regression, prediction and gradient boosting, supervised learning uses patterns to predict the values of the label on additional unlabled data
	* Supervised learning is commonly used in applications where historical data predicts likely future events
	* e.g it can anticipate when credit card transactions are likely to be fraudulent or which isurance customer is likely to file a claim
	* it can predict the price of a house based on different features for houses for which we have historical price data
* Unsupervised Learning:
	* is used against data that has no historical labels
	* the system is not told the 'right answer' The algorithm must figure out what is being shown
	* the goal is to explore the data and find some structure within it
	* or it can find the main attributes that separate customer segments from each other
	* popular techniques include: self-organizing maps, nearest-neighbour mapping, k-means clustering and singular value decomposition
	* these algotruthms are also used to segment text topics recommend items and identify data outliers
* Reinforcement Learnign:
	* is often used for robotics, gaming and navigation
	* with reinforcemnt learing, the algorithm discovers through trial and error which actions yield the greatest rewards
	* this type of learning has three primary components: the agent(the learner or decision maker), the environment (everything the agent interacts with) and actions (what the agent can do)
	* the objective is for the agent to choose actions that maximize the expected reward over a given amountof time
	* the agent will reach the goal much faster by following a good policy
	* the goal in reinforced learning is to learn the best policy
* Each Algorithm or ML topic in this course includes:
	* A reading assignment
	* light overview of theory
	* demonstration lecture using python
	* ML project assignment
	* Overview of Solution for Project

* Disclaimer: Machine Learning is Hard to Learn: Take your Time, Be patient

### Lecture 77 - Machine Learning w/ Python

* We will use the SciKit ML learning package. it's the most popular machine learning package for Python and has a lot of algorithms built-in
* we need to install it. using `conda install scikit-learn` or `pip install scikit-learn`
* we will install it in our plotly env `conda install -n plotly scikit-learn`
* in sccikit-learn every algorithm is exposed via an *Estimator*
* first we import the Model: `from sklearn.family import Model` for example for Linear Regression `from sklearn.linear_model import LiearRegression` 
* the second step is to instantiate the Model
* *Estimator parameters:* all the parameters of an estimator can be set when it is instantiated, and have suitable default values. we can see the possible params in Jupyter with shift+tab
* for example we can instantate the Linear Regresion estimator with: `model = LinearRegression(normalize=True)` setting the normalize param to True. if we print the instance `print(model)` we see all the default params of the model instance `LinearRegression(copy_X=True, fit_intercept=True, normalize=True)`
* once we have our model created with the desired params, we can fit it on some data
* but this data must be split in two sets (training set and test set)
* we import numpy for datasets `import numpy as np`
* we import train test split from scikit 	`from sklearn.cross_validation import train_test_split`
* we then create our dataset and a set of labels `X, y = np.arange(10).reshape((5,2)), range(5)`
* we split our data `X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)` which splits our data in a 3 to 2 ratio and randomizes the order
* with the data split, we can train/fit our model on the trianing data
* this is done with the model.fit() method `model.fit(X_train,y_train)`
* now the model has been fit and trained on the training data
* the model is ready to predict labels or values on the test set!
* the proces we follow is a SUPERVISED LEARNOING process, in unsupervised we dont have the labels
* we get the predictions with `predictions = model.predict(X_test)`
* we can evaluate our model by comparing our predictions to the correct values (y_test)
* the evaluation method depends on the the kind of machine learing algorithm we use (e.g Regression, Classification, Clustering etc)
* On all Estimators the model.fit() method is available. for supervised learning applications it accepts 2 arguments, data X and labels y e.g model.fit(X,y)
* For unsupervised learning apps, this accepts only a single argument the data X
* In All Supervised Estimators we have a model.predict() method: given a trained model , predict the label oif a new set of data, this method accepts one argument the test data X_test and returns a learned label for each object in the array
* for some supervised estimators the model.predict_proba() method is available. it is used for classification problems and returns the probability that a new observaTION has each categorical label. in this case the label with the highest probability is returned by model.predict()
* model.score() is offered for classificsation or regression problems. as most estimators implement a score method. scores are between 0 and 1. a larger score indicates a better fit
* model.predict() is also available in unsupervised estimators to predict labels in clustering algorithms
* model.transform() is available in unsupervised estimators: given an unsupervized (trained) model, transfor,s new data 9test data) in teh the new basis. it accepts X_new and returns the new representation of the data based on the unsupervised model.
* some unsupervised estimators implement the model.fit_transform() method which more efficiently performs a fit and a transform on the same input data

## Section 15 - Linear Regression

### Lecture 78 - Linear Regression Theory

* Linear REgression was devloped in 19th century by Francis Galton. 
* He was investigatiing the relationship between heights of fathers and sons
* He discovered that a mans son tended to be roughly as tall as his father.
* By he found that the sons height tended to be closer to the overall average of all people
* This phenomens was called regression, as a sons height regress (drifts towards) the mean average height
* all we are trying to do when we calculate the regression line is to draw a line that as close to every fdot in the dataset as possible.
* for classic linear regression (Least Squares Method) we measure the closeness in the 'up and down' direction
* we apply the same concept to multiple points
* THe goal in linear regression is to minimize the vertical distance between all the data points and our line.
* So to determine the best line we attempt to minimize the distance between all the points and their distance to the line
* there are many ways to do it (sum of squared errors, sum of absolute errors etc) but all these methods have a general goal of minimizing the distance
* the most common methods is least squares method (minimize the sum of squares of the residues)
* the residuals for an observation is the difference between the observation (y-val) and the  fitted line

### Lecture 80 - Linear Regression with Python Part 1

* we import split as `from sklearn.model_selection import train_test_split`
* we will start off by working with a housing data set trying to create a model to predict housing proces based off of existing features
* to ease our process we will work with artificially created datasets,
* latar on we will use real data sets from kaggle
* we import pandas, numpy pyplot from matplotlib, seaborn and set matplotlib inline
* we load our csv to a dataframe `df = pd.read_csv('USA_Housing.csv')`
* we have averaged value columns for the area the house is. area income, age,roums,bedrooms,population
* we have also the hoses price and address
* we chck our data with `df.info()`
* we can also check `df.describe()` to get statistical info about the data
* we can check our columns with `df.columns()`
* we should get a better insight on the data, so we pairplot the table `sns.pairplot(df)`. we get histograms on the columns and also correlation scatterplots. se see that bedrooms and rooms are segmented as the have integer values (represented as floats)
* we want to predict the price of the house do we do a distribution plot `sns.distplot(df['Price'])`. we see that kde peaks a t 1.2M and distributes evently around it
* we also want an insight of the correlation between columns so we create a heatmap `sns.heatmap(fd.corr())` we immediately see what affects the price.
* we have sufficient insight to start building our linear regression model
* First we have to split our data into an X array containing the features to train on, and a Y array with the target variable (price). we will also delete the address column as we cannot process text. we do it manually setting the columns
```
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
```
* with features and target arrays ready, we now have to split our data into training and test sets. we import the module and do the split specifying the test size. we can see the train_test_split params with Shift+Tab. we use 40% for test data. random state is used to seed the split as it is done randomly. so selection of test and train data is random.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```
* with data ready we need to train our model 
* we first import the Estimator `from sklearn.linear_model import LinearRegression`
* then we instantiate our model (estimator) `lm = LinearRegression()`
* we check the avaialble methods in our model (Shift+tab). we choose to train our model passing the train data `lm.fit(X_train,y_train)`
* we evaluate our model by checking its coefficients and see how we can evaluate them `print(lm.intecept_)` => a number,  the coefficients show the relation to each feature in our data `lm.coef_`
* to see the coefficient mapping better we create a dataframe. `pd.DataFrame(lm.coef_,X.columns=['Coeff'])`
* What these coefficients mean? if i hold all other features fixed. a 1 unit increase of a feature will result to the relevant coefficient amount on the target. these are artificial data so they dont make much sense
* we can load real data from inbuilt sklearn datasets. 
* boston is a dictionary with some keys
```
from sklearn import load_boston
boston = load_boston()
boston.keys()
print(boston['DESCR']) # description
print(boston['data']) # data
print(boston['feature_names']) # column labels
print(boston['target']) # targeet [pricing]
```

### Lecture 81 - Linear Regression with Python Part 2

* we will now see how to get predicitons from our model
* we get the predictions from the model `lm.predict(X_test)`
* `predictions` are the predicted values while `y_test` contains the target test data. we need to compare these two. 
* we want to see the correlation so we create a scatter plot `plt.scatter(y_test, predictions)`. if they line up linearly with relative small jitter we know our prediction is good. a perfect straight line are perfectly correct predictions (unrelaistic)
* we will now do a ditribution of the residuals (difference between predictions and test values) `sns.distplot(y_test-predictions)`. they look normally distributed. this is a GOOD sign. it means we selected the correct model
* there are 3 evaluation metrics for regression problems:
	* mean absolute error (MAE) the  mean of absolute value of errors: easiest to understand (average error)
	* mean squared error (MSE) the mean of squared errors: more popular as it punishes larger errors (real world approach)
	* root mean squared error (RMSE) the square root of the means of the squared errors : more popular than MSE as it is interpretable to Y units (like MAE)
* all these are loss functions , our aim is to minimize them
* we can easily calculate them with sklearn. we import the module `from sklearn import metrics`
* from metrics we use the specialized functions
```
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

### Lecture 82 - Linear Regression Project

* 

## Section 16 - Cross Variation and Bias-Variance Trade-Off

### Lecture 84 - Bias Variance Trade-Off

* Bias Variance Trade Off is a fundamental topic of understanding our model's performance
* in Chapter 2 of ISL book  there is an indepth analysis
* the bias-variance trade off is the point where we are adding just noise by adding model complexity (flexibility)
* the training error goes down as it has to, but the test error is starting to go up
* the model after the bias trade-off begins to overfit
* imagine that the center of the target is a model that perfectly predict the correct values
* as we move away from the bulls-ey our predictions get worse and worse
* sometimes we get a good distibution of training data so we predict well and we are close to bulls eye, while somtimes our training data might be full of outliers or non-standard values rtesulting in poor predictions
* these different realizations result in a scatter of hits on the target (variance)
* we aim for low variance-low bias model (consistent low error predictions)
* high bias-low variace are consistent but off target predictions (high error)
* low-bias-highvariancea are non consistent on target predictions
* a commot temptation for beginners in ML is to continually add complexity to a model until it fits the training set very well. like a polyonymal fit curve matching vert well a set of training points
* doing this can cause the model to overfit (high bias) to our trainign data and can cause large errors on new data like the test set
* we will take a look at an example model on how we can see overfitting occur from an error standpoint using test data!
* we will use a black curve with some "noise" points off of it to represent the True shape the data follows
* we have some data points. we start with simplest linear fit, then quadratic then spline raising the complexity to achieve best fit
* we compare the MSE on test data and train data for these three cases drawing a line we see that the test-train MSE difference  is high  for linear average for quadratic and then starts to deviate high for test low for training data. 
* so initial we have high bias low variance (high error similar train-test) in the middle average (low bias low variance) and then  hisgh varianc e (big deviation) low bias (low MSA for training) . 
* we choose the average (bias-variance trade-off) under this we have underfitting and over it overfiting

## Section 17 - Logistic Regression

### Lecture 85 - Logistic Regression Theory

* Sections 4-4.3 of ISL book
* we want to learn about Logistic Regression as a method for Classification
problems solving
	* Spam versus Good Emails
	* Loan Default (Yes/No)
	* Diesease Diagnosis (Yes/No)
* All the above are examples of Binary Classification
* Classification can be Categorical
* In Linear Regression we saw regression problems where we try to predict a continuous value
* Although the name may be confusing, logistic classification allows us to solve classification problems, where we are trying to predict discrete categories
* the convention for binary classification is to have two classes 0 and 1
* we cannot use normal linear regression model on binary groups. it wont lead to a good fit
* sat we want to predict the probability of defaulting on a loan. the feature is the salary. the higher the salary the higher the probaility of paying back the loan 1.0 is the payback and 0.0 ois to default. outr train data are binary so weither 1 or 0. with linear fit we get bad fit. we might get <0 numbers which make no sense.
* we can transform our linear regression line to a logistic regression curve (s type) bound between 0 and 1.
* this curve is the plot of the Sigmoid (aka Logistic) Function which takes any vlaue and outputs it to be between 0 and 1. *f(z)=1/(1+exp^-z)*
* The SIgmoid Function takes any input. So we can take our Linear Regression Solution and place is into the Sigmoid Function. if our linear model is *y=b0+b1*x* our logistic model will be *p=1/(1+exp^(b0+b1*x))*
* this results in a probability from 0 to 1 of belonging in the 1 class. then we set a cutoff point usually at 0.5 anything above we consider it class  1 (True) and anything below class 0 (False)
* After we train our logistic regression model on some training data, we will evaluate our model's performance on some test data
* We can use *confussion matrix* to evaluate classification models
* e.g if we test for disease (Yes/No) we have a 2by2 matrix. the row will be Actual NO, Actual YES and the columns Predicted NO, Predicted YES
	* Predicted NO & Actual NO 		= True Negative (TN)
	* Predicted YES & Actual YES 	= True Positive (TP)
	* Predicted YES & Actual NO 	= False Positive (FP) Type I Error
	* Predicted NO & Actual YES 	= False Negative (FN) Type II Error
* The first Metric is Accuracy (TP+TN)/total
* Missclassification Rate or Error Rate (How often is it wrong) (FP+FN)/total   Accuracy = 1 - Error Rate
* We will explore Logistic Regression using the famous Titanic data set to attemp to predict wether a passenger survived based off of their features
* Then we will have a project with some Advertising data trying to predict if a customer clicked on an ad

### Lecture 86 - Logistic Regression with Python Part 1

* titanic is a typical beginners se t for classification
* [Kaggle](https://www.kaggle.com/) hosts datasets for training
* Competition offers challenges for some sort of price
* we import all libs (numpy,pandas,matplotlib.pyplot,seaborn)
* we load training data `train = pd.read_csv('titanic_train.csv')` in a panda dataframe
* we check its columns and see there is a survived column that is our target, passenger class,gender,age,sibsp (siblings/spouses aboard)mparch (parents or children aboard), ticket id, ticket fare. cabin, port of embarkation
* we will start investigating our dataset (exploratory data analysis)
* we wull look for missing data with `train.isnull()` which outputs a bolean dataframe which we can feed in a heat map to find missing data `sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')`. we remove tickdata and colorbar. the results are revelatory. ~20% of age data is missing and ~80% of cabin data.
* 20% of missing data is replaceable. at 80% we drop the column, or make it boolean (cabin known/not known)
* its always good to see our target data in depth. we start with a simple countplot `sns.countplot(x='Survived',data=train)` we add a hue based on sex to the countplot. females had much hihgher survival rate than males (we expect to see it in coefficients). we also add ahue based on Pclass. again its revelatory. peopl from 3rd calss had much lower survival rate
* we do a distplot on a numerical value age, removing the nulls `sns.distplot(train['Age'].dropna(),kde=False,bins=30)`. we have a binomial plot. a number of small children then a gap and then the peak of people of young age.
* we explore columns one by one trying to get an understanding of our data set. w ego to SIbSp and do a count plot as it is discrete (int) and few. `sns.countplot(x='SibSp',data=train)`
* we look into fare. we do a pandas histplot train['Fare'].hist(bins=40) fares lean hevily on the cheap side
* we can do an interactive cufflinks plot on fare
```
import cufflinks as cf
cf.go_offline()
train['Fare'].iplotkind="hist",bins=30)
```

### Lecture 87 - Logistic Regression with Python Part 2

* After exploring our data, we will clean them, turn them in an acceptable form for a machine learning algorithm
* we will fill the missing age data with the mean average age of the dataset. this is a common practice. or we can do it more smartly by passing in the average age per passenger class as we guess there will be a correlation between class and age. to investigate it we do a boxplot `sns.boxplot(x='Pclass',y='Age',data=train)`. we see that paseengers in 3class tend to be older than 2nd or 3rd class
* we will create a function to use it to fill in the data (impute)
```
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass == 1:
			return 37
		elif Pclass == 2:
			return 29
		else:
			return 24 
	else: 
		return Age
```
* we will apply this function `train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)`
* we do again the heatmap to confirm correct impute
* we decide to drop the Cabin column completeley along column `train.drop('Cabin',axis=1)`
* with very few missing data left we drop them from dataframe `train.dropna(inplace=True)`
* our data are clean
* we will now turn categorical features to  dummy data (eg. male/female to 0/1) or the Embarked city first letter to nums. we need to transorm these to numbers as machine learning algorithms work with numbers
* we do it using the pandas get_dummies method `pd.get_dummies(train['Sex'])` this produces 2 mutualy exclusive columns. these are perfect predictor of the orher one if male is 1 female is 0. this is an issue called multicolumniarity. this messes up the algorithm so we drop females column with `sex = pd.get_dummies(train['Sex'],drop_first=True)`  and keep the male column as a dataframe
* we follow a similar approach for Embarcked column `embark = pd.get_dummies(train['Embarked'],drop_first=True)` and we concat the tables `train = pd.concat([train,sex,embark],axis=1)`
* we will ingore text based columns when we do our train/test data `train.drop(['Sex','Embarked','Name',Ticket'],axis=1,inplace=True)` 
* we notice the passenger id is an index not useful for machine learning so we drop this. `train.drop(['PassengerId'],axis=1,inplace=True)` 
* now our data is ready for machine learning algorithm
* Pclass column is categorical class column (1,2,3). w ecould do pd.get_dummies on that column (good for ML). we can do it later to see how the algorim will react on using dummy data or keeping categorical data.

### LEcture 88 - Logistic Regression with Python Part 3

* in our folder we have two csv files atraining ans test file. we have been working with the train file sofar and we will use it to get training and test data
* we should clean the test file and use it
* we make the X and y datasets
```
X = train.drop('Survived',axis=1)
y = train['Survived']
```
* we import test_train_split from scikit `from sklearn.model_selection import test_train_split` and do the split in a 30/70 ratio `X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)`
* we import the Estimator Model `from sklearn.linear_model import LogisticRegression`
* we create an  estimator model instance `logmodel = LogisticRegression()`
* we train the model passign the train data leaving default params `logmodel.fit(X_train,y_train)`
* we generate our predictions passing the test data `predictions = logmodel.predict(X_test)`
* we are now reday to evaluate the results.
* scikit learn has a handy tool for classification model evaluation rport tool
* we import it `from sklearn.metrics import classification_report`
* we use it wrapping it with print and passing actual test values and predictions `print(classification_report(y_test,predictions))`
* it return some metrics for both results. precision,recall,f1-score and support
* we can get the pure confusion matrix with 
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
```
* we have 81% accuracy. some ways of improving our scores is increase our training set (use both files),another way is grab feats of the name e.g title, cabin number

### Lecture 89 - Logistic Regression Project

* a good step in exploration is to make a pairplot of our dataset with hue on the target column

## Section 18 - K Nearest Neighbors

### Lecture 91 - KNN Theory

* Theory in ISL ch4. 
* K Nearest Neighbors is a *Classification* algorithm that operates on a very simple principle
* we will use an example to show it
* imagine we had some imaginary data on dogs and horses, with heights and weights
* if we do a jointplot between weight and height of dogs,horses we will see the linear correlation. if we add a hue on the plot baed on kind (horse/dog) we see that dogs are at the low end and horses on the high end. so from a weight/height datapoint it is quite easy to say with accuracy if it is a  dog or a horse
* The KNN algorithm woirks as follows
	* The training algorithm simply stores all data
	* The prediction algorithm calculates the distance of a new data point from all points in our data, it sorts the points in our data by ascending distance from x, it calulates the majority label of the K nearest points and sets it a s the prediction
* k is critical on the algorithm behaviour as it will directly affect what class a new point is assigned to
* chosing 1 picks a lot of noise, chosing a large k creates a bias
* KNN Pros:
	* Very Simple
	* Training is Trivial
	* Works in any number of Classes
	* easy to add more data
	* few params (K, Distance metric)
* KNN Cons: 
	* High Prediction Cost (increases as dataset gets larger)
	* Not good with high dimensional data (distances to many dimensions)
	* Categorical features dont work well
* A common interview task for a data scientisc position is to be given anonymized data and attempt to classify it. without knowing the context of the data.
* We will simulate this scenario using some "classified" data, where what the columns represent is not known but we have to use KNN to classify it

### Lecture 92 - KNN with Python

* we import the libraries
* we read in the classified data csv into df
* it has random column names with numbers and a target class of 1 or 0
* scale of data plays a big role in distance so it affects KNN. what we have to do as preprocess of our data is to standardize the variables rescaling them to the same scale. sklearn has a tool for the task. we impor it `from sklearn.preprocessing import StandardScaler` we instantiate it `scaler = StandardScaler()` and train it or *fit* it to our data (only on the numeric columns, not the target column which is categorical/binary) `scaler.fit(df.drop('TARGET CLASS', axis=1))`
* then we use the trained scaler object to get the scaled features `scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))`
* scaled_features is actually an array of values. we use this array to recreate a features table which we can use in our algorithm `df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])` we set as column names the column names of the original table excluding the last one (targets)
* our dataset is ready
* we import data splitter from sklearn and use it to split our data `X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30)`
* we are ready to use the algorithm. we impor the Model `from sklearn.neighbors import KNeighborsClassifier`
* we instantiate it passing k=1 (num of neighbors) `knn = KNeighborsClassifier(n_neighbors=1)` and fit it on our train data `knn.fit(X_train,y_train)`
* we grab the predictions `pred = knn.predict(X_test)`
* we import and use classification_report and confustion_matrix
* our results are good enough but we will use the embow method to pick a good k value
* the elbow method essentially reruns the knn algorithm for different k and store the mean error, then we plot the error vs k and find the best k
```
error_rate = []
for i in range(1,40)
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test)) # average of where the predictions where not equal to test values
```
* we plot the error rate to the k
```
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```
* we look at the error rate. we stabilize around 20 at the low  range so we run for 17 and get our reports accuracy and precicion are better

Section 19 - Decision Trees and Random Forests

### Lecture 95 - Introduction to Tree Methods

* chapter8 of ISL.
* We will see the rationale behind the Tree Methods by an example.
* Say we paly Tennis with a friend every Saturday. sometimes the friend shows up, sometimes not. For him, it depends on a variety of factors like: weather, temperature, humidity, wind etc
* We keep track of these feats and whetther we showed up to play or not createing a dataset
* We want to use this data to predict whether or not he will show up to play.
* An easy way to do it is through a Decision Tree
* in this tree we have : 
	* nodes. nodes split for the value of a certain attribute
	* edges: outcome of a split to next node
	* root: the node that performs the first split
	* leaves: terminal nodes that predict the outcome
* the Intuition behind Splits is simple. 
* we start bybuilding  a truth table of all feats as columns, with its result being the target value. we write down all combinations anthe outcome. in a simple 3 feat example with a boolean output we see that the output has 1:1 relationship with Y, so the tree has 1 root node (y) and two leaves (outputs). we say that Y gives perfect separation between classes
* other feats (X,Z) dont split classes perfectly. so if we split on these feats first we dont get good separation.
* Entropy and Information Gain are the Mathematical Methoids of choosing the best split.
	* Entropy: *H(S) = -Σi(pi(S)*log2pi(S))*
	* Information Gain: *IG(S,A)=H(S)- Σ u belongsto Values(A)(|Su|/S*H(Su))*
* Aside for the mathematical foundation the real intuition behind the splits is trying to choose the feat that best splits the data (trying to maximize the information gain of the split)
* Random Forest is a way to Improve performance of single decision trees.
* The main weakness of decision trees is that they dont have the  best predictive accuracy
* This is mainly because of the high variance (different splits in the data can lead to very different trees)
* Bagging is a general purpose procedure for reducing the variance for a ML method
* We can build up the idea of Bagging by using Random Forests, Random Forest is a slight variation of this Bag of Trees that has even better perfirmance.
* We will discuss it and code it in R (!?!)
* What we do in Random Forest is we create an ensemble of Trees using Bootstrap samples of the Training Set.
* Bootstrap samples of a training set means sampling from the dataset with replacement.
* However we are builidng each tree, each time a split is considered, a random sample of m features is chosen as a split candidate from the full set of p features.
* The split is only allowed to use one of these m features
* A new random sample of features is chosen fo revery single tree at every single split
* For classification this m random sample of m features is typically chosen to be the square root of p, where p is the full set of features
* WHY TO USE RANDOM FORESTS???
* suppose there is one very strong feature in the dataset, strong in predicting a cetrtain class. When using "bagged" trees (bootrap sampling), most of the trees will use that feature as the top split, reszulting in an ensemble of similar trees that are highly correlated. this is something we want to avoid 
* averaging highly correlated quantities does not significantly reduce variance
* By randomly leaving out candidate features from each split, random forests "decorrelates" the trees, sych that the averaging process can reduce the variance of the resulting model
* with that process, features that really strongly predict the class data do not affect the result
* We will take a look at at asmall data set of Kyphosis patients and try to predict if a corrective spine surgery was successful.
* for our exercise project we will use loan data from Lending CLub to predict default rates

### Lecture 96 - Decision Trees and Random Forest with Python

* [BlogPost](https://towardsdatascience.com/enchanted-random-forest-b08d418cb411)
* import libraries
* we load our dataset from a csv ('kyphosis.csv') which shows the result of the corrective operation on children. the features are: age in months, number of spinal disks operated, and the index of the first spinal disk operated. the target is a boolean (successuful or unscucessful)
* the data set is small (81 samples)
* we do exploratory analysis on the data
* we do a pairplot with the target as hue. `sns.pairplot(df,hue='Kyphosis')`
* we import the splitter method and split our dataset on train and test data and results (droping the Kyphosis column)
* we start by testing a single decision tree.
* we import the estimator `from sklearn.tree import DecisionTreeClassifier`
* we instantitate the model `dtree = DecisionTreeClassifier()`
* we train it with data using default params `dtree.fit(X_train,y_train)`
* we ge the predicitions `predictions = dtree.predict(X_test)`
* we evaluate the results: import and print classification re port and confusion_matrix. results are rather good resustsl
* we will now use random forest:
* we import the estimator `from sklearn.ensemble import RandomForetClassifier`
* we instnatiate it seting the number of estimators `rfc = RandomForestClassifier(n_estimators=200)`
* we train, predict and evaluate the model. we have an improvemnt.
* as dataset gets larger random forest gets better results
* scikit has visualisation for decision trees. we wont use it often as we will use random forests which get better results.
* this plot comes in a lib called pydot that we have to install `pip install pydot`
* we also need the graphviz library. [graphviz](https://www.graphviz.org/) is a separate program altogether `sudo apt-get install graphviz`
```
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 
```
* Random forest is the baseline of ML

### Lecture 97 - Decision Trees and Random Forest Project

* we will use public available data from LendingClub regarding peer-to-peer credit
* our aim is to use not.fully.paid column as taget and predict if the loan was flly paid back or not
* they ask to plot a histogram of FICO score with hue based on categorized columns (hue) using pandas. doing hue withpanda viz is done like this
```
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='red',label='Credit Policy 1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='blue',label='Credit Policy 0')
plt.legend()
plt.xlabel('FICO')
```

## Section 20 - Support Vector Machines

### Lecture 100 - SVM Theory

* chapter9 of ISL
* Support vector machines (SVMs) are supervised learning models with associated learning algorithms tha analyze data and recognize patterns, used for classification and regression analytics
* Given a set of training examples, each marked for belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples into one category or the other, making it a non-probabilistic binary linear classifier
* An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible.
* New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall on
* to show the basic idea behind SVMs we imagine a labeled training data set where the hued(target) joint scatterplot of 2 feats are clearly separated between target classes.
* we want to do binary classification of new points, we can easily draw a separating "hyperplane" between the classes. but there are many options for separating lines (hyperplanes) that separate classes perfectly. how do we choose the best? usually we have one going along the middle of the distance and two parallel ones on the border(margin) of each class.
* The vector points that these margin lines touch are known as Support Vectors.
* We can expand this idea to non linearly separated data through the "kernel trick", the lines might be circles,
* the kernel trick is adding one more dimension to the plot so he can separate them in the 3rd dimension
* we will use support vector machines to predict whether a tumor is malignant or bening
* for our project we will apply the concept to the famous iris flower data set
* we will learn how to tune our models withthe GridSearch

### Lecture 101 - Support Vector Machines with Python

* we import the libs
* we import a scikit learn builtin dataset (Breast cancer) `from sklearn.datasets import load_breast_cancer` and load it to a dictionary `cancer = load_breast_cancer()`
* we see the feats as dictionary keys `cancer.keys()`
* we get info on the origin of the dataset `print(cancer['DESCR'])`
* we build a dataframe from the dicitonary `df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])`
* we check the labels with `df_feat.head(2)` and `df_feat.info()`
* we have 30 cols of medical data. we lack of domain knowledge so we skip visualization
* we build our target df `df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])`
* we go to ML directly. we split the data `X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)`
* we import the estimator `from sklearn.svm import SVC`
* we instantiate the model `model = SVC()`
* we train the model using defaults, we do the prediction and evaluate the results. we get poor results and a Warning: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
* What we see that the model classifies all data to a single  class. so it needs adjustment and probably normaization of the data prior to use
* We will search for the best parameters using a GridSearch, it helps us find the right paramters (c or gamma)
* To skip testing and searching for the right params we will use the grid of parameters to try out all the best possible combinations and see what fits best.
* we import it `from sklearn.grid_search import GridSearchCV`
* GridSearchCV takes in a dictionary that describes the parameters that should be tried in a model to train the grid., the keys are the parameters and the values are a list of settings to be tested.
* C param controls the cost of misclassification on the training data (a large C value gives low bias and high variance). it gives low bias because we penalize the cost of missclassification with a larger C value.
* the gamma parameter has to do with the free parameter of the Gaussian radial basis function (kernel='rbf'). this function is the best kernel to use. small gamma mean as Gaussian for large variance. high gamma leads to high bias low variance in the model
* we set the param_grid passing the params to test and the range of values as a dictionary `param_grid = { 'C':[.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001] }`
* we feed the param grid to the gridsearch `grid = GridSearch(SVC(),param_grid,verbose=3)`
* versbose number controls the amount of text
* we pass the estumator,the param grid and config params
* the output is a estimator model instance we can use to tainr on out training data.
* it finds a best combination and iterates on this to get the best score
* we gan get the best params with `grid.best_params_`
* we gan get the best estimator with `grid.best_estimator_`
* we gan get the best score with `grid.best_score_`
* the *grid* has the best estimator ready to use. so we use it to get the best predictions `grid_predictions = grid.predict(X_test)`
* we print the eval reports. the improvent is dramatical

### Lecture 102 - SVM Project

* we will work with famous iris flower dataset. only 50 samples, 3 samples
* to display image in jupyter notebook:
```
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
```

## Section 21 - K Means Clustering

### Lecture 104 - K Means Algorithm Theory

* the K Means Clustering algotithm will allow us to cluster unlabeled in an unsupervised machine learning algorith
* mathematical explanation in ch10 of ISL
* K Means Clustering is an unsupervised learning algorithm that will attempt to group similar clusters together in your data
* A typical clustering problem:
	* Cluster similar documents
	* Cluster Customers based on Features
	* Market segmentation
	* Identify similar physical groups
*  The overall goal is to divide data into distinct groups such that observations within each group is similar
* The K Means Algorithm:
	* Chose a number of Clusters "K"
	* Randomly assign each point to a cluster
	* Untill clusters stop changing, repeat the following: For each cluster, compute the cluster centroid by taking the mean vector of points in the cluster. => Assign each data point to the cluster for which the centroid is the closest
* Choosing a K value: we have to decide how many clusters we expect in the data
* There is no easy answer for choosing the best "K" value
* One way is the elbow method
	* Compute the sum of squared error (SSE) for some values of k (for example 2,4,6,8)
	* The SSE is defined as the sum of the squared distance between each member of the cluster and its centroid
* If we plot k against the SSE, we will see that the error decreases as k gets larger. this is because when the number of clusters increases, they should be smaller
* the idea of the elbow is to choose the k at which the SSE decreases abruptly
* this produces an elbow effect int the graph (increasing k does not improve SSE so much any more)
* in our project we will get real world data and try to cluster unis into groups, we will try to distinguish private from public

### Lecture 105 - K Means with Python

* we import the usual libraries (data analyzis + data viz)
* in unsupervised lewnring we are not focused on results but on patters in the data
* we import sklearn inbuilt data sets (blobs of data) `from sklearn.datasets import make_blobs`
* we create a dataset for our algo with 200 samples from blob generator passing some parameters to control the distribution of our samples. `data = make_blobs(n_samples=200, n_features=, centers=4, cluster_std=1.8,random_state=101)`
* with data in hand we will explore them. data is a tuple and data[0] is a numpy array. this numpy array is a number of columns and 2 columns of dfeatures `data[0].shape` => (200,2) 200samples with 2 features per sample
* we have defines 4 centers in our data so we  expect to see 4 blobs
* we plot a scatterplot of the 2 feat columns `plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')` with color set on the samples cluster group as it was outputed by make_blob
* we import our estimator `from sklearn.cluster import KMeans`
* we create an instnace of our model. we need to define the number of clusters we want (we know it) `kmeans = KMeans(n_clusters=4)` 
* we train the algorithm passing a numpy array (unlike supervised algs) `kmeans.fit(data[0])`
* we get the centers (x,y) of the clusters on the 2 feat defined space. as a numpy array of nested arrays
* we get the lables the agorithm beleaves our samples should have regarding the clusters they belong `kmeans.labels_` whic is a numpy array
* if we work with real data and we dont know beforehand the labels and the actual clustering our work is done. but now we know the actual clustering from the blob generator output
* we will plot the predicted clustering to the actual clustering, we plot 2 plots in one sharing y axis
```
fig, (ax1,ax2) =  plt.subplots(1,2,sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow)
ax2.set_title('Original')
ax1.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
```
* the plots are look alikes so kmeans did a good job. if we try kemans with 2 or 3 clusters the output is very rreasonable too

### Lecture 106 - K Means Project

* EDA means (exploratory data analysis)
* to set a specific value in a dataframe `df.set_value('95','Grad.Rate',100)`. this creates empty rows with index integer. we try `df.iloc[95, df.columns.get_loc('Grad.Rate')] = 100`

## Section 22 - Principal Component Analysis

### Lecture 108 - Principal Component Analysis

* math in ch10.2 of ISL
* PCA is an unsupervised statistical technique used to examine the interrelations among a set of variables in order tp identify the underlying structure of those variables
* It is also known sometimes as a general *factor analysis*
* Where regression determines a line of best fit to a data set, factor analysis determines several orthogonal lines of best fit on the data set
* Orthogonal means "at right angles"
	* These lines are perpendicular to each other in n-dimensional space
* n-Dimensional Space is the variable sample space
	* there are as many dimensions as there are variables, so in a data set with 4 variables the  sample space is 4-dimensional
* To understand the intuition behind it we plot some data alonf 2 features in a scatterplot.
* we plot the regression line of best fit
* we add an orthogonal line to the regression line.
* now we can start to understand the components of our dataset
* *Components* are a linear transformation that chooses a variable system for the data  set such that the greatest variance of the dataset comes to lie on the first axis, the second greatest variance on the second axis and so on...
* This process allows us to reduce the number of variables used in an analysis
* in our plot the regression line "explains" 70% of the variation, the orthogonal line "explains" the 28% of the variation. 2% of the variation remains unexplained
* Components are uncorrelated, since in the sample space they are orthogonal to each other
* we can continue this analysis into higher dmensions although its impossible to plot over the 3rd dimencion
* If we use this technique on a data set with a large number of variables, we can compress the amount of explained variation to just a few components
* The most challenging part of PCA is interpretting the components
* For our work with PCA in Python, we will use scikit learn
* We usually want to standardize our data by some scale for PCA, so we will see how to do it
* The algorithm is used for analysis of data and not a fully deployable model, we wont have a project for the topic

### Lecture 109 - PCA with Python

* PCA is a unsupervised learning algorithm used for component reduction
* PCA is just a transformation of our data and attempts to find out what features explain the most variance in our data
* we import the usual libs (data analysis and vis)
* we load sklearn builtin datasets breast cancer data `from sklearn.datasets import load_breast_cancer` and `cancer = load_breast_cancer()` it is a dictionary like object holding data and labels as key value pairs
* we make a dataframe out of it `df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])`
* the dataset has 30 attibutes
* if we used an other calssification algorithm we would do PCA first to see what feats are important in determining if a tumor is malignant or benign
* we will use PCA to find the 2 principal components and then visualize data inthe 2 dimensional space
* we first scale our data `from sklearn.preprocessing import StandardScaler`
* we will scale our data so every feat has 1 unit variance
* we make a scale rinstance `scaler = StandardScaler()`
* we fit it to our dataframe `scaler.fit(df)`
* we transform our df with scaler `scaled_data = scaler.transform(df)`
* we perform now the PCA. we import it `from sklearn.decomposition import PCA`
* we instantiate it passing the num of components we want to keep `pca = PCA(n_components=2)`
* we fit pca to our scaled data `pca.fit(scaled_data)`
* we transform the data to its first principal component `x_pca=pca.transform(scaled_data)`
* our original scaled data shape is `scaled_data.shape` => (569,30)
* our decomposed data `x_pca.shape` => (569,2)
* we plot out these dimensions
```
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
```
* it is difficult to understand the usefuless of this plot, we add a color depeding on target class `plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])` we see the clustering in full extent. 
* based only on the first and second principle component we have aclear separation
* PCA works like a compression algorithm in machine learning
* these components we generate do not map 1to1 to specific feats in our data
* components are combinations of the original features
* if we print the array of components with `pca.components_` each row represents a principal component and each column relates it  back to the original features
* we can visualize this relationship with a heatmap to see the depnedence of a component on the feats
```
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
```
* The idea is to feed pca output to a classification algorithm reducing the complexity (logistic regression)

## Section 23 - Recommender Systems

### Lecture 110 - Recommender Systems

* reading textbook: Recommender Systems by Jannach and Zanker
* Fully developed and deployed recommendation systems are extremely complex and resource intensive
* there are 2 notebooks for this course. a) Recommender Systems w/ Python b) Advanced Recommender Systems w/ Python
* Full recommender systems require a heavy linear algebra background (WE HAVE IT!!) the Advanced Recommender System notebook is provided as an optional resource
* A simpler version of creating a recommendation system using item similarity is used for the course example
* The two most common types of recommender systems are *Content Based* and *Collaborative Filtering (CF)*
	* Collaborative Filtering produces recommendations based on the knowledge of users attitude to items, that is it uses the "wisdom of the crowd" to recommend items
	* Content-based recommender systems focus on the attributes of the items and give you recommendations based on the similarity between them
* In real world, Collaborative filtering (CF) is more commonly used than content-based systemsbecause it usually gives better results and is relatively easy to understand (From an implementation perspective)
* THe algorithm has the ability to do feature learning on its own, which means that it can start to learn for itself what features to use
* CF can be divided into two subcategories: *Memory-Based Collaborative Filtering* and *Model-Based Collaborative Filtering*
* In the advanced notebook, we implement Model-Based CF using a singular value decomposition (SVD) and Memory-Based CF by computing cosine similarity
* For the simple Python implementation, we will create a content based recommender system for a data set of movies
* This movie data set is usually a student's first data set when beginning to learn about recommender systems
* It is quite large compared to some of the data set we ve used so far. in general, recommender systems in real  life deal with much larger data sets

### Lecture 111 - Recommender Systems w/ Python Part 1

* we import the usual data analytics and vis libraries
* we create a list of the column names `columns_names=['user_id','item_id','rating','timestamp']`
* we load the data from csv `df = pd.read_csv('u.data',sep='\t',names=columns_names)` which is tab separated. we also pas the column names as csv is just raw data. 
* the data set we go through is a famous one called movielens dataset. which contains user ratings for movies
* we grab movie_titles from another csv `movie_titles=pd.read_csv('Movie_Id_Titles')`. this dataset maps item ids to movietitles. we will replace item-ids in the master dataset with the actual titles doing a merge `df=pd.merge(df,movie_titles, on='item_id')`.the merged dataframe contains id and titles now
* we will show the best rated movies. to do this we will create a rating dataframe `df.groupby('title')['rating'].mean()` calculating the avearge rate for each title. we can sort them now in ascending order `df.groupby('title')['rating'].mean().sort_values(ascening=False).head()`
* the info might be misleading as the average score might be due to only one review. so we also count the votes `df.groupby('title')['rating'].count().sort_values(ascening=False).head()`
* we make a new dataframe called ratings with the titles and the ratings `ratings = pd.DataFrame(gf.groupby('title')['rating'].mean())`. 
* we add a num of rating column next to the ratings `rating['num of rating'] = pd.DataFrame(df.groupby('title')['rating'].count())`
* we explore the new ratings dataset with some histograms. `ratings['num of ratings'].hist(bins=70)` most titles have very few ratings
* we do a histogram of the average ratings per title. there are peaks in whole nums (as titles get  whole num rates per review) `ratings['rating'].hist(bins=70)`
* we use jointplot to see distribution of ratings vs nums `sns.jointplot(x='rating',y='num of ratings', data=ratings, alpha=0.5)`

### Lecture 112 - Recommender Systems w/ Python Part 2

* we will create a matrix that has user ids in one axis and movie titles on other axis. each cell will have the rating the user gave to the movie, we will use pivot to bring df in matrix form `moviemat = df.pivot_table(index='user_id',columns='title',values='rating')`. this matrix has a alot of null values
* we check the top ten rated movies `ratings.sort_values('num of ratings',ascending=True).head(10)`
* we choose 2 titles: Star Wars and Liar Liar and make datasets with their user ratings.
```
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
```
* the two dataframes have user_id rows with the ratings they gave
* we will use the corrwith() method yo vompute the pairwise correlation between rows or columns of two Dataframes `similar_to_starwars = moviemat.corrwith(starwars_user_ratings)` we get  all the movies and their correlation with the starwars movie rating
* what we are looking for is the correlation of every other movie to the user behavior on the star wars movie
* we do the same for liarliar `similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)`
* we transform them to dataframes erasing the nuls 
```
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
```
* what we end up is a dataframe with all the movies and a valuie that shows the correlation of their user rating to the user ratings of StarWars. what we want to get are the most correlated. the more similar movies `corr_starwars.sort_values('Correlation',ascending=False).head(10)`. we get movies with perfect correlation. these movies in our opinion dont make sense. probably are movies whit just 1 review from a person who gave star wars 5 stars. so we would want to filter out movies with less that a threshold of reviews. like 100.
* we join the table with the num of rATINGS COLUMN `corr_starwars = corr_starwars.join(ratings['num of ratings'])` we filter out movies with <100 revies and sort it again by correlation `corr_starwars[corr_starwars['num of ratings']> 100].sort_values('Correlation',ascending=False).head(10)`
* the results now do make sense we get perfect correlation with the same movie. we get goo d correlation with similar movies and then the correlation drops significantlky
* now we do the same drill for liarliar
```
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar .dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head(10)
```
* we can play with threshold to improve recommandations relevance

## Section 24 - natural Language Processing

### Lecture 113 - Natural Language Processing Theory 

* Read [Wikipedia Article](https://en.wikipedia.org/wiki/Natural_language_processing) for theory 
* NLP serves a lot of purposes when working with structured or unstructured text data
* use cases: group news articles by topic, sift through 1000s of legal documents to find relevant ones
* With NLP we would like to: Compile Documents, Featurize Them, Compare their features
* Say we have 2 docs:
	* "Blue House"
	* "Red House"
* A way to featurize docs is based on word count (transform them as vectorized word count)
	* "Blue house" => (red,blue,house) => (0,1,1)
	* "Red house" => (red,blue,house) => (1,0,1)
* A document represented as a vector of word countsis called a "bag of words"
* we can use cosine similarity on teh vectors made to determine similarity *sim(A,B)=cos(θ)=(AdotB)/(|A||B|)*
* we can improve on Bag of Words by adjusting word counts based on their frequency in corpus(the group of all dosuments)
* we can use TF-IDF (Term Frequency-Inverse Document Frequency) tod o this
* This technique is used in ElasticSearch
* Term Frequency: Importance of a term within that document
	* TF(d,t) = Number of occurences of term t in document d
* Inverse Document Frequency: Importance of the termn in the corpus
	* IDF(t)=log(D/t) where D=total num of documents, t=number of documents with the term
* Mathematically we can express TF-IDF as: Wx,y=TFx,y * log(N/dfx)
	* TF-IDF = term x within document y
	* TFx,y = frequency of x in y
	* dfx = number of documents containing x
	* N = total number of documents
* TF-IDF contains the idea of importance of a word
* before we get started with NLP and Python we'll need to download an additional library, `conda install nltk` or `pip install nltk` we will do it  in plotly env
* our example notebook will be on a spam filter, our real project will be working with real data from Yelp, a review site

### Lecture 114 - NLP with Python Part 1

* we `import nltk`
* we run `nltk.download_shell()` which runs an interactive shell in jupyter to  select options
* we type *l* and see the list of packages
* we type *d* and then *stopwords* to download the stopwords library
* we will use a UCI dataset called SMS Spam collection DataSet . we have it in the spamcollection folder
* we use list comprehension to load the data in a list `messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]` and print its length `print(len(messages))` it has 5574 messages
* we will iterate through the 10 first messages to see their context
```
for mees_no,message in enumerate(messages[:10]):
	print(mess_no,message)
	print('\n')
```
* messages are labeled as spam or ham with the word then tab. we can check it with `messages[0]`
* we will use pandas to easily parse the message in dataset `messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t', names=['label','message'])` using the tab separation and adding labels
* We will now do EDA on our data. we start with `messages.describe()` which revails that we have quite a few duplicate messages 
* we will group by label and then describy to see how feats differentate between two classes `messages.groupby('label').describe()`
* our bulk of work with NLP is features engineering. to extract feats we need domain knowledge. the better we know the dataset the more insight we can get
* we check the length of the text messages and add it as a new column `messages['length'] = messages['message'].apply(len)`
* we will visualize the lengths of the messages in a histogram `messages['length'].plot(bins=50, kind='hist') ` as we increase the bins we get bimodal behaviour
* weget some insight in length `messages.length.describe()` the max length is 910
* its quite wierd so we opt ot look into it `messages[messages['length'] == 910]['message'].iloc[0]` this is clear an outlier
* we now want to plot the histogram of length but for both labels separated `messages.hist(column='length',by='label',bins=50,figsize=(12,4))` the kdes are clearly different

### Lecture 115 - NLP with Python Part 2

* our effort now will focus on text preprocessing (vectors)
* we `import string` library
* our first task is to remove punctuation
* we create atest string `mess = 'Sample message! Notice: it has punctuation.'`
* `string.punctuation` has abunch of punctuation suymbols to be used for filtering
* we will use list comprehencion on teh string to clear the string from punctuation `nopunc = [c for c in mess if char not in string.punctuation]`. this is a punc free list of chars from our string. we reassemble it `nopunc = ''.join(nopunc)`
* our task is now to remove stopwords
* we import stopwords lib `from nltk.corpus import stopwords`
* we test it `stopwords.words('english')` prints out all english stopwords
* we split our string to list of words `nopunc.split()` and again use list conmprehencion to clear it from stopwords `clean_mess=[word for word in nopunc.split() if  word.lower() not it stopwords.word('english')]`
* we bundle up all in a funct to apply in our dataset
```
def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)  
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
```
* we apply it to 5 first rows for testing `messages['message'].head(5).apply(text_process)` and IT WORKS
* we could further normalize our dataset (remove single letter words)
* stemming isa very common way to continue processing our text  data (considers similar meaning works as unique). stemming needs a reference dictionary to do it. nltk has that
* we now proceed with vectorization of our data
* we will create our bag of words in 3 steps
	* Count how many times does a word occur in each message (Known as term frequency)
	* Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
	* Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
* our result table will be a matrix of messages vs words(unique) with a counter of word appearance in a message. the columns will be the word vectors or bags of words
* we expect to have a lot of 0s in our matrix. scikit learn will output a Sparse matrix
* we import CountVectorizer `from sklearn.feature_extraction.text import CountVectorizer` which accepts a lot of params
* we instantiate it passing our custom function as analyzer. we directly chain the fit method to fit in on our data `bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])`
* we want to see how many words our vocabulary contains `print(len(bow_transformer.vocabulary_))` it more than 11000
* our bow_transformer is ready to be used on our dataset
* we test it on a sample message. 
```
message4 = messages['message'][3]
bow4 = bow_transformer.transform([message4])
bow4.shape
```
* the vector has the indexes of the words and the num of occurences in the message.
* its shape is a vector spanning the full vocabulary
* to get the actual word in teh vocabulary by its id we use `print(bow_transformer.get_feature_names()[4073])` => U

### Lecture 116 - NLP with Python Part 3