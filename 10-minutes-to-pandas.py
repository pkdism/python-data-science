# 1. Import
import numpy as np
import pandas as pd


# 2. Creating a series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)


# 3. Creating a dataframe by passing a numpy array, with a datetime index and labeled columns
dates = pd.date_range('20130101', periods = 6)
print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))
print(df)


# 4. Creating a dataframe by passing a dict of objects that can be converted to series-like
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index = list(range(4)), dtype = 'float32'),
                    'D': np.array([3] * 4, dtype = 'int32'),
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'})
print(df2)


# 5. Columns of the dataframe have different dtypes
df2.dtypes


# 6. Viewing the top and bottom rows of a dataframe
df.head()
df.tail(3)


# 7. Display the index, columns
print(df.index)
print(df.columns)

# 8. NumPy arrays have one dtype for the entire array, while pandas DataFrames have one dtype per column.
# When you call DataFrame.to_numpy(), pandas will find the NumPy dtype that can hold 
# all of the dtypes in the DataFrame. This may end up being object, which requires casting 
# every value to a Python object.

# 9. describe() shows a quick statistic summary of your data
df.describe()


# 10. Transposing your data
df.T


# 11. Sorting by an axis and values
df.sort_index(axis = 1, ascending = False)
df.sort_values(by = 'B')


# 12. Selecting a single column, which yields a Series, equivalent to df.A
df['A']


# 13. Selecting via [], which slices the rows
df[0:3]
df['20130101':'20130104']


# 14. Getting a cross section using a label
df.loc[dates[0]]


#15. Selecting on a multi-axis by label
df.loc[:, ['A', 'B']]


#16. Showing label slicing, both endpoints are included
df.loc['20130102':'20130104', ['A', 'B']]


#17. Reduction in the dimensions of the returned object
df.loc['20130102', ['A', 'B']]


#18. Getting a scalar value
df.loc[dates[0], 'A']


#19. Getting fast access to a scalar (equivalent to the prior method)
df.at[dates[0], 'A']


#20. Select via the position of the passed integers
df.iloc[3]


#21. By integer slices and integer position locations, acting similar to numpy/python
df.iloc[3:5, 0:2]
df.iloc[[1, 2, 4], [0, 2]]


#22. For slicing rows, columns and value, each explicitly
df.iloc[1:3, :]
df.iloc[:, 1:3]
df.iloc[1, 1]


#23. For getting fast access to a scalar
df.iat[1, 1]


#24. Using a single column's values to select data
df[df.A > 0]


#25. Selecting values from a dataframe where boolean condition is met
df[df > 0]


#26. Using the isin() method for filtering
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2)
df2[df2['E'].isin(['two', 'four'])]


#27. Setting a new column automatically aligns the data by the indexes
s1 = pd.Series([1, 2, 3, 4, 5, 6], index = pd.date_range('20130102', periods = 6))
print(s1)
df['F'] = s1
print(df)


#28. Setting values by label, position and by assigning with a numpy array
df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5] * len(df))


#29. A where operation with setting
df2 = df.copy
df2[df2 > 0] = -df2
print(df2)


#30. Reindexing allows you to change/add/delete the index on a specified axis.
# This returns a copy of the data
df1 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1

# To drop any rows that have missing values or fill with a value
df1.dropna(how = 'any')
df1.fillna(value = 5)
pd.isna(df1)




































