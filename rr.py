#neccecary modules importing
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

#date reading
data = pd.read_csv('transaction.csv', parse_dates=['purchase_date', 'next_purchase_date'])

#data preprocessing
data.dropna(inplace=True)
encoder = LabelEncoder()
data['CustomerID'] = encoder.fit_transform(data['CustomerID'])
data['ProductID'] = encoder.fit_transform(data['ProductID'])

#creating target variable
target = data['next_purchase_date'] - data['purchase_date']

#data extraction
data['purchase_frequency'] = data.groupby(['CustomerID', 'ProductID'])[ 'purchase_date'].transform( 'count')

# Define the variable 'X' after adding the missing columns
data['average_purchase_value'] = data['amount'] / data['quantity']
data['days_since_last_purchase'] = (data['next_purchase_date'] - data['purchase_date']).dt.days
data['product_category'] = data['ProductID'].apply(lambda x: data['product_category'][x])
y = data['purchase_frequency']

X = data[['average_purchase_value', 'days_since_last_purchase', 'product_category']]

#data splitting
# Encode the categorical feature 'product_category'
encoder = OrdinalEncoder()
X.loc[:, 'product_category'] = encoder.fit_transform(X.loc[:, ['product_category']])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model performance
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
