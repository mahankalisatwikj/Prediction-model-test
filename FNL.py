import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Read the data
data = pd.read_csv('transaction.csv', parse_dates=['purchase_date', 'next_purchase_date'])

# Preprocess the data
data.dropna(inplace=True)
encoder = LabelEncoder()
data['CustomerID'] = encoder.fit_transform(data['CustomerID'])
data['ProductID'] = encoder.fit_transform(data['ProductID'])

# Create the target variable
target = data['next_purchase_date'] - data['purchase_date']

# Extract features
data['purchase_frequency'] = data.groupby(['CustomerID', 'ProductID'])[ 'purchase_date'].transform( 'count')
data['average_purchase_value'] = data['amount'] / data['quantity']
data['days_since_last_purchase'] = (data['next_purchase_date'] - data['purchase_date']).dt.days
data['product_category'] = data['ProductID'].apply(lambda x: data['product_category'][x])
y = data['purchase_frequency']

X = data[['average_purchase_value', 'days_since_last_purchase', 'product_category']]

# Encode the categorical feature 'product_category'
encoder = LabelEncoder()
X.loc[:, 'product_category'] = encoder.fit_transform(X.loc[:, ['product_category']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Get the next expected purchase date for a customer
customer_id = 10
product_id = 5

# Get the customer's data
customer_data = data[data['CustomerID'] == customer_id]
product_data = customer_data[customer_data['ProductID'] == product_id]

# Calculate the average purchase value and days since last purchase
average_purchase_value = product_data['average_purchase_value'].mean()
days_since_last_purchase = (datetime.datetime.now() - product_data['purchase_date'].max()).days

# Get the product category
product_category = encoder.transform([product_data['product_category'].iloc[0]])[0]

# Create a new data point for prediction
new_data = pd.DataFrame({
    'average_purchase_value': [average_purchase_value],
    'days_since_last_purchase': [days_since_last_purchase],
    'product_category': [product_category]
})

# Make a prediction
next_purchase_frequency = model.predict(new_data)

# Calculate the next expected purchase date
next_purchase_date = product_data['purchase_date'].max() + datetime.timedelta(days=next_purchase_frequency[0])

# Print the result
print('Next expected purchase date:', next_purchase_date)
