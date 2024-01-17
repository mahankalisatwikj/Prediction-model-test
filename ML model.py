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

# Encoding the product_category
encoder = LabelEncoder()
X.loc[:, 'product_category'] = encoder.fit_transform(X.loc[:, ['product_category']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Get the next expected purchase date for a customer
customer_id = 7
product_id = 3

customer_data = data[data['CustomerID'] == customer_id]
product_data = customer_data[customer_data['ProductID'] == product_id]

average_purchase_value = product_data['average_purchase_value'].mean()
days_since_last_purchase = (datetime.datetime.now() - product_data['purchase_date'].max()).days

product_category = encoder.transform([product_data['product_category'].iloc[0]])[0]

# Create a new data point for prediction
new_data = pd.DataFrame({
    'average_purchase_value': [average_purchase_value],
    'days_since_last_purchase': [days_since_last_purchase],
    'product_category': [product_category]
})

next_purchase_frequency = model.predict(new_data)

next_purchase_date = product_data['purchase_date'].max() + datetime.timedelta(days=next_purchase_frequency[0])

print('Next expected purchase date:', next_purchase_date)
