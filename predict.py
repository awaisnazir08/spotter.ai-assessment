"""Trains the freight rate model and writes both prediction files."""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


train = pd.read_csv('train-test.csv')
validation = pd.read_csv('validation.csv')

train['date'] = pd.to_datetime(train['date'])
validation['date'] = pd.to_datetime(validation['date'])

# weights cannot be negative so taking the absolute value
train['weight'] = train['weight'].abs()
validation['weight'] = validation['weight'].abs()

weight_median = train['weight'].median()
market_index_median = train['market_index'].median()

train['weight'] = train['weight'].fillna(weight_median)
train['market_index'] = train['market_index'].fillna(market_index_median)

# rate per mile is the normal way to compare freight prices
train['rate_per_mile'] = train['posted_rate'] / train['distance']

# the log shows three separate distributions, the two side ones are bad data
train['log_rate_per_mile'] = np.log(train['rate_per_mile'])
train = train[(train['log_rate_per_mile'] > 0.4) & (train['log_rate_per_mile'] < 1.3)]
print('training rows after removing the bad ones :', len(train))


def make_features(data):
    features = pd.DataFrame()
    features['distance'] = data['distance']
    features['market_index'] = data['market_index'].fillna(market_index_median)
    features['weight'] = data['weight'].fillna(weight_median)
    features['equipment'] = data['equipment'].map({'Dry Van': 0, 'Reefer': 1, 'Flatbed': 2})
    features['pickup_lat'] = data['pickup_lat']
    features['pickup_lon'] = data['pickup_lon']
    features['delivery_lat'] = data['delivery_lat']
    features['delivery_lon'] = data['delivery_lon']
    features['quote_signal'] = data['quote_signal']
    return features


final_model = ExtraTreesRegressor(n_estimators=200, random_state=36)
final_model.fit(make_features(train), train['rate_per_mile'])
print('model trained')

os.makedirs('model', exist_ok=True)
pickle.dump(final_model, open('model/model.pkl', 'wb'))
print('saved model/model.pkl')

# predicting the validation loads
validation_predicted = final_model.predict(make_features(validation)) * validation['distance']

predictions = pd.DataFrame({'load_id': validation['load_id'],
                            'predicted_rate': validation_predicted.round(2)})
predictions.to_csv('validation_predictions.csv', index=False)
print('saved validation_predictions.csv :', predictions.shape)

# predicting the december chart, only the date changes in these rows
december = pd.read_csv('december-chart-inputs.csv')
december['date'] = pd.to_datetime(december['date'])

# the december file has no market index so taking daily average from validation
market_by_date = validation.groupby('date')['market_index'].mean()
december['market_index'] = december['date'].map(market_by_date)

# the december file has no coordinates either so getting them from the training data
lexington = train[train['pickup'] == 'Lexington'][['pickup_lat', 'pickup_lon']].iloc[0]
fort_wayne = train[train['delivery'] == 'Fort Wayne'][['delivery_lat', 'delivery_lon']].iloc[0]
december['pickup_lat'] = lexington['pickup_lat']
december['pickup_lon'] = lexington['pickup_lon']
december['delivery_lat'] = fort_wayne['delivery_lat']
december['delivery_lon'] = fort_wayne['delivery_lon']
december['quote_signal'] = train['quote_signal'].mean()

december['predicted_rate'] = (final_model.predict(make_features(december)) *
                              december['distance']).round(2)

# the scorer wants the original seven columns only
original_columns = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']
december_to_save = december[original_columns].copy()
december_to_save['date'] = december_to_save['date'].dt.strftime('%Y-%m-%d')
december_to_save.to_csv('december-chart-inputs.csv', index=False)
print('saved december-chart-inputs.csv :', december_to_save.shape)
