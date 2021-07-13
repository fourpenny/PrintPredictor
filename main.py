#to do list: create additional features using .gcode file
#create ability to automatically pull data from octoprint instead of downloading from csv
#make into octoprint plugin that can be used straight from the server gui

import pandas as pd 
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

model = RandomForestRegressor(n_estimators=100, random_state=0)

csv_file_path = "sample_data.csv"
features = ["Spool Vendor","Spool Name","Material","Calculated Length [mm]"]

numerical_cols = ["Calculated Length [mm]"]
categorical_cols = ["Spool Vendor","Spool Name","Material"]

csv = pd.read_csv(csv_file_path)

#transform the target data from categorical to numerical data
le = preprocessing.LabelEncoder()
csv["Print result [success canceled failed]"] = le.fit_transform(csv["Print result [success canceled failed]"])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = SimpleImputer(strategy='constant')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

y = csv["Print result [success canceled failed]"]
x = csv[features]

train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

my_pipeline.fit(train_X, train_y)

preds = my_pipeline.predict(val_X)

score = mean_absolute_error(val_y, preds)
print('MAE:', score)
print(preds)