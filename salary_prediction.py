import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

df = pd.read_csv('Salary_Data.csv')
df.dropna(subset=['Salary'], inplace=True)

X = df.drop('Salary', axis=1)
y = df['Salary']

num_cols = ['Age', 'Years of Experience']
cat_cols = ['Gender', 'Education Level', 'Job Title', 'Remote Work Status', 'Industry', 'Company Size', 'City or Region']

num_transformer = Pipeline([('scaler', StandardScaler())])
cat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

rf = RandomForestRegressor(n_estimators=100, random_state=0)
gb = GradientBoostingRegressor(n_estimators=100, random_state=0)
voting = VotingRegressor([('rf', rf), ('gb', gb)])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', voting)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Voting R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

joblib.dump(model, 'voting_model.pkl')
