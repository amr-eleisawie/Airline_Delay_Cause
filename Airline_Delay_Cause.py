import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error 


data = pd.read_csv("C:\\Users\\user\\Downloads\\Airline_Delay_Cause.csv")
print(data.info())
# حذف الصفوف اللي فيها missing values
data.dropna(inplace=True)

# إعادة ضبط الفهارس بعد الحذف 
data.reset_index(drop=True, inplace=True)

# حذف الأعمدة غير المطلوبة
data = data.drop(['carrier_name', 'airport', 'airport_name', 'carrier'], axis=1)

# فصل X و y
X = data.drop('late_aircraft_ct', axis=1)
y = data['late_aircraft_ct']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=44, shuffle=True
)

# تطبيع البيانات (Standardization) لجميع النماذج اللي تحتاج تطبيع
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تعريف النماذج
LinearRegressionModel = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
SGDRegressionModel = SGDRegressor(alpha=0.1, random_state=33, penalty='l2', loss='huber')
LassoRegressionModel = Lasso(alpha=1.0, random_state=33)
RidgeRegressionModel = Ridge(alpha=1.0, random_state=33)
RandomForestRegressorModel = RandomForestRegressor(n_estimators=1000, max_depth=8, random_state=33)
GBRModel = GradientBoostingRegressor(n_estimators=500, max_depth=7, learning_rate=1.5, random_state=33)
SVRModel = SVR(C=1.0, epsilon=0.1, kernel='rbf')
DecisionTreeRegressorModel = DecisionTreeRegressor(max_depth=3, random_state=33)
KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')

# قائمة بالنماذج
Models = [
    LinearRegressionModel,
    SGDRegressionModel,
    LassoRegressionModel,
    RidgeRegressionModel,
    RandomForestRegressorModel,
    GBRModel,
    SVRModel,
    DecisionTreeRegressorModel,
    KNeighborsRegressorModel
]

# تدريب النماذج وطباعتها
for Model in Models:
    print(f'For Model: {str(Model).split("(")[0]}')
    
    # تدريب النموذج
    Model.fit(X_train_scaled, y_train)
    
    # طباعة الدرجات
    print(f'Train Score is: {Model.score(X_train_scaled, y_train)}')
    print(f'Test Score is: {Model.score(X_test_scaled, y_test)}')

    # التنبؤ
    y_pred = Model.predict(X_test_scaled)
    
    # حساب المؤشرات
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mdse = median_absolute_error(y_test, y_pred)
    
    # طباعة المؤشرات
    print(f'MAE value is: {mae}')
    print(f'MSE value is: {mse}')
    print(f'MdSE value is: {mdse}')
    print('=================================================')
print ("End")