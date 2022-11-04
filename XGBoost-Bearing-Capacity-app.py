import pandas as pd
import streamlit as st
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm 
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from PIL import Image


st.write("""
# Prediction of Seismic Bearing Capacity Factor
***
""")

# Loads the FRP-RC columns Dataset
frprccolumns = pd.read_csv("bearingcapacity.csv")
print(frprccolumns.x1)

# Convert data
frprccolumns['x1'] = frprccolumns['x1'].astype(float)
frprccolumns['x2'] = frprccolumns['x2'].astype(float)
frprccolumns['x3'] = frprccolumns['x3'].astype(float)
frprccolumns['x4'] = frprccolumns['x4'].astype(float)
frprccolumns['x5'] = frprccolumns['x5'].astype(float)
frprccolumns['y'] = frprccolumns['y'].astype(float)

frprccolumns = frprccolumns[['x1','x2','x3','x4','x5','y']]
Y = frprccolumns['y'].copy()
X = frprccolumns.drop('y', axis=1).copy()

# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
value = ("0", "1")
options = list(range(len(value)))


def input_variable():
    x1 = st.sidebar.slider('σci/γB', float(frprccolumns.x1.min()), float(frprccolumns.x1.max()),
                               float(frprccolumns.x1.mean()))
    x2 = st.sidebar.slider('d/B', float(frprccolumns.x2.min()), float(frprccolumns.x2.max()),
                           float(frprccolumns.x2.mean()))
    x3 = st.sidebar.slider('GSI', float(frprccolumns.x3.min()), float(frprccolumns.x3.max()),
                            float(frprccolumns.x3.mean()))
    x4 = st.sidebar.slider('kh', float(frprccolumns.x4.min()),
                              float(frprccolumns.x4.max()), float(frprccolumns.x4.mean()))
    x5 = st.sidebar.slider('mi', float(frprccolumns.x5.min()),
                              float(frprccolumns.x5.max()), float(frprccolumns.x5.mean()))


    data = {'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5,
            }

    features = pd.DataFrame(data, index=[0])
    return features


# def get_regressor(clf_name):
#     global clf
#     if clf_name == "Linear Regression":
#         clf = LinearRegression()

#     elif clf_name == "Random Forest":
#         clf = RandomForestRegressor(random_state=0)

#     elif clf_name == "SVM":
#         clf = SVR(C=1)

#     elif clf_name == "Decision Trees":
#         clf = DecisionTreeRegressor()

#     elif clf_name == "KNN":
#         clf = KNeighborsRegressor()

#     elif clf_name == "XGBoost":
#         clf = xgb.XGBRegressor(objective='reg:squarederror',
#                          colsample_bytree=0.9,
#                          n_estimators=200,
#                          learning_rate=0.3,
#                          max_depth=4,
#                          reg_lambda=7,
#                          random_state=40)

#     return clf

# clf = get_regressor(regressor)

df = input_variable()

st.header('Specified Input Parameters')
st.table(df)
st.write("Here, x1 = σci/γB")
st.write("x2 = d/B")
st.write("x3 = GSI")
st.write("x4 = kh")
st.write("x5 = mi")
st.write('---')


st.header('Relative Importance of Each Feature in the XGBoost Model')

image = Image.open('RI.png')
st.image(image, use_column_width=True)

maxes = []
mins = []
for col in frprccolumns.columns:
  if col in ['x1','x2','x3','x4','x5']:
       mins.append(frprccolumns[col].min())
       maxes.append(frprccolumns[col].max())
       frprccolumns[col] = (frprccolumns[col] - mins[-1]) / (maxes[-1] - mins[-1])

#Build Model
def model():
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=40)
    # MinMax Scaling / Normalization of data
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    test_data=df
    model_regression=xgb.XGBRegressor(objective='reg:squarederror',
                         colsample_bytree=0.9,
                         n_estimators=200,
                         learning_rate=0.3,
                         max_depth=4,
                         reg_lambda=7,
                         random_state=40)
    model_run = model_regression.fit(X_train,Y_train)
    prediction = model_run.predict(test_data)
    return prediction

prediction=model()

st.header('Predicted Seismic Bearing Capacity Factor')
st.subheader("Regressor Used: XGBoost")
st.write('Seismic Bearing Capacity Factor (N) =', prediction[0])
st.write('---')
