from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

data = pd.DataFrame({
    "cgpa":[6,7,8,9],
    "internships":[0,1,2,3],
    "projects":[1,2,4,5],
    "placement":[40,60,80,95]
})

X = data[["cgpa","internships","projects"]]
y = data["placement"]

model = LinearRegression()
model.fit(X,y)

joblib.dump(model,"model.pkl")