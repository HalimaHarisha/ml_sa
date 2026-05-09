# ml_sa
# INTRODUCTION TO MACHINE LEARNING
# NAME : A.HALIMA HARISHA 
# REG NO : 212224040094
# QUESTION NO :01
Create a scatter plot between cylinder vs Co2Emission (green color)
# PROGRAM
```
import pandas as pd
import matplotlib.pyplot as plt
data = {
    'cylinder': [4, 6, 8, 4, 6, 8, 4, 6],
    'Co2Emission': [120, 180, 250, 130, 190, 260, 140, 200]
}
df = pd.DataFrame(data)
plt.scatter(df['cylinder'], df['Co2Emission'], color='green')
plt.xlabel('Cylinder')
plt.ylabel('Co2Emission')
plt.title('Cylinder vs Co2Emission')
plt.show()
```
# OUTPUT


<img width="724" height="565" alt="image" src="https://github.com/user-attachments/assets/97524ba5-01f0-4f40-8014-a0eab3680c04" />



# QUESTION NO :02
Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors
# PROGRAM
```
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'cylinder': [4, 6, 8, 4, 6, 8, 4, 6],
    'EngineSize': [1.5, 2.0, 3.5, 1.6, 2.2, 4.0, 1.8, 2.5],
    'Co2Emission': [120, 180, 250, 130, 190, 260, 140, 200]
}
df = pd.DataFrame(data)
plt.scatter(df['cylinder'], df['Co2Emission'],
            color='green', label='Cylinder vs Co2Emission')
plt.scatter(df['EngineSize'], df['Co2Emission'],
            color='blue', label='EngineSize vs Co2Emission')
plt.xlabel('Cylinder / EngineSize')
plt.ylabel('Co2Emission')
plt.title('Comparison Scatter Plot')
plt.legend()
plt.show()
```
# OUTPUT

<img width="721" height="570" alt="image" src="https://github.com/user-attachments/assets/bd175002-d2a6-4a66-ba86-64eaf97523e0" />




# QUESTION NO :03 
Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors
 # PROGRAM
 ```
import pandas as pd
import matplotlib.pyplot as plt
data = {
    'cylinder': [4, 6, 8, 4, 6, 8, 4, 6],
    'EngineSize': [1.5, 2.0, 3.5, 1.6, 2.2, 4.0, 1.8, 2.5],
    'FuelConsumption_comb': [6.5, 8.2, 12.5, 7.0, 9.0, 13.2, 7.5, 10.1],
    'Co2Emission': [120, 180, 250, 130, 190, 260, 140, 200]
}
df = pd.DataFrame(data)
plt.scatter(df['cylinder'], df['Co2Emission'],
            color='green', label='Cylinder vs Co2Emission')
plt.scatter(df['EngineSize'], df['Co2Emission'],
            color='blue', label='EngineSize vs Co2Emission')
plt.scatter(df['FuelConsumption_comb'], df['Co2Emission'],
            color='red', label='FuelConsumption_comb vs Co2Emission')
plt.xlabel('Features')
plt.ylabel('Co2Emission')
plt.title('Scatter Plot Comparison')
plt.legend()
plt.show()
```
 # OUTPUT

 <img width="719" height="632" alt="image" src="https://github.com/user-attachments/assets/b93a870d-afd9-4a52-9194-d500fb77410b" />




# QUESTION NO :04 
Train your model with independent variable as cylinder and dependent variable as Co2Emission
# PROGRAM 
```
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'cylinder': [4, 6, 8, 4, 6, 8, 4, 6],
    'Co2Emission': [120, 180, 250, 130, 190, 260, 140, 200]
}

df = pd.DataFrame(data)

X = df[['cylinder']]
y = df['Co2Emission']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')

plt.xlabel('Cylinder')
plt.ylabel('Co2Emission')
plt.title('Linear Regression: Cylinder vs Co2Emission')

plt.show()
```
# OUTPUT


<img width="729" height="629" alt="image" src="https://github.com/user-attachments/assets/a081953c-554c-4667-aeb4-32f22a39b888" />


# QUESTION NO :05

Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission

# PROGRAM
```
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'FuelConsumption_comb': [6.5, 8.2, 12.5, 7.0, 9.0, 13.2, 7.5, 10.1],
    'Co2Emission': [120, 180, 250, 130, 190, 260, 140, 200]
}

df = pd.DataFrame(data)

X = df[['FuelConsumption_comb']]
y = df['Co2Emission']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y, color='green')
plt.plot(X, y_pred, color='red')

plt.xlabel('FuelConsumption_comb')
plt.ylabel('Co2Emission')
plt.title('Linear Regression: FuelConsumption_comb vs Co2Emission')

plt.show()
```
# OUTPUT

<img width="740" height="566" alt="image" src="https://github.com/user-attachments/assets/25a3dab1-54a2-4af4-9906-825cf92d106d" />



# QUESTION NO:06
 Train your model on different train test ratio and train the models and note down their accuracies
 # PROGRAM 
 ```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = {
    'FuelConsumption_comb': [6.5, 8.2, 12.5, 7.0, 9.0, 13.2, 7.5, 10.1],
    'Co2Emission': [120, 180, 250, 130, 190, 260, 140, 200]
}

df = pd.DataFrame(data)

X = df[['FuelConsumption_comb']]
y = df['Co2Emission']

ratios = [0.2, 0.3, 0.4]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = r2_score(y_test, y_pred)

    print("Train Test Ratio:", 1-ratio, ":", ratio)
    print("Accuracy:", accuracy)
    print()
```

# OUTPUT

<img width="466" height="296" alt="image" src="https://github.com/user-attachments/assets/a821f425-9f3e-4a5b-beb1-35608bb7065c" />







