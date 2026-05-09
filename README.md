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
<img width="523" height="420" alt="image" src="https://github.com/user-attachments/assets/9b789f0a-48d6-49ba-86fa-e1e070f4116a" />


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
<img width="755" height="606" alt="image" src="https://github.com/user-attachments/assets/8183fcf7-f9e9-41c5-8c93-1d0217634551" />


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
 <img width="721" height="608" alt="image" src="https://github.com/user-attachments/assets/9f973e75-6b03-4696-86df-4fffdee810e8" />


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

<img width="733" height="625" alt="image" src="https://github.com/user-attachments/assets/ed3d7212-35ef-4fb3-b237-129ae248ac8a" />

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
<img width="689" height="579" alt="image" src="https://github.com/user-attachments/assets/968a493b-56f1-4914-8338-c1eb33753dd2" />

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







