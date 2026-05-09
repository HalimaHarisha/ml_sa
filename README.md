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
<img width="695" height="545" alt="image" src="https://github.com/user-attachments/assets/8b5352be-f84a-4b92-9e23-441883295f4b" />

