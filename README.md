import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    'Age': [20, 25, 30, 35, 40],
    'Salary': [20000, 30000, 40000, 50000, 60000],
    'Experience': [1, 3, 5, 7, 10]
}

df = pd.DataFrame(data)

# Feature Generation
df['Salary_Per_Year'] = df['Salary'] / (df['Experience'] + 1)
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 25, 35, 100],
    labels=['Young', 'Middle', 'Senior']
)

# Feature Scaling
scaler = StandardScaler()
df['Scaled_Salary'] = scaler.fit_transform(df[['Salary']])

# Display generated features
print(df)
