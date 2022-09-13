import pandas as pd

data = pd.read_csv('resources/titanic.csv')
print(data.columns)
print(data.head(5))
print(data[4:7])

print(data.loc[1:7,['sex','age']])
print(data.iloc[1:7,:4])

print(data['name'].head())

data[(data['sex'] == 'female')]['name'].head()

print(data[(data['sex'] == 'women') & (data['age'] > 50) | (data['sex'] == 'male')].head())

data.rename(columns={'name':'Name'}, inplace=True)

def get_last_name(name):
    return name.split(',')[0].strip()

last_names = data['Name'].apply(get_last_name)
print(last_names)

data["Last Name"] = last_names
print(data)

data.drop('Last Name', axis=1, inplace=True)
print("Удаление столбца\n",data)

print("первые пять пустых ячеек столбца boat\n", data['boat'].isnull())
print("первые пять пустых ячеек столбца boat\n", data[data['boat'].isnull()])
print("первые пять пустых ячеек столбца boat\n", data[data['boat'].isnull()].head())

data[data['boat'].notnull()].head() # первые пять непустых ячеек столбца boat
print("первые пять непустых ячеек столбца boat\n", data['boat'].notnull())
print("первые пять непустых ячеек столбца boat\n", data[data['boat'].notnull()])
print("первые пять непустых ячеек столбца boat\n", data[data['boat'].notnull()].head())

print("Grouping")
print("First 10 people: \n", data.loc[1:10,["sex","pclass","fare"]])
print("First 10 people Value_counts: \n", data[1:10].groupby('sex')['pclass'].value_counts())
print("Value_counts: \n", data.groupby('sex')['pclass'].value_counts())

print(set(data["sex"]))
print(set(data["pclass"]))

print(data.groupby('pclass')['fare'].describe())
print("MAX fare: ",data["fare"].max() )


print("First 10 people: \n", data.loc[1:10,["sex","pclass","fare","survived"]])
print(data.loc[1:10].groupby('sex')['survived'].mean())
print(data.groupby('sex')['survived'].mean())

data.to_csv('titanic_2.csv', index=False)