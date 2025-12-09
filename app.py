import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import  LabelEncoder


# Metadata contains type time testid filename etc..
meta = pd.read_csv('cleaned_dataset/metadata.csv')

# print(meta.head(10))
# print(meta.type.unique())

# Filtering only discharge file just for the basic health calculation
discharge_files = meta[meta.type == 'discharge']

# print(len(discharge_files.battery_id.unique()))

# discharging test file names
files = discharge_files[['filename', 'battery_id', 'start_time']].reset_index(drop=True)
files = files.sort_values(by=['filename', 'start_time'])

dfs = []
cycles = pd.DataFrame(columns=['battery_id','avg_voltage','avg_current','avg_temp','avg_load_current','avg_load_voltage','time', 'capacity'])

#create DataFrame for individual files
for i in range(len(files)):
    file = pd.read_csv(f'cleaned_dataset/data/{files.iloc[i].filename}')
    file = file[file.Voltage_load > 0]

    avg_voltage = file.Voltage_measured.mean()
    avg_current = file.Current_measured.mean()
    avg_temp = file.Temperature_measured.mean()
    avg_load_voltage = file.Voltage_load.mean()
    avg_load_current = file.Current_load.mean()
    time = file.Time.max()
    capacity = avg_current * time * -1
    cycle = {'battery_id':[files.iloc[i].battery_id] ,'avg_voltage':[avg_voltage] ,'avg_current':[avg_current] ,'avg_temp':[avg_temp] ,'avg_load_current':[avg_load_current] ,'avg_load_voltage':[avg_load_voltage] ,'time':[time], 'capacity':[capacity]}
    cycle = pd.DataFrame(cycle)
    #print(cycle)
    cycles = pd.concat([cycles,cycle], ignore_index=True)
    dfs.append(file)


## Assign battery health to each row equally divided 
#cycles['health'] = (cycles.groupby('battery_id')['battery_id'].transform(lambda x: np.linspace(100, 0, len(x))))
# percent life completed
cycles['life_cmpl'] = (cycles.groupby('battery_id')['battery_id'].transform(lambda x: np.linspace(0,100, len(x))))

cycles['battery_id'] = LabelEncoder().fit_transform(cycles.battery_id)

#MODEL CREATION
X = cycles[['time', 'battery_id']]
y = cycles['life_cmpl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))
print("R2 Sore", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Remaining Life")
plt.show()

# print(cycles)
# # print(dfs[0][dfs[0].Voltage_load == min(dfs[0].Voltage_load)])
battery = cycles[cycles.battery_id == cycles.iloc[0].battery_id]
# battery2 = cycles[cycles.battery_id == cycles.iloc[72].battery_id]
# battery3 = cycles[cycles.battery_id == cycles.iloc[144].battery_id]
plt.figure()
x= battery.life_cmpl
y= battery.time
# x2= battery2.index - 72
# y2= battery2.capacity
# x3= battery3.index - 144
# y3= battery3.capacity
# x4= battery.index
# y4= battery.avg_temp
# plt.xlabel("Cycles")
# plt.ylabel("Capaity ( avg I*time)")
# # x= dfs[0].Time
# # y= dfs[0].Voltage_load
# # x2= dfs[100].Time
# # y2= dfs[100].Voltage_load
plt.plot(x,y, linewidth=2)
# #plt.plot(x2,y2, linewidth=2)
# #plt.plot(x3,y3, linewidth=2)
# #plt.plot(x4,y4, linewidth=2)
plt.show()

# print(battery.describe())
# print(battery.describe())

# for i in dfs:
#     print(dfs[0].info())

# for i in dfs:
#     b_id = 
#     cycle = {'battery_id','avg_voltage','avg_current','avg_temp','avg_load_current','avg_load_voltage','time'} 




# df = pd.read_csv('cleaned_dataset/data/00001.csv')

# print("First \n",df.head(2))

# avg_voltage = df.Voltage_measured.mean()
# avg_current = df.Current_measured.mean()
# avg_temp = df.Temperature_measured.mean()

# print("EOF \n",df.tail(2))