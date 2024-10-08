"import library"
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
"----------------------------------------------------------------------------"
"import data"
foot_ball=pd.read_csv(r"C:\Users\EL-BOSTAN\OneDrive\Desktop\archive\fpl_playerstats_2024-25.csv")

"----------------------------------------------------------------------------"

"show data"
foot_ball.head()
"-----------------"
foot_ball.info()
"-----------------"
foot_ball.describe()

"-----------------------------------------------------------------------------"

"preperation the data"

foot_ball.isnull().sum()

"fill NA"

data = foot_ball.fillna({
    "gw1_points": foot_ball["gw1_points"].mode()[0],
    "gw2_points": foot_ball["gw2_points"].mode()[0],
    "gw3_points": foot_ball["gw3_points"].mode()[0],
    "gw4_points": foot_ball["gw4_points"].mode()[0],
    "team_form": "league A",
})

data.isnull().sum()

"----------------------------------------------------------------------------"
sn.set_style("ticks")
sn.pairplot(data,diag_kind="kde",kind="scatter",palette="husl")
plt.show()



# Assuming 'data' is your DataFrame and it contains 'first_name' and 'expected_goals' columns
plt.bar(data["player_position"], data["expected_goals"])
plt.xlabel('player_position')
plt.ylabel('Expected Goals')
plt.title('Expected Goals by Player')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if there are many players
plt.show()

"------------------------------------------------------------------"

"------------------------------------------------------------------"



"----------------------------------------------------------------------"

x=data["player_position"]
y=data["creativity"]
plt.plot(x,y)
plt.xlabel("player_position")
plt.ylabel("creativity")
plt.title('player_position and creativity ')
plt.show()

"------------------------------------------------------------------------------"
"my model"
num_columns = len(data.columns)
print("Number of columns:", num_columns)
data = data.drop(["first_name", "second_name","team_name","status","player_position"], axis=1)


from sklearn.model_selection import train_test_split
X = data.iloc[:, 0:31] 
y=data.iloc[:, -1].values  # Drop columns 0 to 33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



"""# 3. Standardize the features (important for models like Logistic Regression)"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)