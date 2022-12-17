import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data= pd.read_csv("Acoustic Features.csv", sep=",")

Y= data["Class"]
X=data.drop("Class",axis=1)

print(X.shape) #400,50

from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.2,random_state=42)



def objective_function(solution):
  model = KNeighborsClassifier()
  if(sum(solution)==0):
    return 0
  model.fit(X_Train.loc[:,solution],Y_Train)
  accuracy = model.score(X_Test.loc[:,solution],Y_Test)
  return accuracy


best_obj = 0

for i in range(1000):
  solution = np.random.random(50)>0.5
  obj_val = objective_function(solution)
  if(obj_val>best_obj):
    best_obj = obj_val
    best_sol = solution.copy()
  print(best_obj)