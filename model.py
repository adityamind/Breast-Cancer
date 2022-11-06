import numpy as np
import pandas as pd
import pickle
class knn_classifier:
#finding k nearest neighbor
  def KNeighbor(self,train_x,test_x,train_y,k):
    distances=list()
    k_distance=list()
    for (row,d) in zip(train_x,train_y):
          d=0.0
          x1=row
          x2=test_x
    for i in range(len(x1)):
      d+=(x1[i]-x2[i])*(x1[i]-x2[i]);
      dist=np.sqrt(d)
      distances.append((d,dist))
    distances.sort(key=lambda z:z[1])
    for i in range(k):
      k_distance.append(distances[i][0])
    return k_distance

#prediction
  def prediction(self,train_x,test_x,train_y,k):
    predict=self.KNeighbor(train_x,test_x,train_y,k)
    return max(set(predict),key=predict.count)

#calling for test data
  def KNN(self,train_x,test_x,train_y,k=5):
    if self.prediction(train_x,test_x,train_y,k)==1:
          return "Malignant"
    return "Benign";
  
  def predict(self,testdata):
      df=pd.read_csv('data.csv')
      df=df.replace({'diagnosis':{'M':1,'B':0}})
      dataset=np.array(df)
      train_y=[row[1] for row in dataset]
      train_x=[row[2:32] for row in dataset]
      return self.KNN(train_x,testdata,train_y,5)

MyKnn=knn_classifier()

pickle.dump(MyKnn,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
