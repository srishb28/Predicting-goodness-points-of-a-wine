# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt



def RidgeReg(x,y,l):
     xbar=np.transpose(x)
     concat_one=np.ones(xbar.shape[1])
     xbar=np.vstack([xbar,concat_one])
     xbartrans=np.transpose(xbar)
     i=np.identity(xbar.shape[0])
     c=np.dot(xbar,xbartrans)+l*i
     cinv=inv(c)
     d=np.dot(xbar,y)
     wbar=np.dot(cinv,d)
     wbartrans=np.transpose(wbar)
     w1=np.delete(wbar,-1,0)
     b=wbar[-1]
     sq=np.square(wbar)
     w=sq.sum()- b*b
     ytrans=np.transpose(y)
     obj1=w*l
     obj2=(np.square(np.dot(wbartrans,xbar)-ytrans)).sum()
     obj = obj1 + obj2
     cverror=np.zeros(y.shape)
     for col in range(xbar.shape[1]):
          num=np.dot(wbartrans,xbar[:,col])-y[col]
          denom=1-((np.transpose(xbar[:,col])@cinv)@xbar[:,col])
          error=num/denom
          cverror[col]=error
     return [w1,b,obj,cverror]
        
     
def rmse(w,x,y,n):
     xbar=np.transpose(x)
     concat_one=np.ones(xbar.shape[1])
     xbar=np.vstack([xbar,concat_one])
     rmseerrors=[]
     rmse_final=math.sqrt(((np.square(np.dot(np.transpose(w),xbar)-np.transpose(y))).sum())/n)
     #print(rmse_final)
     rmseerrors.append(rmse_final)
     return rmseerrors

#loading training data
train_data_x=pd.read_csv('C:/Users/Srishti/Desktop/ml/kaggle/trainData.csv',header=None)
train_data_x=train_data_x.drop(train_data_x.columns[0],axis=1)
train_data_y=pd.read_csv('C:/Users/Srishti/Desktop/ml/kaggle/trainLabels.csv',header=None)
train_data_y=train_data_y.drop(train_data_y.columns[0],axis=1)
train_x=train_data_x.values
train_y=train_data_y.values
n_train=train_data_x.shape[0]
k_train=train_data_x.shape[1]

#loading validation data
validate_data_x=pd.read_csv('C:/Users/Srishti/Desktop/ml/kaggle/valData.csv',header=None)
validate_data_x=validate_data_x.drop(validate_data_x.columns[0],axis=1)
validate_data_y=pd.read_csv('C:/Users/Srishti/Desktop/ml/kaggle/valLabels.csv',header=None)
validate_data_y=validate_data_y.drop(validate_data_y.columns[0],axis=1)
validate_x=validate_data_x.values
validate_y=validate_data_y.values
n_validate=validate_data_x.shape[0]
k_validate=validate_data_x.shape[1]

cverror=[]
rmse_train=[]
rmse_validate=[]
lam=[0.01,0.1,1,10,100,1000]
minimum=99999999
obj=[]
w=[]
w1=[]




#iterating through different lambdas given in the assignment
for i in lam:
     weight,bias,objective,cverr=RidgeReg(train_x,train_y,i)
     cverrs=(np.square(cverr)).sum()
     if(cverrs<minimum):
          minimum=cverrs
          ind=lam.index(i)
          minlam=i
          min2lam=lam[lam.index(i)-1]
     obj.append(objective)
     cverror.append(cverrs)
     w1.append(weight)
     wbar=np.vstack([weight,bias])
     w.append(wbar)
     rmse_train.append(rmse(wbar,train_x,train_y,n_train))
     rmse_validate.append(rmse(wbar,validate_x,validate_y,n_validate))
print(w)
print("rmse vales for training data :-")
print(rmse_train)
print("rmse vales for validation data :-")
print(rmse_validate)
#for finding out a more accurate lambda a loop between the minimum
#and second minimum lambda value is run

cverror1=[]
w_minlam=[]
w1_minlam=[]
minimum1=99999
obj1=[]
lam2=np.arange(min2lam,minlam,0.1)
lam1=[]
for i in lam2:
    lam1.append(i)

#print(lam1)
for i in lam1:
     weight1,bias1,objective1,cverr1=RidgeReg(train_x,train_y,i)
     cverrs1=(np.square(cverr1)).sum()
     if(cverrs1<minimum1):
          minimum1=cverrs1
          ind1=lam1.index(i)
          minlam1=i
     obj1.append(objective1)
     cverror1.append(cverrs1)
     w1_minlam.append(weight1)
     wbar1=np.vstack([weight1,bias1])
     w_minlam.append(wbar1)
     

#plotting errors
plt.figure(1)
plt.subplot(221)
x_plot=range(len(rmse_train))
plt.plot(x_plot,rmse_train)
plt.plot(x_plot,rmse_validate)
plt.xticks(x_plot,lam)
plt.legend(['y=rmse_train','y=rmse_validate'])
plt.xlabel('lambda')
plt.grid(True)
plt.subplot(223)
plt.plot(x_plot,cverror)
plt.ylabel('loocv')
plt.xlabel('lambda')
plt.grid(True)
plt.xticks(x_plot,lam)
plt.show()



#printing minimum loocv,regularization and sum of squared errors for lambda given in the question
print("printing values for min lambda according to the values given in assignment:-")
print("minimum value of loocv is "+ str(minimum) + " at lambda " + str(minlam))
regularizationterm=np.dot(np.transpose(w[ind]),w[ind])
print ("Objective value for minimum LOOCV lambda is :-"+ str(obj[ind]))
print("Regularization term for minimum LOOCV data is :-"+ str(np.dot(np.transpose(w_minlam[ind1]),w_minlam[ind1])))
print("the sum of squared errors is"+ str(obj[ind]-regularizationterm))
plt.show()
#printing minimum loocv,regularization and sum of squared errors for more accurate lambdas
print("printing values for more accurate value of lambdas:-")
print("minimum value of loocv is "+ str(minimum1) + " at lambda " + str(minlam1))
regularizationterm=np.dot(np.transpose(w_minlam[ind1]),w_minlam[ind1])
print ("Objective value for minimum LOOCV lambda is :-"+ str(obj1[ind1]))
print("Regularization term for minimum LOOCV data is :-"+ str(np.dot(np.transpose(w_minlam[ind1]),w_minlam[ind1])))
print("the sum of squared errors is"+ str(obj1[ind1]-regularizationterm))



#printing most important and least important data
labels=pd.read_csv('C:/Users/Srishti/Desktop/ml/kaggle/featureTypes.txt',sep="\n",header=None)
featureweights=np.delete(w_minlam[ind1],-1,0)
featurematrix=np.hstack([featureweights,labels.values])
#print(featurematrix)
featurematrix=featurematrix[np.argsort(featurematrix[:,0])][::-1]
print("top 10 most important features are:-")
print(featurematrix[:10])
#print(featurematrix[:10])
print("top 10 least important features are:-")
print(featurematrix[-10:])
print("the features which have the least value of weight makes sense since sound,love,soft,spices,relatively etc doesnt really contribute to the goodness of a wine, neither do they describe properly how good a wine tastes.on the other hand features with high weights like infused,red,pineapple orange,little heavy etc describes wine goodness as they describe different flavors which can help us better,taste,feel") 



#predicting values based on a more accurate value of lambda calculated above
test_data_x=pd.read_csv('C:/Users/Srishti/Desktop/ml/kaggle/testData.csv',header=None)
test_data_x=test_data_x.drop(test_data_x.columns[0],axis=1)
test_x=test_data_x.values
test_transpose=np.transpose(test_x)
concat_one=np.ones(test_transpose.shape[1])
test_transpose=np.vstack([test_transpose,concat_one])
predicted_values=(np.dot(np.transpose(w_minlam[ind1]),test_transpose))
#print(predicted_values)  
df=pd.DataFrame(np.transpose(predicted_values),columns={'prediction'})
df.index.name='id'
df.to_csv('C:/Users/Srishti/Desktop/ml/kaggle/predTestLabels.csv')

