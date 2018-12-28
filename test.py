import numpy as np
import math

# a = np.array([[1, 2, 3], [4, 5, 6]])
temp = [x1,y1,x2,y2] = 1,2,3,4
# np.insert(a, [1, 5,5],[1, 5,5] ,axis=0)
# np.insert(a ,[x1],[y1],[x2],[y2],axis=1)
a = []
a.append(list(temp))
a.append(list(temp))
a= set(a)
print(np.array(a))
#
# b = np.array([1,2])
# print(b.shape)
#
# vector_one = np.array([5 ,4])
# vector_two = np.array([5,2])
# print(np.dot(vector_two,vector_one))
# print(np.sum(np.dot(vector_two,vector_one)/(np.absolute(vector_one))))
# # cos_theta = math.fabs(np.dot(vector_two,vector_one)/(np.absolute(vector_one)*np.absolute(vector_two)))