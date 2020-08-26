import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
# save numpy array as npy file
from numpy import asarray
from numpy import save
# define data
data = np.random.rand(100,100)
# save to npy file
save('q3-input.npy', data)
A = np.copy(data)
graph = plt.imshow(data, cmap ="Greys") 
plt.colorbar(graph)
plt.title('input data density map', fontweight ="bold") 
plt.show() 

#a 
# data_3a = np.asarray(data).reshape(-1)
temp = np.reshape(data,(1,10000))
temp.sort()
save('q3-output-sorted.npy', temp)

extent = 0,10000,0,600
graph_3a = plt.imshow(temp, cmap ="Greys", alpha = 0.7,  
           interpolation ='bilinear', extent = extent) 
# graph_3a = plt.imshow(data_3a, cmap ="Blues",) 

plt.colorbar(graph_3a)
plt.title('input data density map', fontweight ="bold") 
plt.show() 

#b
data_array = data.reshape(-1)
  
# Creating plot 
fig = plt.figure(figsize =(10, 7)) 
  
graph_3b = plt.hist(data_array, bins = 20, rwidth=0.8, color="grey")  
  
plt.title("Numpy Histogram")  
  
# show plot 
plt.show() 

#c
bot_left =  [A[i][:50] for i in range(50, 100)]
X = bot_left
save('q3-output-x.npy', bot_left)
graph_3c = plt.imshow(bot_left, cmap ="Greys", alpha = 0.7) 
plt.colorbar(graph_3c)
plt.title('bottom left quadrant of A', fontweight ="bold") 
plt.show() 
save('q3-output-x.npy', graph_3c)

#d
mean = data.mean()
Y = A - mean 
graph_3d = plt.imshow(Y, cmap ="Greys", alpha = 0.7) 
# graph_3d=plt.imshow(Y, cmap='gray', alpha = 0.7,vmin=0, vmax=255)
plt.colorbar(graph_3d)
plt.title('intensity value substracted from A', fontweight ="bold") 
plt.show() 
save('q3-output-y.npy', graph_3d)

#e
mean = np.mean(A)
zeros1 = np.zeros([100,100])
zeros2=np.zeros([100,100])
greater = np.where(A>=mean, 1, 0)
Z= np.dstack((greater, zeros1, zeros2))
plt.imshow(Z)
plt.show()
plt.imsave('q3-output-y.png', Z)