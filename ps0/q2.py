import numpy as np
def random_dice(N):
    x = np.random.rand(N)*6 + 1 
    y = np.floor(x).astype(int)
    return y
def reshape_vector(y):
    x =  y.reshape(-1,2)
    return x

def max_value(z):
    x = np.max(z)
    y = np.where(z == x)
    return y


def count_ones(v):
    x = v.count(1)
    return x



# test
#2a random dice
print(random_dice(20))
#2b reshape
y = np.array([1, 2, 3, 4, 5, 6])
print(reshape_vector(y))
#2c max position
z = np.array([[1,2],[3,4],[5,6]])
print(max_value(z))
#2d count ones
v = [1,1,2,1,6,7,8,1,9,1]
print(count_ones(v))