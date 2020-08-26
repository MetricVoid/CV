import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np
plt.set_cmap('gray')
img = im.imread('q4-input.png')
img = img[:,:,0:3]
def split_channels(img: np.ndarray):
    assert len(img.shape) == 3 and img.shape[-1] == 3
    return np.squeeze(np.split(img, img.shape[-1], -1), axis=-1)
[B, G, R] = split_channels(img)
temp = G 
G = R
R = temp
img4a=np.dstack((R,G,B))
plt.imsave("4a.png",img4a)
# print(img)
plt.title("4a swapped")
plt.imshow(img4a)
plt.imsave("q4-output-swapped.png",img4a)
plt.show()
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = im.imread('q4-input.png')
img = img[:,:,0:3]
gray = rgb2gray(img)
# plt.imshow(gray)
# plt.title("test gray")
# plt.show()
# print(np.shape(gray))
realgray = np.copy(gray)

plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.imsave('greyscale.png', gray, format='png', cmap='gray')
img4b=im.imread('greyscale.png')
plt.imsave("q4-output-grayscale.png",img4b)
plt.title("4b gray")

# plt.savefig('greyscale.png')
plt.show()

k=im.imread('greyscale.png')
#image sclicing into 2D. 
x=k[:,:,0]
# x co-ordinate denotation. 
plt.xlabel("Value")
# y co-ordinate denotation.
plt.ylabel("pixels Frequency")
# title of an image .
# plt.title("Original Image")
# imshow function with comperision of gray level value.
plt.imshow(x,cmap="gray")
#plot the image on a plane.
# plt.show()
y=np.shape(x)
z=np.zeros(y)
#convert the image into its negative value.
z=255-x
img4c = np.copy(z)
plt.xlabel("Value")
plt.ylabel("pixels Frequency")
plt.title("Negative image ")
plt.imshow(z,cmap="gray")
plt.title("4c negative")
plt.imsave("q4-output-negative.png",img4c)
plt.show()

def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    return np.dot(rgb[...,:3], [0.144, 0.299, 0.587])
f1 = im.imread('greyscale.png')
# plt.imshow(f1)
# plt.show()
# f1 = rgb2gray(f1)
f2=f1[...,::-1,:]
plt.imshow(f2)
img4d = np.copy(f2)
plt.title("4d mirror")
plt.show()
plt.imsave("q4-output-mirror.png",img4d)
f = np.array(f1)/2 + np.array(f2)/2
plt.imshow(f)
img4e = np.copy(f)
plt.title("4e mix")
plt.imsave("q4-output-average.png",img4e)
plt.show()

gray = im.imread('greyscale.png')[:,:,0:3]
filter2 = np.random.rand(100,100)
filter3 = np.stack((filter2, filter2, filter2), axis = -1)
# print(filter3.shape)
# print(filter3[0])
plt.imshow(filter3)
plt.imsave("q4-noise.png",filter3)
save("q4-noise.npy",filter3)
plt.show()

# plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.imshow(gray, cmap ="gray") 
final = -filter3/2 + gray
final = np.clip(final,0,1)
graph_4f = plt.imshow(final, cmap = plt.get_cmap('gray')) 
img4f=final
plt.imsave("q4-output-noise.png",final)
plt.show()

fig, ax = plt.subplots(3, 2)
ax[0, 0].imshow(img4a)
ax[0, 1].imshow(img4b)
ax[1, 0].imshow(img4c)
ax[1, 1].imshow(img4d)
ax[2, 0].imshow(img4e)
ax[2, 1].imshow(img4f)

plt.show()


  
fig.suptitle('Final graph of 6 results') 
plt.show() 