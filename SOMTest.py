import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from minisom import MiniSom

img = cv2.imread("/Users/tschmidt/Desktop/93060443_521533372071021_4184655849890775040_n.png",cv2.IMREAD_GRAYSCALE)

# +
#cv2.imwrite("/Users/tschmidt/Desktop/derp.jpg", img)
xDm = img.shape[0]
yDm = img.shape[1]

#TODO: ADD A BETTER SYSTEM OF DEALING WITH ODD NUMS
img1 = img[0:xDm//2,0:yDm//2]
img2 = img[0:xDm//2,yDm//2:-1]
img3 = img[xDm//2:-1,0:yDm//2]
img4 = img[xDm//2:-1,yDm//2:-1]

dm1 = img1.shape
dm2 = img2.shape
dm3 = img3.shape
dm4 = img4.shape

# +
img1 = img1.flatten()
img2 = img2.flatten()
img3 = img3.flatten()
img4 = img4.flatten()

print(img1.size)
print(img2.size)
print(img3.size)
print(img4.size)

dataset = np.column_stack([img1,img2,img3,img4])

# +
#run SOM -- this code creates/trains the SOM and calculates stats of interest

nx = 2
ny = 2
#N = 5000
#data = dataset[-N:].T
data = dataset.T

print(data.shape)
print(dataset.shape)

#make, initialize, and train the SOM
som = MiniSom(nx, ny, len(data[0]), sigma=1., learning_rate=0.5) # initialization of (ny x nx) SOM
som.pca_weights_init(data)
som.train_random(data, 5) # trains the SOM (WAS N*2)

qnt = som.quantization(data) #this is the pattern of the BMU of each observation (ie: has same size as data input to SOM)
bmu_patterns = som.get_weights() #this is the pattern of each BMU; size = (nx, ny, len(data[0]))
QE = som.quantization_error(data) #quantization error of map
TE = som.topographic_error(data) #topographic error of map

#calculate the BMU of each observation
bmus = []
bmus_num = []
for kk in range(len(data)):
    bmus.append(som.winner(data[kk]))
    num = bmus[kk][0]*ny + bmus[kk][1]
    bmus_num.append(num)
    
inds = []
for ii in range(ny):
    for jj in range(nx):
        inds.append((ii,jj))
     
#compute the frequency of each BMU
freq = np.zeros((nx,ny))
for bmu in bmus:
    freq[bmu[0]][bmu[1]]+=1
freq/=N

# +
#visualize

vmin = np.min(bmu_patterns)
vmax = np.max(bmu_patterns)

print(bmu_patterns.shape)
print(data.shape)
print(len(data[:,0]))
print(bmu_patterns[0,0,0])

plt.figure(figsize=(5*nx,5*ny))
for kk in range(nx*ny):   
    plt.subplot(ny,nx,kk+1)
    indx = inds[kk][1]
    indy = inds[kk][0]
    plt.imshow(np.reshape(bmu_patterns[indx][indy],(100,50)),cmap='RdBu',aspect='auto', vmin = vmin, vmax = vmax)
    plt.xlabel('Longitude', fontsize = 20)
    plt.ylabel('Latitude', fontsize = 20)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.title('Node : ' + str(inds[kk]) + '\n Freq = ' + str(freq[indx][indy]*100)[:4] + '%', fontsize = 24)
    
plt.tight_layout()
    
plt.show()
