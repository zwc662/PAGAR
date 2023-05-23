import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


 
# get the map image as an array so we can plot it 
import matplotlib.image as mpimg 
map_img = mpimg.imread('bg.png') 

# making and plotting heatmap 


rewards1 = np.array([[1.2e-6, 1.5e-6, 4.6e-6, 4.6, 0.12,-0.2, -0.2],
                    [-0.2, -0.2, -0.2, 2e-6, -0.2, -0.2, -0.2],
                    [-0.2, 4.3e-6, 2.1e-6, 4.3e-6, -0.2, -0.2, -0.2],
                    [-0.2, 0.65, -0.2, -0.2, -0.2, -0.2, -0.2],
                    [-0.2, 4.8, 3.45, 3.28, 1.6e-6, 4.37e-6, 1.22],
                    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 2.9],
                    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.48],])

rewards = np.array([[0.0008, 0.001, 0.0006, 3.2, 0.69, -0.2, -0.2],
                    [-0.2, -0.2, -0.2, 0.0005, -0.2, -0.2, -0.2],
                    [-0.2, 0.07, 0.005, 0.006, -0.2, -0.2, -0.2],
                    [-0.2, 2.5, -0.2, -0.2, -0.2, -0.2, -0.2],
                    [-0.2, 3.25, 1.35, 0.69, 0.007, 0.07, 3.1],
                    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 1.3],
                    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.53],])

import seaborn as sns; sns.set()

hmax = sns.heatmap(rewards,
            cmap = 'RdGy_r', # this worked but I didn't like it
            #cmap = matplotlib.cm.winter,
            alpha = 0.4, # whole heatmap is translucent
            annot = False, #True,
            zorder = 2,
            )
 
hmax.imshow(map_img,
          aspect = hmax.get_aspect(),
          extent = (-1., 8., 8, -1),
          zorder = 1) #put the map under the heatmap

from matplotlib.pyplot import show 
show()

"""
fig, ax = plt.subplots()
img = plt.imread("airlines.jpg")
im = ax.imshow(rewards, cmap = 'RdGy_r')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("reward", rotation=-90, va="bottom")
#ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

# Show all ticks and label them with the respective list entries
 
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.

 
fig.tight_layout()
plt.show()


# add alpha (transparency) to a colormap
import matplotlib.cm from matplotlib.colors 
import LinearSegmentedColormap 
wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
wd['alpha'] =  ((-0.2, -0.2, 0.3), 
               (0.3, 0.3, 1.0),
               (1.0, 1.0, 1.0))
"""