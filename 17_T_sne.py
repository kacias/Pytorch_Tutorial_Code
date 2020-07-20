
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


#x = [[1,2], [3,4], [5,6], [5,5], [8,7]]
#x = [[1,2,3,4,5,6], [4,5,3,4,5,6]]

#from google.colab import files
#uploaded = files.upload()
#print("uploaded:{}".format(uploaded))

x = np.loadtxt("./sequence_int.txt", dtype='float', delimiter=',', skiprows = 1)
print("y.shape:{}".format(x))



tsne = TSNE(n_components=2)
y = tsne.fit_transform(x)

print(y)


#plt.scatter(y[:,0], y[:,1], color='0.75')
#plt.scatter(y[:,1], color='0.25')
plt.scatter(y[:,0], y[:,1], alpha=0.9, c=y[:,0], s=3, cmap='viridis')


'''
plt.scatter(y[:,0], # x
            y[:, 1], # y
           #alpha=0.2,
           s = 1 ,
           #s=200*iris.petal_width, # marker size
           #c=len(y), # marker color
           #cmap='viridis'
           )
'''

#plt.title('Scatter Plot with Size(Petal Width) & Color(Petal Length)', fontsize=14)
#plt.xlabel('Sepal Length', fontsize=12)
#plt.ylabel('Sepal Width', fontsize=12)
#plt.colorbar()

plt.show()
