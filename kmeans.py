import pandas as pd
import numpy as np
import random

df=pd.read_csv("Mall_Customers.csv")
print df.shape #information about the df
print "\nSample information\n\n",df.head()
print "\n\nSample stats\n\n",df.describe()
x_lab=df.iloc[:,[3,4]].values #values we need to cluster / form a prediction out of.
# print x_lab
for item in x_lab:
    print item
y_lab=[]

def random_centers(dim,k): #when you first get clusters, the centers are random. This is to get those random centers
    centers=[] #array to store centers
    for i in range(k):
        center=[] #for n clusters, we define n centers. store these as random variables in this array
        for d in range(dim):
            rand=random.randint(0,110)
            center.append(rand)
        centers.append(center) #random centers. Print this array to get original random centers
    return centers

def point_clustering(data, centers, dim, first_cluster=False): #to get the clusters for points in data.
    for point in data:
        nearest_center, nearest_center_distance = 0, None #set the nearest centers for all clusters as 0 / None
        for i in range(0,len(centers)):
            euc_distance=0 #no distance calculation to begin with
            for d in range(0,dim):
                dist=abs(point[d] - centers[i][d]) # calculate squared euclidian distance
                euc_distance+=dist #calculate and add aquared euclidian distance for all the given centers
            euc_distance=np.sqrt(euc_distance) # get euclidean distances
            if nearest_center_distance == None or nearest_center_distance > euc_distance: #set cluster centers
                nearest_center_distance = euc_distance 
                nearest_center = i #cluster number - i
        if first_cluster:
            point.append(nearest_center) #if it is the first cluster, first value in point should be center of this cluster
        else:
            point[-1]=nearest_center #else, replace from the last value in the array. Basically, add a new center for every new cluster that is formed.
    return data #return data values
                
def mean_center(data,centers,dim): #k-means forms means for each of the k clusters as centers.
    print "Centers : ", centers,"\nDim : ",dim
    new_centers=[] #centers get updated each epoch as such, till no randomness remains in data.
    for i in range(len(centers)): #for all clusters
        new_center, total_of_points, number_of_points = [], [] , 0 #define cluster variables - number of elements in a cluster, center of cluster, and total val in the cluster
        for point in data:
            if point[-1] == i: #cluster number
                number_of_points+=1 #increase number of points
                for d in range(0,dim):
                    if d < len(total_of_points): #number of clusters depends on dimensionality as well
                        total_of_points[d]+=point[d] #add dimensions and total number of points in a cluster
                    else:
                        total_of_points.append(point[d]) #append total of values in the cluster for all clusters
        if len(total_of_points) != 0: #so long clusters are available / formed
            for d in range(0,dim):
                print "Point total : ",total_of_points,"Dim : ",d #print them
                new_center.append(total_of_points[d]/number_of_points) #calculation of mean value
            new_centers.append(new_center) #update the new mean for a cluster
        else:
            new_centers.append(centers[i]) #if the mean value is not calcualated, i.e., the total value array is empty
    return new_centers #return new mean centers
def train_k_mean_clustering(data, k, epochs): #train target data for n epochs and k clusters.
    dims=len(data[0])
    print "Data[0] : ",data[0]
    centers=random_centers(dims,k) #form rando clusters at first
    
    clustered_data=point_clustering(data,centers,dims,first_cluster=True) #these are first clusters, so set random data and new centers
    
    for i in range(epochs):
        centers=mean_center(clustered_data, centers, dims) #get new mean for all centers in k clusters
        clustered_data=point_clustering(data, centers, dims, first_cluster=False) #not first clusters, so take and evolve on previous inputs
    
    return centers #return the trained cluster centers. This is the final return.
def predict_k_means_clustering(point, centers):
    dims=len(point)
    center_dims=len(centers[0])
    
    if dims != center_dims:
        raise ValueError('Point given for prediction has ',dims,' dimensions, but centers have ',center_dims,' dimensions')
    nearest_center, nearest_distance = None, None
    
    for i in range(len(centers)):
        euc_dist=0
        for dim in range(1,dims):
            dist=point[dim] - centers[i][dim]
            euc_dist+= dist**2
        euc_dist=np.sqrt(euc_dist)
        if nearest_distance == None or nearest_distance > euc_dist:
            nearest_distance=euc_dist
            nearest_center = i
        print "Center : ",i," Distance : ",euc_dist
    
    return nearest_center            
new_x_lab=x_lab #processing the data to suit the algorithm format
new_list = [] #original data we have is a numpy.ndarray. Convert it to an array of [x,y] values
for i in range(len(new_x_lab)):
    temp=[]
    temp.append(new_x_lab[i][0])
    temp.append(new_x_lab[i][1])
    new_list.append(temp) #new_list has all values of x_lab, except theyre comma separated and not space separated


centers = train_k_mean_clustering(new_list,5,1000) #train data for 1000 epochs, and form 5 clusters of data
print "\n Centers : ",centers #centers to measure goodness of fit against sklearn.cluster metrics

dataset = pd.read_csv('Mall_Customers.csv')
xlab_1 = dataset.iloc[:, [3, 4]].values

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

ylab_1 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(xlab_1)
    ylab_1.append(kmeans.inertia_)
plt.plot(range(1, 11), ylab_1)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('lab')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(xlab_1)


plt.scatter(xlab_1[y_kmeans==0,0],xlab_1[y_kmeans==0,1],s=100,c='blue',label='Cluster 1')
plt.scatter(xlab_1[y_kmeans==1,0],xlab_1[y_kmeans==1,1],s=100,c='green',label='Cluster 2')
plt.scatter(xlab_1[y_kmeans==2,0],xlab_1[y_kmeans==2,1],s=100,c='grey',label='Cluster 3')
plt.scatter(xlab_1[y_kmeans==3,0],xlab_1[y_kmeans==3,1],s=100,c='red',label='Cluster 4')
plt.scatter(xlab_1[y_kmeans==4,0],xlab_1[y_kmeans==4,1],s=100,c='orange',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label= 'Centroids')

plt.title('Customer Clusters')
plt.xlabel('Annual Income in Dollars')
plt.ylabel('SpendingScore(1-100)')
plt.legend()
plt.show()


print "Sklearn Centers : ",
for item in kmeans.cluster_centers_[:]:
    print item,
print "\nCalculated Centers : ",centers