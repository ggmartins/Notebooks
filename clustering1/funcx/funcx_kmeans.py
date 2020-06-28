funcx_kmeans.py 
from funcx.sdk.client import FuncXClient

fxc = FuncXClient()

def funcx_sum(items):
    return sum(items)

def funcx_KMeans(n_clusters, data):
    from sklearn.cluster import KMeans as sklearn_cluster_KMeans
    k_means = sklearn_cluster_KMeans(n_clusters=n_clusters)
    k_means.fit(data)
    return k_means, k_means.predict(data)
    

#func_uuid = fxc.register_function(funcx_sum,
#                                  description="A summation function")
#print(func_uuid)

func_uuid = fxc.register_function(funcx_KMeans,
                                  description="A wrapper for sklearn.cluster.Kmeans")

print(func_uuid)
