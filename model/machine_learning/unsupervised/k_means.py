import numpy as np

class KMeans:
    def __init__(self):
        self.clusters = None
        
    def distance(self, x1, x2):
        return np.sum((x1 - x2)**2)
    
    def get_clusters(self):
        return self.clusters
    
    def init_clusters(self, X, k, m):
        D = np.full(m, np.inf)
        new_clusters = np.zeros((k, X.shape[1]))
        random_index = np.random.randint(0, m)
        cluster = X[random_index]  
              
        for i in range(k):
            new_clusters[i] = cluster
            if i == k - 1:
                break
            
            next_cluster = None
            max_d = -1
            for j in range(m):
                dist = self.distance(cluster, X[j])
                if D[j] > dist:
                    D[j] = dist
                if D[j] > max_d:
                    max_d = D[j]
                    next_cluster = X[j]
                    
            cluster = next_cluster
            
        return new_clusters
    
    def get_index_clusters(self, X, clusters, k, m):
        index = np.zeros(m, dtype=int)
        
        for i in range(m):
            min_dist = np.inf
            index_cluster = -1
            for j in range(k):
                dist = self.distance(X[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    index_cluster = j
            index[i] = index_cluster
            
        return index
    
    def compute_mean(self, X):
        return np.mean(X, axis=0)
    
    def move_clusters(self, X, clusters, k, m, max_iters, in_place=True):
        if not in_place:
            clusters = clusters.copy()
            
        updated = True
        index = None
        for _ in range(max_iters):
            if not updated:
                break
            updated = False
            # improvement needed
            index_clusters = self.get_index_clusters(X, clusters, k, m)
            index = index_clusters
            
            for j in range(k):
                points = X[index_clusters == j]
                # need improvement
                if len(points) == 0:
                    continue
                next_clusters = self.compute_mean(points)
                if not updated and not np.allclose(clusters[j], next_clusters):
                    updated = True
                clusters[j] = next_clusters
        
        return clusters, index
            
    def compute_score(self, X, index, k, m):
        sum_s = 0
        
        for i in range(m):
            # compute a
            cluster_size = 0
            current_index_cluster = index[i]
            total_dist = 0
            for j in range(m):
                if i == j:
                    continue
                if index[j] == current_index_cluster:
                    cluster_size += 1
                    total_dist += self.distance(X[i], X[j])
                    
            a = total_dist / cluster_size if cluster_size > 0 else 0
            
            # compute b
            other_dists = np.zeros((k, 2))
            other_dists[current_index_cluster] = (np.inf, 0)
            for j in range(m):
                other_index_cluster = index[j]
                if other_index_cluster == current_index_cluster:
                    continue
                dist = self.distance(X[i], X[j])
                other_dists[other_index_cluster] += (dist, 1)
                
            b = np.inf
            for other_dist in other_dists:
                if other_dist[1] == 0:
                    continue
                b = min(b, other_dist[0] / other_dist[1])
        
            den = max(b, a)
            if den > 0:
                sum_s += (b - a) / den
    
        return sum_s / m
        
    def fit(self, X, max_iters=100):
        m = X.shape[0]
        self.clusters = None
        best_score = -1
        for k in range(2, min(m, 10)):
            clusters = self.init_clusters(X, k, m)
            _, index = self.move_clusters(X, clusters, k, m, max_iters)
            score = self.compute_score(X, index, k, m)
            if score > best_score:
                best_score = score
                self.clusters = clusters.copy()
    