import numpy as np

class KMeans:
    def __init__(self):
        self.clusters = None
    
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
            
            dists = np.sum((X - cluster)**2, axis=1)
            D = np.minimum(D, dists)
            cluster = X[np.argmax(D)]
                                
        return new_clusters
    
    def get_index_clusters(self, X, clusters):
        
        # diff[i][j][k] = X[i][k] - clusters[j][k]
        # (m, f) - (k, f)
        # = (m, 1, f) - (1, k, f)
        # = (m, k, f) == diff.shape
        
        diff = X[:, None, :] - clusters[None, :, :]
        
        distances = np.sum(diff**2, axis=2)  
        
        index = np.argmin(distances, axis=1)

        return index
    
    def move_clusters(self, X, clusters, k, m, max_iters, in_place=True):
        if not in_place:
            clusters = clusters.copy()
            
        updated = True
        index = None
        for _ in range(max_iters):
            if not updated:
                break
            updated = False
            index_clusters = self.get_index_clusters(X, clusters)
            
            if np.array_equal(index_clusters, index):
                break
            
            index = index_clusters
            
            for j in range(k):
                mask = (index_clusters == j)
                if not np.any(mask):
                    clusters[j] = X[np.random.randint(0, m)]
                    updated = True
                    continue
                next_clusters = X[mask].mean(axis=0)
                if not updated and not np.allclose(clusters[j], next_clusters):
                    updated = True
                clusters[j] = next_clusters
        
        return clusters, index
            
    def compute_score(self, dist_matrix, index, k, m):
        sum_s = 0
        cluster_masks = [index == c for c in range(k)]
        
        for i in range(m):
            # compute a
            current_index_cluster = index[i]
            same_mask = cluster_masks[current_index_cluster] & (np.arange(m) != i)
            a = np.mean(dist_matrix[i, same_mask]) if np.any(same_mask) else 0
            
            # compute b
            b = np.inf
            for c in range(k):
                if c == current_index_cluster:
                    continue
                other_mask = cluster_masks[c]
                b = min(b, np.mean(dist_matrix[i, other_mask]))
        
            den = max(b, a)
            if den > 0:
                sum_s += (b - a) / den
    
        return sum_s / m
    
    def compute_dist_matrix(self, X):
        # diff[i][j][k] = X[i][k] - X[j][k]
        diff = X[:, None, :] - X[None, :, :]
        
        # dist_matrix[i][j] = self.distance(X[i], X[j])
        dist_matrix = np.sum(diff**2, axis=2)
        
        return dist_matrix
        
    def fit(self, X, max_iters=100):
        m = X.shape[0]
        self.clusters = None
        best_score = -1
        dist_matrix = self.compute_dist_matrix(X)
        for k in range(2, min(m, 10)):
            clusters = self.init_clusters(X, k, m)
            _, index = self.move_clusters(X, clusters, k, m, max_iters)
            score = self.compute_score(dist_matrix, index, k, m)
            if score > best_score:
                best_score = score
                self.clusters = clusters.copy()
    