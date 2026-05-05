import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# Two simple data points (e.g., customer age and income in thousands)
point_a = np.array([[25, 50]])
point_b = np.array([[30, 55]])
point_c = np.array([[60, 120]])

# Euclidean distance
dist_ab = euclidean_distances(point_a, point_b)
dist_ac = euclidean_distances(point_a, point_c)

print("Euclidean Distance A to B:", dist_ab[0][0])
print("Euclidean Distance A to C:", dist_ac[0][0])
