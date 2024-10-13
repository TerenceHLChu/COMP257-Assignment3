# Student name: Terence Chu
# Student number: 301220117

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from operator import itemgetter

# Load Olivetti faces dataset
olivetti = fetch_olivetti_faces()

print('Olivetti faces data shape', olivetti.data.shape) # 64 × 64
print('Olivetti faces target shape', olivetti.target.shape)

X = olivetti.data
y = olivetti.target

print('\nPixel values:\n', X)
print('Pixel maximum:', X.max())
print('Pixel minimum:', X.min())
print('Data is already normalized')

# Display the first 12 images of the dataset
plt.figure(figsize=(7,7))

for i in range(12):
    plt.subplot(3, 4, i+1) # 3 rows, 4 columns
    plt.imshow(X[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

# Split the dataset into train, test, and validation sets
# stratify=y ensures the class distribution in each split set is the same as the original dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=17)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.25, random_state=17)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_valid shape:', X_valid.shape)

# Train an SVC classifier 
svc_classifier = SVC(kernel='linear')

# Define cross-validation that splits the data into 5 folds, ensuring the class distribution in each fold is the same as the original dataset
stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=17) 

# Calculate the cross-validation score to evaluate the classifier's performance (accuracy in this case)
cross_val_sore = cross_val_score(svc_classifier, X_train, y_train, cv=stratified_k_fold, scoring='accuracy')
print('\nCross validation score of the 5 folds:', cross_val_sore)

# Train the classifier
svc_classifier.fit(X_train, y_train)

# Assess the classifer on the validation data
score_valid = svc_classifier.score(X_valid, y_valid)
print('\nModel accuracy on validation set', score_valid)

# ---------- Dimensionality reduction with AHC - Euclidean metric - average linkage ---------- 
# Find the number of clusters that generate the highest silhouette score (Euclidean metric and average linkage)
# Try number of clusters between 3 and 109
euclidean_silhouette_scores = []
print('\nAgglomerative Hierarchical Clustering Silhouette scores - Euclidean metric - average linkage')

for i in range(3, 110):
    # Build the classifier
    clf = AgglomerativeClustering(n_clusters=i, metric="euclidean", linkage="average")
    
    # Train model with X_train and cluster X_train's data points
    clf.fit(X_train)
    
    # Retrieve the cluster label for each of X_train's data point
    data_labels = clf.labels_

    # Calculate silhouette score
    sil_score = silhouette_score(X_train, data_labels)
    
    # Add the number of clusters and associated silhouette score to the list as a tuple
    euclidean_silhouette_scores.append((i, sil_score))
    
    print(f'Number of clusters | Silhouette Score: {i} | {sil_score}')

# Retrieve the number of clusters that generated the highest silhouette score
# key=itemgetter(1) instructs the max function to find the maximum based on the second element (index 1) of the tuples (which are the silhouette scores)
max_euclidean_silhouette_scores = max(euclidean_silhouette_scores, key=itemgetter(1)) 
print('(Best clusters, top silhouette score):', max_euclidean_silhouette_scores)

# Extract the number of clusters associated with the highest silhouette score
# This corresponds to the number of dimensions to reduce to 
euclidean_reduced_dim = max_euclidean_silhouette_scores[0]
print('Number of dimensions reduced to:', euclidean_reduced_dim)

# ---------- Dimensionality reduction with AHC - Minkowski metric - average linkage ---------- 
minkowski_silhouette_scores = []
print('\nAgglomerative Hierarchical Clustering Silhouette scores - Minkowski metric - average linkage')

for i in range(3, 110):
    clf = AgglomerativeClustering(n_clusters=i, metric="minkowski", linkage="average")
    
    clf.fit(X_train)
    data_labels = clf.labels_
    
    sil_score = silhouette_score(X_train, data_labels)
    
    minkowski_silhouette_scores.append((i, sil_score))
    
    print(f'Number of clusters | Silhouette Score: {i} | {sil_score}')
    
max_minkowski_silhouette_scores = max(minkowski_silhouette_scores, key=itemgetter(1))
print('(Best clusters, top silhouette score):', max_minkowski_silhouette_scores)

minkowski_reduced_dim = max_minkowski_silhouette_scores[0]
print('Number of dimensions reduced to:', minkowski_reduced_dim)

# ---------- Dimensionality reduction with AHC - Cosine metric - average linkage ---------- 
cosine_silhouette_scores = []
print('\nAgglomerative Hierarchical Clustering Silhouette scores - Cosine metric - average linkage')

for i in range(3, 130):
    clf = AgglomerativeClustering(n_clusters=i, metric="cosine", linkage="average")
    
    clf.fit(X_train)
    data_labels = clf.labels_
    
    sil_score = silhouette_score(X_train, data_labels)
    
    cosine_silhouette_scores.append((i, sil_score))
    
    print(f'Number of clusters | Silhouette Score: {i} | {sil_score}')
 
max_cosine_silhouette_scores = max(cosine_silhouette_scores, key=itemgetter(1))
print('(Best clusters, top silhouette score):', max_cosine_silhouette_scores)

cosine_reduced_dim = max_cosine_silhouette_scores[0]
print('Number of dimensions reduced to:', cosine_reduced_dim)

# ---------- Train classifiers with reduced dimensions ----------
# Reduce to 94 dimensions
kmeans_reduced_dims_94 = KMeans(n_clusters=euclidean_reduced_dim, random_state=17)

# fit applies the KMeans clustering to X_train
# transform X_train to 94 clusters 
# The data only needs to be fitted once - only transforms are needed subsequently 
X_train_reduced_dims_94 = kmeans_reduced_dims_94.fit_transform(X_train) # Number of dimensions reduced to 94

X_valid_reduced_dims_94 = kmeans_reduced_dims_94.transform(X_valid) # Number of dimensions reduced to 94

X_test_reduced_dims_94 = kmeans_reduced_dims_94.transform(X_test) # Number of dimensions reduced to 94

print('\nShape of X_train_reduced_dims:', X_train_reduced_dims_94.shape) 
print('Shape of X_valid_reduced_dims:', X_valid_reduced_dims_94.shape)
print('Shape of X_test_reduced_dims:', X_test_reduced_dims_94.shape)

# Calculate the cross-validation score to evaluate the classifier's performance (accuracy in this case)
cross_val_sore_reduced = cross_val_score(svc_classifier, X_train_reduced_dims_94, y_train, cv=stratified_k_fold, scoring='accuracy')
print('\nCross validation score of the 5 folds (94 dimensions):', cross_val_sore_reduced)

# Train the classifier
svc_classifier.fit(X_train_reduced_dims_94, y_train)

# Assess the classifer on the validation data
score_valid = svc_classifier.score(X_valid_reduced_dims_94, y_valid)
print('\nModel accuracy on validation set (94 dimensions)', score_valid)

# Reduce to 117 dimensions
kmeans_reduced_dims_117 = KMeans(n_clusters=cosine_reduced_dim, random_state=17)

# fit applies the KMeans clustering to X_train
# transform X_train to 94 clusters 
# The data only needs to be fitted once - only transforms are needed subsequently 
X_train_reduced_dims_117 = kmeans_reduced_dims_117.fit_transform(X_train) # Number of dimensions reduced to 117

X_valid_reduced_dims_117 = kmeans_reduced_dims_117.transform(X_valid) # Number of dimensions reduced to 117

X_test_reduced_dims_117 = kmeans_reduced_dims_117.transform(X_test) # Number of dimensions reduced to 117

print('\nShape of X_train_reduced_dims:', X_train_reduced_dims_117.shape) 
print('Shape of X_valid_reduced_dims:', X_valid_reduced_dims_117.shape)
print('Shape of X_test_reduced_dims:', X_test_reduced_dims_117.shape)

# Calculate the cross-validation score to evaluate the classifier's performance (accuracy in this case)
cross_val_sore_reduced = cross_val_score(svc_classifier, X_train_reduced_dims_117, y_train, cv=stratified_k_fold, scoring='accuracy')
print('\nCross validation score of the 5 folds (117 dimensions):', cross_val_sore_reduced)

# Train the classifier
svc_classifier.fit(X_train_reduced_dims_117, y_train)

# Assess the classifer on the validation data
score_valid = svc_classifier.score(X_valid_reduced_dims_117, y_valid)
print('\nModel accuracy on validation set (117 dimensions)', score_valid)

# # ---------- Visualize images per cluster ----------
# clf = AgglomerativeClustering(n_clusters=94, metric="euclidean", linkage="average")

# # Train model with X_train and cluster X_train's data points
# clf.fit(X_train)

# # Retrieve the cluster label for each of X_train's data point
# data_labels = clf.labels_

# data_point_cluster_labels_list = []

# for i in range(len(data_labels)):
#     data_point_cluster_labels_list.append((i, data_labels[i])) # Generate tuples of data points (images) and cluster labels
#                                                                # Example: [(0, 44), (1, 26)] - Zeroth image categorized into the 44th cluster; first image into the 26th cluster  

# # Sort the tuples by the second element (index 1) (i.e., the cluster labels)
# # Example: [(11, 0), (126, 0), (148, 0), (27, 1), (209, 1)] - 11th, 126th, and 148th images categorized into the zeroth cluster; 27th and 209th images categorized into the first cluster
# data_point_cluster_labels_list_sorted_by_cluster = sorted(data_point_cluster_labels_list, key=itemgetter(1))

# print(data_point_cluster_labels_list_sorted_by_cluster)

# # Dictionary to store the counts of the second value in the tuple
# count_dict = {}

# # Iterate through the list and count occurrences of the second value in the tuples
# for tup in data_point_cluster_labels_list_sorted_by_cluster:
#     second_value = tup[1]
#     if second_value in count_dict:
#         count_dict[second_value] += 1
#     else:
#         count_dict[second_value] = 1

# # Print the counts
# print('Dictionary of counts', count_dict) # Example: {0: 3, 1: 2} - Three images were categorized into the zeroth cluster; two images were categorized into the first cluster

# rows = 1
# list_index = 0

# for j in range (94):
#     fig = plt.figure(figsize=(7,10))
#     plot_index = 1
#     columns = count_dict[j] # The number of columns correspond to the number of images categorized into each cluster
    
#     for i in range(columns):
    
#         # print(plot_index)
#         # print(list_index)
#         # print(columns, '\n')
        
#         # At each iteration, move to the next index in the sorted list and retrieve the image number (index)
#         face_index = data_point_cluster_labels_list_sorted_by_cluster[list_index][0]
        
#         # print(face_index)
        
#         # Display the face image
#         fig.add_subplot(rows, columns, plot_index)
#         plt.imshow(X_train[face_index].reshape(64, 64), cmap='gray')
        
#         plt.xticks([])
#         plt.yticks([])
        
#         plot_index = plot_index + 1
#         list_index = list_index + 1