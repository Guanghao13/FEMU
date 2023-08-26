import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

SLICE_LEN = 1024
tt_pgs = 0
K = 5
common_params = {
    "n_init": "auto",
    "random_state": 1037,
}

def readfile(Path):
    global tt_pgs
    X = []
    i = -1
    fo = open(Path, "r", errors = "replace")
    for line in fo.readlines():
        i += 1
        X.append([])
        line = line.strip()
        line = line.split("  ")
        for k in range(len(line)):
            line[k] = line[k].split(" ")
        # X[i].append(int(line[0][0]))        
        X[i].append(int(line[1][0])-int(line[2][0]))  #w
        X[i].append(int(line[2][0])-int(line[3][0]))  #w
        X[i].append(int(line[1][1])-int(line[2][1]))  #r
        X[i].append(int(line[2][1])-int(line[3][1]))  #r
        X[i].extend( [int(x) for x in line[4]] )
    tt_pgs += (i+1) * SLICE_LEN
    fo.close()
    return X

def predict(X):
    global kmeans
    kmeans = KMeans(n_clusters=5, **common_params)
    y_pred = kmeans.fit_predict(X)
    return y_pred

def normalize(X):
    scale = np.array(([X[:,i].max()-X[:,i].min() for i in range(X.shape[1])]))
    X = X/scale 
    X = np.nan_to_num(X)
    for i in [0,1,4]:  #write time gap and frequency
        X[:,i] = X[:,i] * 4
    return X

def reduce(X_norm):
    global pca
    global kernel_pca
    pca = PCA(n_components=2)
    X_reduced =  pca.fit_transform(X_norm)
    # print(pca.explained_variance_ratio_)
    kernel_pca = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True)
    X_test_kernel_pca = kernel_pca.fit(X_norm).transform(X_norm)
    return X_reduced, X_test_kernel_pca

def Plot(X_reduced):
    # Put the result into a color plot
    h = 0.02
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    # Plot all the samples in X
    plt.plot(X_reduced[:, 0], X_reduced[:, 1], "k.", markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",  # Plot the centroids as a white X
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the SCLICEs’ timestamps and frequency (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

def change_label():
    centroids_inv = kernel_pca.inverse_transform(kmeans.cluster_centers_)
    clusters_index = np.argsort(centroids_inv[:,4]) 
    return clusters_index

def Output(Out):
    fo = open('/home/guanghao/Documents/FEMU/hw/femu/bbssd/clusters','w')
    for i in range(Out.shape[0]-1):
        fo.write(str(Out[i,-1])+'\n')
    fo.write(str(Out[-1,-1]))
    fo.close()

if __name__ == "__main__":
    while True:
        Path = "/home/guanghao/Documents/FEMU/hw/femu/bbssd//femu_timetbl"
        X = readfile(Path)
        X_norm = normalize(np.array(X))
        # print(X_norm)
        X_reduced, X_kernel_pca_reduced = reduce(X_norm)
        y_pred = predict(X_kernel_pca_reduced)
        y = y_pred.reshape(len(y_pred), 1)
        Out = np.hstack((X, y))
        Plot(X_kernel_pca_reduced)
        clusters_label = change_label()
        for i in range(Out.shape[0]):
            Out[i,-1] = clusters_label[Out[i,-1]]
        Output(Out)
        time.sleep(60)
        # plt.show()
        # print(Out)
