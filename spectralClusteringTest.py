import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D


with open('coords76.pickle', 'rb') as handle:
    new_coords = pickle.load(handle)

k = 3
# SIGMA = 1.0

# M = np.random.rand(2, 10).T
M = np.array(new_coords)

# Rows are the ARTISTS

num_artists = M.shape[0]

sigmas = [2**(i) for i in range(-6,2)]

def spectral_clustering(sigs, plotClusters = False,plotY = False, verbose = False):
    best_sigma, best_score = None, None

    for SIGMA in sigs:
        A = np.zeros((num_artists, num_artists))

        for i in range(num_artists):
            for j in range(num_artists):
                A[i,j] = math.exp(-(1.0/(2*(SIGMA**2)))*sum((M[i,:]-M[j,:])**2)) if i != j else 0

        if verbose:
            print("A:")
            print()
            print(A)
            print()

        D = np.zeros((num_artists, num_artists))
        for i in range(num_artists):
            D[i,i] = sum(A[i,:])

        DminusHalf = np.zeros((num_artists, num_artists))
        for i in range(num_artists):
            DminusHalf[i, i] = 1.0/((D[i,i])**0.5)

        if verbose:
            print("D:")
            print()
            print(D)
            print()

        L = np.dot(np.dot(DminusHalf, A), DminusHalf)

        if verbose:
            print("L:")
            print()
            print(L)
            print()

        w,v = np.linalg.eigh(L)

        if verbose:
            print("W", w)
            print()
            print("V:", v)

        top_k_eigenvectors = [v[:,-(1+i)] for i in range(k)]

        if verbose:
            print("Largest Eigenvectors:")
            print()
            print(top_k_eigenvectors)

            print("DOT PRODUCT:")
            print(np.dot(top_k_eigenvectors[0], top_k_eigenvectors[1]))
            print()

        X = np.array([top_k_eigenvectors[i] for i in range(k)]).T

        if verbose:
            print("X:")
            print()
            print(X)
            print()

        Y = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i,j] = X[i,j]*1.0/(sum(np.square(X[i,:]))**0.5)

        if verbose:
            print("Y:")
            print()
            print(Y)
            print()

        kmeans = KMeans(n_clusters=k).fit(Y)
        kmeans_score = kmeans.score(Y)
        print("SCORE WITH SIGMA = " + str(SIGMA) + ":", kmeans_score)

        if best_sigma is None:
            best_sigma, best_score = SIGMA, kmeans_score

        if kmeans_score > best_score:
            best_sigma, best_score = SIGMA, kmeans_score

        if verbose:
            print("M:")
            print()
            print(M)
            print()

        colors = ["r", "b", "g", "y", "c", "k", "m", "w"]

        if plotClusters:
            for cluster_num in range(k):
                cluster_indices = []
                for i in range(M.shape[0]):
                    if kmeans.labels_[i] == cluster_num:
                        cluster_indices.append(i)
                M_submatrix_indices = np.array([list(M[i]) for i in cluster_indices])
                plt.plot(M_submatrix_indices[:, 0], M_submatrix_indices[:, 1], colors[cluster_num]+"o")

            plt.title("Spectral Clustering, k = " + str(k) + ", SIGMA = " + str(SIGMA))
            plt.show()

        if plotY:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for cluster_num in range(k):
                cluster_indices = []
                for i in range(Y.shape[0]):
                    if kmeans.labels_[i] == cluster_num:
                        cluster_indices.append(i)

                Y_submatrix_indices = np.array([list(Y[i]) for i in cluster_indices])
                ax.scatter(Y_submatrix_indices[:, 0], Y_submatrix_indices[:, 1], Y_submatrix_indices[:, 2], c=colors[cluster_num], marker='o')

            plt.xlim((-1,1))
            plt.ylim((-1,1))
            # plt.zlim((-1,1))
            plt.show()

    return best_sigma, best_score

best_sig, best_sc = spectral_clustering(sigmas)

print("BEST SIG:", best_sig)

sig, sc = spectral_clustering([best_sig], plotClusters=True,plotY=True)