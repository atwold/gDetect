# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █ ██████ graphutilities ██████                                                                                        █
# █ Purpose : offers convenient (and often reused) functions that construct graphs/trees and computes their attributes  █
# █           more specifically, this module:                                                                           █
# █              - constructs KMSTs (K-Minimum Spanning Tree)                                                           █
# █              - constructs KNNGs (K-Nearest Neighbors Graph)                                                         █
# █              - constructs KNNLs (K-Nearest Neighbors Link)                                                          █
# █              - computes helpful graph attributes, like node degrees, edge counts, etc                               █
# █              - translates between adjacency matrices and edge lists                                                 █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ IMPORTS, MODULES, AND PACKAGES █                                                                                  ▐
# ▌ Purpose : imports the necessary modules and packages                                                                ▐
# ▌              - additionally defines the MST_FILE, a compiled shared object library that constructs a KMST from C    ▐
# ▌                code for faster performance                                                                          ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
import numpy as np
import ctypes
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import f

# define the location of the MST.so file
MST_FILE = os.path.join(os.path.dirname(__file__), "MST.so")



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ KMST, KNNG, AND GRAPH ATTRIBUTES █                                                                                ▐
# ▌ Purpose :                                                                                                           ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ ANYIN() METADATA                                                                                                    ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : anyin                                                                                                 ║
# ║ Purpose     : determines if any elements in a given set are present in another given set                            ║
# ║ Arguments   :                                                                                                       ║
# ║    - set1 (set) : the first set. all elements of this set have their membership checked for the second set. usually ║
# ║                   a predetermined set of possible values denoting the allowable scan statisti                       ║
# ║    - set2 (set) : the second set. Are any of the elements in set1 in set2? Usually the user-specified set of scan   ║
# ║                   statistics                                                                                        ║
# ║ Returns     : a boolean value                                                                                       ║
# ║                  - True if any of the elements in set1 are in set2                                                  ║
# ║                  - False is none of the lements in set1 belong to set2                                              ║
# ║               for our purposes, this function is used to select which scan statistics to compute and return based   ║
# ║               on user input                                                                                         ║
# ║ Author      : written by Alex Wold                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def anyin(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    return any(x in set2 for x in set1)

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# similarity measure: L2 distance
# X is the data in matrix form
def L2dist(X):
    N = X.shape[0]
    G = np.matmul(X, np.transpose(X))
    g = np.diagonal(G)
    L2 = np.reshape(np.tile(g, N), (N, -1), order="F")+\
         np.reshape(np.tile(g, N), (N, -1), order="C")-\
         2*G
    L2 = np.sqrt(L2)
    return L2

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ MSTGRAPH() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : MSTgraph                                                                                              ║
# ║ Purpose     : constructs a K-MST purely in python                                                                   ║
# ║ Arguments   :                                                                                                       ║
# ║    - distance_matrix (<type>) :                                                                                     ║
# ║    - nlig (<type>) :                                                                                                ║
# ║    - ngmax (<type>) :                                                                                               ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# function for kmst
#    DM:   array of inter-node edge lengths
#     N:   nodes are numbered 0, 1, 2, ..., N-1
#   MST:   array in which edge list of MST is placed
#  IMST:   number of edges in array MST
#   CST:   sum of edge lengths of edges of tree
#   NIT:   array of nodes not yet in tree
#  NITP:   number of nodes in array NIT
# JI(i):   node of partial MST  closest to node NIT[i]
# UI(i):   length of edge from NIT(i) to JI(i)
#    KP:   next node to be added to array MST
def MSTgraph(distance_matrix, nlig, ngmax):
    voisi = np.zeros(nlig*nlig, dtype = np.int8)
    borne = 1e20
    lig = int(nlig)
    N = int(nlig)
    numgmax = int(ngmax)
    DM = np.zeros((N, N))
    voisiloc = np.zeros((N, N), dtype = np.int64)
    MST = np.zeros((2, N), dtype = np.int64)
    UI = np.zeros(N)
    JI = np.zeros(N, dtype = np.int64)
    NIT = np.zeros(N, dtype = np.int64)
    
    k = 0
    for i in np.arange(lig):
        for j in np.arange(lig):
            DM[i][j] = distance_matrix[k]
            k = k+1
    for i in np.arange(N):
        DM[i][i] = borne

    for numg in np.arange(1, numgmax+1):
        CST = 0
        NITP = int(N-2)
        KP = int(N-1)
        IMST = int(0)
        for i in np.arange(NITP+1):
            NIT[i] = int(i)
            UI[i] = DM[i][KP]
            JI[i] = int(KP)
        while NITP >= 0:
            for i in np.arange(NITP+1):
                NI = int(NIT[i])
                D = DM[NI][KP]
                if UI[i] > D:
                    UI[i] = D
                    JI[i] = int(KP)
            UK = UI[0]
            for i in np.arange(NITP+1):
                if UI[i] <= UK:
                    UK = UI[i]
                    k = i
            MST[0][IMST] = int(NIT[k])
            MST[1][IMST] = int(JI[k])
            
            IMST = int(IMST+1)
            CST = CST+UK
            KP = int(NIT[k])
            
            UI[k] = UI[NITP]
            NIT[k] = int(NIT[NITP])
            JI[k] = int(JI[NITP])
            NITP = int(NITP-1)
        for i in np.arange(IMST):
            voisiloc[MST[0][i]][MST[1][i]] = numg
            voisiloc[MST[1][i]][MST[0][i]] = numg
            DM[MST[0][i]][MST[1][i]] = borne
            DM[MST[1][i]][MST[0][i]] = borne
    for i in np.arange(lig):
        for j in np.arange(lig):
            a0 = voisiloc[i][j]
            if (a0 > 0) and (a0 <= numgmax):
                voisiloc[i][j] = 1
            else:
                voisiloc[i][j] = 0
    k = 0
    for i in np.arange(lig):
        for j in np.arange(lig):
            voisi[k] = voisiloc[i][j]
            k = k+1
    return np.reshape(voisi, (nlig, nlig), order = "F")
    return voisiloc

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# plotMST function
# G is the MST (an N by N symmetric matrix of 0s and 1s representing the connections between each node)
# data is the data matrix
# nID are the nodes belonging to sample 1
# mID are the nodes belonging to sample 2
def plotMST(G, data, nID, mID):
    edges = np.nonzero(np.triu(G))
    for i in np.arange(edges[0].size):
        if edges[0][i] in nID and edges[1][i] in nID:
            plt.plot([data[edges[0][i], 0], data[edges[1][i], 0]], [data[edges[0][i], 1], data[edges[1][i], 1]], c="red", alpha=1, zorder=1)
        elif edges[0][i] in mID and edges[1][i] in mID:
            plt.plot([data[edges[0][i], 0], data[edges[1][i], 0]], [data[edges[0][i], 1], data[edges[1][i], 1]], c="blue", alpha=1, zorder=1)
        else:
            plt.plot([data[edges[0][i], 0], data[edges[1][i], 0]], [data[edges[0][i], 1], data[edges[1][i], 1]], c="lime", alpha=1, zorder=1)
    plt.scatter(x=data[nID, 0], y=data[nID, 1], marker="o", c="red", sizes=[8], alpha=1)
    plt.scatter(x=data[mID, 0], y=data[mID, 1], marker="o", c="blue", sizes=[8], alpha=1)
    plt.show()

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# as.dist() python counterpart
# we don't really need this function
def as_dist(X):
    return np.tril(X, k=-1)

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     : constructs a K-MST using the python implementation of MSTgraph()                                      ║
# ║ Arguments   :                                                                                                       ║
# ║    - data_matrix (<type>)     :                                                                                     ║
# ║    - distance_matrix (<type>) :                                                                                     ║
# ║    - k (integer)              :                                                                                     ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# mstree function
# note: distance matrix should be in the form returned by L2dist()
# if data matrix is provided, only the euclidean distance is used
# k is the k in K-MST
def kmst(data_matrix=None, distance_matrix=None, k=5):
    if (data_matrix is not None) and (distance_matrix is None):
        distance_matrix = euclidean_distances(data_matrix)
    elif (data_matrix is None) and (distance_matrix is None):
        return None
    nlig = distance_matrix.shape[0]
    distance_matrix = distance_matrix.flatten("F")
    k = 1 if k < 1 else int(k)
    k = 1 if k >= nlig else int(k)
    kmst = MSTgraph(distance_matrix=distance_matrix, nlig=nlig, ngmax=k)
    return kmst

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     : constructs a K-MST from a given data matrix (or a given distance matrix) using the C implementation   ║
# ║               of MSTgraph(). this is generally much faster than its python counterpart, kmst(). this function makes ║
# ║               a call to the shared object library: MST.so                                                           ║
# ║ Arguments   :                                                                                                       ║
# ║    - data_matrix (ndarray, shape (N, d))     :                                                                      ║
# ║    - distance_matrix (ndarray, shape (N, d)) :                                                                      ║
# ║    - k (integer)                             :                                                                      ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# precompiled c function for mstree
# requires the MST.so shared library file
# to create this, run the following line
# g++ -fPIC -shared -o MST.so MST.cpp
def Ckmst(data_matrix=None, distance_matrix=None, k=5):
    if (data_matrix is not None) and (distance_matrix is None):
        distance_matrix = euclidean_distances(data_matrix)
    elif (data_matrix is None) and (distance_matrix is None):
        return None
    
    clibrary = ctypes.CDLL(MST_FILE)
    mstree = clibrary.MSTgraph
    mstree.restype = ctypes.POINTER(ctypes.c_double)
    
    nlig = distance_matrix.shape[0]
    distance_matrix = distance_matrix.flatten("F").ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    voisi = np.zeros(nlig*nlig, dtype=float).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    k = 1 if k < 1 else int(k)
    k = 1 if k >= nlig else int(k)
    
    dbl_ptr = mstree(distance_matrix, ctypes.byref(ctypes.c_int(nlig)), ctypes.byref(ctypes.c_int(k)), voisi)
    kmst = np.ctypeslib.as_array(dbl_ptr, (nlig, nlig)).astype("int64")
    return kmst

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# k nearest neighbors function
# note: distance matrix should be symmetric
# if data matrix is provided, only the euclidean distance is used
# k is the k in K-NNG
def knng(data_matrix=None, distance_matrix=None, k=5):
    if (data_matrix is not None) and (distance_matrix is None):
        distance_matrix = L2dist(data_matrix)
    elif (data_matrix is None) and (distance_matrix is None):
        return None
    nlig = distance_matrix.shape[0]
    k = 1 if k < 1 else int(k)
    k = 1 if k >= nlig else int(k)
    knng = kneighbors_graph(X=distance_matrix, n_neighbors=k, metric="precomputed").toarray("C").astype("int64")
    return knng

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# |G| function
def edge_count(G):
    return 1/2*np.sum(G > 0)

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# return the degrees of all nodes in a graph
# G is the graph (not just an upper or lower triangular matrix)
def degrees(G):
    G = G > 0
    di = np.sum(G, 1)
    return di

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# C
# this is the number of edge pairs that share a common node
def C(G):
    # N = G.shape[0] # number of nodes
    # Gi2 = np.zeros(N)
    # for i in range(0, N):
    #     Gi2[i] = (2*edge_count(G[i]))**2
    # return 1/2*np.sum(Gi2)-edge_count(G)
    di = degrees(G)
    return 1/2*np.sum(di*(di-1))

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# data is the Nxd data matrix
# nID is an array of the nodes (observations) in sample 1
# mID is an array of the nodes (observations) in sample 2
# the hotelling test performs 
def HotellingTest(data, nID, mID, equal_cov=False):
    N = data.shape[0]
    d = data.shape[1]
    X = data[nID, :]
    Y = data[mID, :]
    n = X.shape[0]
    m = Y.shape[0]
    
    Xbar = np.mean(X, 0).reshape((d, 1))
    Ybar = np.mean(Y, 0).reshape((d, 1))
    
    SX = np.cov(X, rowvar=False)
    SY = np.cov(Y, rowvar=False)
    
    if equal_cov == True:
        teststat = ((n*m)/(n+m))*(((Xbar-Ybar).T)@(np.linalg.inv(((n-1)*SX+(m-1)*SY)/(n+m-2)))@(Xbar-Ybar))[0, 0]
        teststat = ((n+m-d-1)/((n+m-2)*d))*teststat
        pval = f.sf(teststat, d, n+m-d-1)
    else:
        teststat = (((Xbar-Ybar).T)@(np.linalg.inv((SX/n)+(SY/m)))@(Xbar-Ybar))[0, 0]
        pval = chi2.sf(teststat, d)
    
    results = {"hotelling.test.statistic" : teststat, "hotelling.pval" : pval}
    return(results)



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ KNNL █                                                                                                            ▐
# ▌ Purpose : provides functions for constructing KNNLs from data sequences with repeated observations (the discrete    ▐
# ▌           setting)                                                                                                  ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# this function returns the unsorted unique (by row) matrix from a given matrix
def unsorted_unique(matrix):
    sorted_unique_matrix, ind = np.unique(matrix, axis=0, return_index=True)
    unsorted_unique_matrix = sorted_unique_matrix[np.argsort(ind)]
    return unsorted_unique_matrix

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# this function generates an unsorted listing of unique rows in the data matrix
# then it matches the rows in the data matrix with those in the unique matrix
# it returns the indices of the matched rows
# basically, this returns the indices of the unique matrix that can be used to reconstruct the data matrix
def order_obs(data_matrix):
    unique, ind, inv = np.unique(data_matrix, return_index=True, return_inverse=True, axis=0)
    unsorted_unique_inv = ind.argsort()
    inv2 = inv.copy()
    for i in np.arange(unsorted_unique_inv.size):
        inv2[inv==unsorted_unique_inv[i]] = i
    return inv2

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# this function takes the sample indices and either a data matrix or array of ordered observations
# and returns the counts as a K by 2 matrix
# nID is an array of the indices for sample 1 in the data
# mID is an array of the indices for sample 2 in the data
def getCounts(nID, mID, data_matrix=None, order=None):
    if (data_matrix is None) and (order is None):
        return None
    elif (data_matrix is not None) and (order is None):
        order = order_obs(data_matrix)
    
    unique_ordered_obs = np.unique(order)
    K = unique_ordered_obs.size
    counts = np.zeros((K, 2), dtype=np.int64)
    for i in np.arange(K):
        counts[i, 0] = np.count_nonzero(order[nID]==unique_ordered_obs[i])
        counts[i, 1] = np.count_nonzero(order[mID]==unique_ordered_obs[i])
    return counts

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# depth first search
def dfs(s, visited, adj):
    visited[s] = 1
    for i in np.arange((adj[s]).size):
        if visited[adj[s][i]] == 0:
            visited = dfs(adj[s][i], visited, adj)
    return visited

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def nnlink_com(distance_matrix):
    nodes = distance_matrix.shape[0] # the number of nodes in the graph
    adj = [[] for _ in np.arange(nodes)] # create a list of empty lists
    com = [[] for _ in np.arange(nodes)] # create a list of empty lists
    components = 0
    for i in np.arange(nodes): 
        edgeid = (np.asarray(distance_matrix[i, ]==np.delete(distance_matrix[i, ], i).min()).nonzero())[0] # find the "nearest" (min distance) node
        adj[i].extend(edgeid)
        for j in np.arange(edgeid.size):
            adj[edgeid[j]].append(i) # preserve symmetry, will produce duplicates
    edgenum = 0
    for i in np.arange(nodes):
        adj[i] = np.unique(adj[i]) # sort adjacency nodes and remove duplicates
        edgenum += adj[i].size
    edgenum //= 2
    edgenum = np.int64(edgenum)
    visited = np.zeros((nodes, ), dtype=np.int64)
    for i in np.arange(nodes):
        if visited[i] == 0:
            visited = dfs(i, visited, adj)
            com[components] = (visited.nonzero())[0]
            components += 1
    if components > 0:
        for i in np.arange(components, 0, -1, dtype=np.int64):
            com[i] = np.setdiff1d(com[i], com[i-1], assume_unique=False) # np.setdiff1d sorts (ascending) if assume_unique=False
    (com[0]).sort()
    com = com[:components]
    return {"com" : com, "adj" : adj, "edgenum" : edgenum}

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def getcomdist(g1, g2, distance_matrix):
    open_mesh = np.ix_(g1, g2) # construct index arrays that will index the cross product of g1 and g2
    tempdis = distance_matrix[open_mesh[0], open_mesh[1]]
    mindis = tempdis.min()
    minid = np.asarray(tempdis==tempdis.min()).nonzero()
    minID = np.concatenate((np.array(g1)[minid[0]], np.array(g2)[minid[1]]), dtype=np.int64).reshape((-1, 2))
    return {"mindis" : mindis, "minID" : minID}

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def nnlink(distance_matrix):
    nodes = distance_matrix.shape[0]
    temp = nnlink_com(distance_matrix)
    com, adj, edgenum = temp["com"], temp["adj"], temp["edgenum"]
    components = len(com)
    while True:
        if components == 1:
            E = np.zeros((edgenum, 2), dtype=np.int64)
            e = 0
            for i in np.arange(nodes):
                if adj[i].size > 0:
                    for j in np.arange(adj[i].size):
                        E[e, 0] = i
                        E[e, 1] = adj[i][j]
                        adj[E[e, 1]] = np.setdiff1d(adj[E[e, 1]], i) # handle symmetry
                        e += 1
            break
        else:
            newdist = np.zeros((components, components))
            id_edges_candidate = np.zeros((components, components), dtype=np.int64)
            edges_candidate = [[] for _ in np.arange(components*(components-1)//2)]
            edge_com = 0
            for i in np.arange((components-1)):
                for j in np.arange((i+1), components):
                    tt = getcomdist(com[i], com[j], distance_matrix)
                    newdist[i, j] = newdist[j, i] = tt["mindis"]
                    edges_candidate[edge_com] = tt["minID"]
                    id_edges_candidate[i, j] = id_edges_candidate[j, i] = edge_com
                    edge_com += 1
            temp2 = nnlink_com(newdist)
            adj2 = temp2["adj"]
            for i in np.arange(components):
                if adj2[i].size > 0:
                    for j in np.arange(adj2[i].size):
                        e1 = i
                        e2 = adj2[i][j]
                        id_com = id_edges_candidate[e1, e2]
                        addedge = edges_candidate[id_com]
                        for k in np.arange(addedge.shape[0]):
                            adj[addedge[k, 0]] = np.append(adj[addedge[k, 0]], addedge[k, 1])
                            adj[addedge[k, 1]] = np.append(adj[addedge[k, 1]], addedge[k, 0])
                        adj2[e2] = np.setdiff1d(adj2[e2], i)
            edgenum = 0
            for i in np.arange(nodes):
                adj[i] = np.unique(adj[i])
                edgenum += adj[i].size
            edgenum //= 2
            com2 = temp2["com"]
            components = len(com2)
            newcom = [[] for _ in np.arange(components)]
            for i in np.arange(components):
                for j in np.arange(com2[i].size):
                    newcom[i].extend(com[com2[i][j]])
                newcom[i].sort()
            com = newcom
    return E

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def nnl(data_matrix=None, distance_matrix=None, k=5):
    if (data_matrix is not None) and (distance_matrix is None):
        unsorted_unique_data_matrix = unsorted_unique(data_matrix)
        distance_matrix = euclidean_distances(unsorted_unique_data_matrix)
    elif (data_matrix is None) and (distance_matrix is None):
        return None
    distance_matrix_copy = np.copy(distance_matrix)
    maxdis = 1e5*np.max(distance_matrix_copy) # max distance
    nodes = distance_matrix_copy.shape[0]
    tempE = [[] for _ in np.arange(k)]
    E = np.empty((0, 2), dtype=np.int64)
    for i in np.arange(k):
        tempE[i] = nnlink(distance_matrix_copy)
        E = np.concatenate((E, tempE[i]), axis=0, dtype=np.int64)
        E = unsorted_unique(E)
        if i == k:
            break
        for e in np.arange(tempE[i].shape[0]):
            e1 = tempE[i][e, 0]
            e2 = tempE[i][e, 1]
            distance_matrix_copy[e1, e2] = distance_matrix_copy[e2, e1] = maxdis
    # return E
    # note that E is an edge list, not an edge matrix
    return edgelist_to_edgematrix(E, nodes=nodes, directed=False)



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ ADJACENCY MATRIX HANDLING █                                                                                       ▐
# ▌ Purpose : provides functions for managing and testing adjacency matrices. more specifically:                        ▐
# ▌              - testing if a matrix is symmetric (good for checking if a given graph is undirected)                  ▐
# ▌              - converting an adjacency matrix into a list of edges indexed by node                                  ▐
# ▌              - converting an adjacency matrix (edge matrix) into an edge list                                       ▐
# ▌              - converting an edge list back into an adjacency matrix (edge matrix)                                  ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ EDGEMATRIX_TO_EDGELIST() METADATA                                                                                   ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : edgematrix_to_edgelist                                                                                ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : Alex Wold                                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# convert an edge matrix into an edge list
def edgematrix_to_edgelist(edgematrix):
    if issymmetric(edgematrix):
        edgeind = (np.triu(edgematrix)).nonzero()
        return np.concatenate((edgeind[0], edgeind[1])).reshape((-1, 2), order="F")
    else:
        edgeind = edgematrix.nonzero()
        return np.concatenate((edgeind[0], edgeind[1])).reshape((-1, 2), order="F")

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ EDGELIST_TO_EDGEMATRIX() METADATA                                                                                   ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : edgelist_to_edgematrix                                                                                ║
# ║ Purpose     : converts an edge list into an edge matrix                                                             ║
# ║ Arguments   :                                                                                                       ║
# ║    - edgelist (ndarray, shape: (edgenum, 2)) : a matrix with two columns representing all of the (i, j) pairs of    ║
# ║                                                edges in the graph this structure represents. that is, the first     ║
# ║                                                column is the i component which represents the first node in an edge ║
# ║                                                e_{ij}. the second column is the j component which represents the    ║
# ║                                                second node in that same edge e_{ij}. for undirected graphs, (i, j)  ║
# ║                                                is the same as (j, i) and should not be included within edgelist. in ║
# ║                                                other words, only distinct pairs of (i, j) should be included within ║
# ║                                                edgelist. there should be edgenum rows in this matrix                ║
# ║                                                   - edgenum is the total number of edges in the graph               ║
# ║    - N (integer)                             : the total number of observations in the data sequence (the total     ║
# ║                                                number of nodes in the graph)                                        ║
# ║    - directed (boolean, optional)            : should be True if the graph edgelist represents is a directed, i.e.  ║
# ║                                                edge e_{ij} is different from e_{ji}. should be False if the graph   ║
# ║                                                edgelist represents is undirected, i.e. edge_{ij} is the same as     ║
# ║                                                edge_{ji}                                                            ║
# ║                                                   - Default is False (we assume edgelist represents an undirected   ║
# ║                                                     graph)                                                          ║
# ║ Returns     : E, an ndarray of shape (N, N) representing the adjacency matrix of the same graph that edgelist       ║
# ║               represented. if directed=True, the edge matrix will not be symmetric. if directed=False, the          ║
# ║               edge matrix will be symmetric                                                                         ║
# ║ Author      : Alex Wold                                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def edgelist_to_edgematrix(edgelist, N, directed=False):
    E = np.zeros((N, N), dtype=np.int64)
    if directed == False:
        for i in np.arange(edgelist.shape[0]):
            E[edgelist[i, 0], edgelist[i, 1]] = 1
            E[edgelist[i, 1], edgelist[i, 0]] = 1
    else:
        for i in np.arange(edgelist.shape[0]):
            E[edgelist[i, 0], edgelist[i, 1]] = 1
    return E

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ EDGEMATRIX_TO_EDGEBYNODE() METADATA                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : edgematrix_to_edgebynode                                                                              ║
# ║ Purpose     : converts an N by N edgematrix into lists of connected edges indexed by node                           ║
# ║ Arguments   :                                                                                                       ║
# ║    - edgematrix (ndarray, shape: (N, N)) : a symmetric N by N adjacency matrix                                      ║  
# ║                                               - N is the number of observations (nodes)                             ║
# ║ Returns     : a python list of length N whose elements are ndarrays; the length of these ndarrays is equal to the   ║
# ║               degree of the respective node in the graph                                                            ║
# ║                  - the indices of this list represent the nodes of the graph (0 to N-1) for nodes 1 to N            ║
# ║                  - the elements of this list are ndarrays of integers that (together with the index) represent the  ║
# ║                    edges of the graph                                                                               ║
# ║               for instance, if element 0 contains the ndarray [1, 2, 7], then the first node has edges with the     ║
# ║               second, third, and eigth nodes. In this case, edges e_{1, 2}, e_{1, 3}, and e_{1, 8} represented by   ║
# ║               the pairs (0, 1), (0, 2), and (0, 7) exist in the graph. Essentially this is a list that, for each    ║
# ║               node, stores an array of additional nodes that the index-specified node is connected to               ║
# ║ Author      : written by Alex Wold                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def edgematrix_to_edgebynode(edgematrix):
    N = (edgematrix.shape)[0] # N is the total number of nodes in the graph
    # create a container list of length N to hold the list of edges each node is connected to
    edgebynode = [[] for _ in np.arange(N)]
    # for each node i, get all other nodes that are connected to this node, store as an ndarray in element i
    for i in np.arange(N):
        edgebynode[i] = ((edgematrix[i, :]).nonzero())[0]
    return edgebynode

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    :                                                                                                       ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# this function returns true if the given matrix is symmetric
# this function returns false if the given matrix is not symmetric
def issymmetric(matrix):
    # check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # check if the matrix is equal to its transpose
    return np.array_equal(matrix, matrix.transpose())





















