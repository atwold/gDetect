# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █ ██████ continuousonline ██████                                                                                      █
# █ Purpose : provides the main and supporting functions for detecting single change-points in the online setting       █
# █              -  gstream: detects single change-points in a sequence of sequentially generated observations          █
# █           every other function is not meant to be called by the user, rather, they are subfunctions that support    █
# █           the functionality of gchangepoint and gchangeinterval                                                     █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ IMPORTS, MODULES, AND PACKAGES █                                                                                  ▐
# ▌ Purpose : imports the necessary modules and packages                                                                ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
from . import graphutilities as gu # use relative reference for an internal import
import numpy as np
from scipy.stats import norm, chi2
import scipy.integrate as integrate
from functools import wraps # for decorating critical value functions



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ CONTINUOUS ONLINE CHANGE-POINT DETECTION █                                                                        ▐
# ▌ Purpose : defines functions for detecting change-points in the online setting                                       ▐
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
def gstream(distance_matrix, L, N0, k, statistic={"all"}, n0=None, n1=None, arl=10000, alpha=.05, skew_corr=False, asymp=False):
    if n0 is None: n0 = (.3*L)-1
    if n1 is None: n1 = (.7*L)-1
    if N0 < L: return None
    if n0 < 1: n0 = 1
    if n1 > L-3: n1 = L-3
    if n0 > n1 or n1 < n0:
        n0 = 1
        n1 = L-2
    n0 = np.int64(np.ceil(n0))
    n1 = np.int64(np.floor(n1))
    N = (distance_matrix.shape)[0]
    r1 = {}
    r1["scanZ"] = getscanZ(distance_matrix, L, N0, k, statistic, n0, n1, N)
    r1["b"] = getb(distance_matrix, L, N0, k, statistic, n0, n1, arl, alpha, skew_corr, asymp)

    r1["stops"] = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        r1["stops"]["original"] = np.where(r1["scanZ"]["original"]>r1["b"]["original"])[0]
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        r1["stops"]["weighted"] = np.where(r1["scanZ"]["weighted"]>r1["b"]["weighted"])[0]
    if gu.anyin({"all", "max", "m"}, statistic):
        r1["stops"]["max_type"] = np.where(r1["scanZ"]["max_type"]>r1["b"]["max_type"])[0]
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        r1["stops"]["generalized"] = np.where(r1["scanZ"]["generalized"]>r1["b"]["generalized"])[0]
    
    return r1



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ SCAN STATISTICS █                                                                                                 ▐
# ▌ Purpose :                                                                                                           ▐
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
def getscanZ(distance_matrix, L, N0, k, statistic, n0, n1, N):
    maxZ = np.arange(N0, N, dtype=np.float64)
    maxZw = np.arange(N0, N, dtype=np.float64)
    maxM = np.arange(N0, N, dtype=np.float64)
    maxS = np.arange(N0, N, dtype=np.float64)
    for i in np.arange((N0+1), (N+1)):
        tests = getZL(distance_matrix[np.ix_(np.arange((i-L), i), np.arange((i-L), i))], k)
        maxZ[((i-N0)-1)] = ((tests["Z"][n0:(n1+1)]).max())
        maxZw[((i-N0)-1)] = ((tests["Zw"][n0:(n1+1)]).max())
        maxM[((i-N0)-1)] = ((tests["M"][n0:(n1+1)]).max())
        maxS[((i-N0)-1)] = ((tests["S"][n0:(n1+1)]).max())
    
    output = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        output["original"] = maxZ
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        output["weighted"] = maxZw
    if gu.anyin({"all", "max", "m"}, statistic):
        output["max_type"] = maxM
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        output["generalized"] = maxS
    return output
    
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
def getZL(distance_matrix, k=1):
    np.seterr(divide="ignore", invalid="ignore")
    L = (distance_matrix.shape)[0]
    A = np.zeros((L, k), dtype=np.int64)
    for i in np.arange(L):
        A[i, :] = ((distance_matrix[i, :L]).argsort())[:k]
    ind, freq = np.unique(A, return_counts=True)
    deg = np.zeros(L)
    deg[ind] = freq
    deg_sumsq = ((deg**2).sum())
    cn = (((deg-k)**2).sum())/L/k
    count = 0
    for i in np.arange(L):
        ids = A[i, :]
        count += ((A[ids, :]==i).sum())
    vn = count/L/k
    ts = np.arange(1, L)
    q = (L-ts-1)/(L-2)
    p = (ts-1)/(L-2)
    
    EX1L = 2*k*(ts)*(ts-1)/(L-1)
    EX2L = 2*k*(L-ts)*(L-ts-1)/(L-1)
    EX = 4*k*ts*(L-ts)/(L-1)
    
    config1 = (2*k*L+2*k*L*vn)
    config2 = (3*k**2*L+deg_sumsq-2*k*L-2*k*L*vn)
    config3 = (4*L**2*k**2+4*k*L+4*k*L*vn-12*k**2*L-4*deg_sumsq)
    
    f11 = 2*(ts)*(ts-1)/L/(L-1)
    f21 = 4*(ts)*(ts-1)*(ts-2)/L/(L-1)/(L-2)
    f31 = (ts)*(ts-1)*(ts-2)*(ts-3)/L/(L-1)/(L-2)/(L-3)
    
    f12 = 2*(L-ts)*(L-ts-1)/L/(L-1)
    f22 = 4*(L-ts)*(L-ts-1)*(L-ts-2)/L/(L-1)/(L-2)
    f32 = (L-ts)*(L-ts-1)*(L-ts-2)*(L-ts-3)/L/(L-1)/(L-2)/(L-3)
    
    h = 4*(ts-1)*(L-ts-1)/((L-2)*(L-3))
    VX = EX*(h*(1+vn-2*k/(L-1))+(1-h)*cn)
    
    var1 = config1*f11+config2*f21+config3*f31-EX1L**2
    var2 = config1*f12+config2*f22+config3*f32-EX2L**2
    v12 = config3*((ts)*(ts-1)*(L-ts)*(L-ts-1))/(L*(L-1)*(L-2)*(L-3))-EX1L*EX2L

    X = np.zeros((L-1))
    X1 = np.zeros((L-1))
    X2 = np.zeros((L-1))
    for i in np.arange((L-1)):
        X2[i] = 2*((A[(i+1):L, :]>i).sum())
        X1[i] = 2*((A[:(i+1), :]<=i).sum())
        X[i] = 2*(((A[:(i+1), :]>i).sum())+((A[(i+1):L, :]<=i).sum()))
    Rw = q*X1+p*X2
    ERw = q*EX1L+p*EX2L
    varRw = q**2*var1+p**2*var2+2*p*q*v12
    Zw = (Rw-ERw)/np.sqrt(varRw)
    
    Zdiff = ((X1-X2)-(EX1L-EX2L))/np.sqrt(var1+var2-2*v12)
    S = Zw**2+Zdiff**2
    M = np.stack((np.abs(Zdiff), Zw), axis=1).max(axis=1)
    Z = (EX-X)/np.sqrt(VX)

    return {"R" : X, "R1" : X1, "R2" : X2, "Rw" : Rw, "Z1" : (X1-EX1L)/np.sqrt(var1), "Z2" : (X2-EX2L)/np.sqrt(var2),
            "Zdiff" : Zdiff, "Z" : Z, "Zw" : Zw, "M" : M, "S" : S}


# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ THRESHOLDS █                                                                                                      ▐
# ▌ Purpose :                                                                                                           ▐
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
def getb(distance_matrix, L, N0, k, statistic, n0, n1, arl, alpha, skew_corr, asymp, dif=1e-10, nIterMax=1e2):
    quantities = gb_quantities(distance_matrix, N0, k)
    psum = quantities["psum"]
    qsum = quantities["qsum"]
    psumk = quantities["psumk"]
    qsumk = quantities["qsumk"]
    psumk1 = quantities["psumk1"]
    qsumk1 = quantities["qsumk1"]
    psumk2 = quantities["psumk2"]
    qsumk2 = quantities["qsumk2"]
    deg_sumsq = quantities["deg_sumsq"]
    deg_sum3 = quantities["deg_sum3_n"]
    aaa1 = quantities["aaa1_n"]
    aaa2 = quantities["aaa2_n"]
    dda = quantities["dda_n"]
    daa = quantities["daa_n"]
    
    output = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        output["original"] = getbZ(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=3, bmax=5, skew_corr=skew_corr)
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        output["weighted"] = getbZw(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=3, bmax=5, skew_corr=skew_corr, asymp=asymp)
    if gu.anyin({"all", "max", "m"}, statistic):
        output["max_type"] = getbM(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=8, bmax=20, skew_corr=skew_corr, asymp=asymp)
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        output["generalized"] = getbS(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=20, bmax=30, asymp=asymp)
    return output
    
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
# obtain graph based quantities
def gb_quantities(distance_matrix, N0, k):
    n = N0
    An = np.zeros((n, (k+2)), dtype=np.int64)
    for i in np.arange(n):
        An[i, :] = ((distance_matrix[i, :n]).argsort())[:(k+2)]
    ind, freq = np.unique(An[:, :k], return_counts=True)
    deg = np.zeros(n)
    deg[ind] = freq
    deg_sumsq = ((deg**2).sum())
    deg_sum3 = ((deg**3).sum())
    count = daa = dda = aaa1 = aaa2 = 0
    for i in np.arange(n):
        ids = An[i, :k]
        count += ((An[ids, :k]==i).sum())
        daa += deg[i]*((An[ids, :k]==i).sum())
        dda += deg[i]*((deg[ids]).sum())
        for j in ids:
            u = An[j, :k]
            aaa1 += ((An[u, :k]==i).sum())
            aaa2 += ((np.isin(ids, u)).sum())
    psum = count/n
    qsum = deg_sumsq/n-k # j,l cannot be the same
    deg_sum3_n = deg_sum3
    aaa1_n = aaa1
    aaa2_n = aaa2
    daa_n = daa
    dda_n = dda

    count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = 0
    for i in np.arange(n):
        ids = An[i, (k-1)]
        count1 += ((An[ids, :k]==i).sum())
        count2 += ((np.delete(An[:, :k], i, axis=0)==ids).sum())
        
        ids1 = An[i, k]
        count3 += ((An[ids1, :k]==i).sum())
        count4 += ((np.delete(An[:, :k], i, axis=0)==ids1).sum())
        count7 += ((An[ids1, k]==i).sum())
        count8 += ((np.delete(An[:, k], i, axis=0)==ids1).sum())
        
        ids2 = An[i, k+1]
        count5 += ((An[ids2, :k]==i).sum())
        count6 += ((np.delete(An[:, :k], i, axis=0)==ids2).sum())
    psumk = count1/n
    qsumk = count2/n
    psumk1 = count3/n
    qsumk1 = count4/n
    psumk2 = count5/n
    qsumk2 = count6/n
    pLk1 = count7/n
    qLk1 = count8/n
    
    return {"psum" : psum, "qsum" : qsum, "psumk1" : psumk1, "qsumk1" : qsumk1,
            "psumk2" : psumk2, "qsumk2" : qsumk2, "pLk1" : pLk1, "qLk1" : qLk1,
            "psumk" : psumk, "qsumk" : qsumk, "deg_sumsq" : deg_sumsq, "deg_sum3_n" : deg_sum3_n,
            "aaa1_n" : aaa1_n, "aaa2_n" : aaa2_n, "daa_n" : daa_n, "dda_n" : dda_n}

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
def getbZ(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=3, bmax=5, skew_corr=False, dif=1e-10, nIterMax=1e2):
    m0 = arl*alpha
    if skew_corr == False:
        pm = T3_lambda(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk)*m0
        while pm < alpha:
            bmin -= 1
            pm = T3_lambda(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk)*m0
        pM = T3_lambda(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk)*m0
        while pM > alpha:
            bmax += 1
            pM = T3_lambda(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk)*m0
        b = (bmin+bmax)/2
        p = T3_lambda(b, L, k, n0, n1, psum, qsum, psumk, qsumk)*m0
        nIter = 1
        while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
            if p < alpha:
                bmax = b
            else:
                bmin = b
            b = (bmin+bmax)/2
            p = T3_lambda(b, L, k, n0, n1, psum, qsum, psumk, qsumk)*m0
            nIter += 1
    else:
        pm = T3_skewed_lambda(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1,aaa2, daa, dda)*m0
        while pm < alpha:
            bmin -=1
            pm = T3_skewed_lambda(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1,aaa2, daa, dda)*m0
        pM = T3_skewed_lambda(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1,aaa2, daa, dda)*m0
        while pM > alpha:
            bmax += 1
            pM = T3_skewed_lambda(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1,aaa2, daa, dda)*m0
        b = (bmin+bmax)/2
        p = T3_skewed_lambda(b, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1,aaa2, daa, dda)*m0
        nIter = 1
        while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
            if p < alpha:
                bmax = b
            else:
                bmin = b
            b = (bmin+bmax)/2
            p = T3_skewed_lambda(b, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1,aaa2, daa, dda)*m0
            nIter += 1
    return b

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
def getbZw(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=3, bmax=5, skew_corr=False, asymp=False, dif=1e-10, nIterMax=1e2):
    m0 = arl*alpha
    if skew_corr == False:
        pm = T3_lambdaZw(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
        while pm < alpha:
            bmin -= 1
            pm = T3_lambdaZw(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
        pM = T3_lambdaZw(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
        while pM > alpha:
            bmax += 1
            pM = T3_lambdaZw(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
        b = (bmin+bmax)/2
        p = T3_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
        nIter = 1
        while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
            if p < alpha:
                bmax = b
            else:
                bmin = b
            b = (bmin+bmax)/2
            p = T3_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
            nIter += 1
    else:
        pm = T3_skewed_lambdaZw(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)*m0
        while pm < alpha:
            bmin -= 1
            pm = T3_skewed_lambdaZw(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)*m0
        pM = T3_skewed_lambdaZw(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)*m0
        while pM > alpha:
            bmax += 1
            pM = T3_skewed_lambdaZw(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)*m0
        b = (bmin+bmax)/2
        p = T3_skewed_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)*m0
        nIter = 1
        while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
            if p < alpha:
                bmax = b
            else:
                bmin = b
            b = (bmin+bmax)/2
            p = T3_skewed_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)*m0
            nIter += 1
    return b

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
def getbM(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=3, bmax=5, skew_corr=False, asymp=False, dif=1e-10, nIterMax=1e2):
    m0 = arl*alpha
    if skew_corr == False:
        pm = T3_lambdaM(bmin, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
        while pm < alpha:
            bmin -= 1
            pm = T3_lambdaM(bmin, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
        pM = T3_lambdaM(bmax, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
        while pM > alpha:
            bmax += 1
            pM = T3_lambdaM(bmax, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
        b = (bmin+bmax)/2
        p = T3_lambdaM(b, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
        nIter = 1
        while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
            if p < alpha:
                bmax = b
            else:
                bmin = b
            b = (bmin+bmax)/2
            p = T3_lambdaM(b, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
            nIter += 1
    else:
        pm = T3_skewed_lambdaM(bmin, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
        while pm < alpha:
            bmin -= 1
            pm = T3_skewed_lambdaM(bmin, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
        pM = T3_skewed_lambdaM(bmax, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
        while pM > alpha:
            bmax += 1
            pM = T3_skewed_lambdaM(bmax, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
        b = (bmin+bmax)/2
        p = T3_skewed_lambdaM(b, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
        nIter = 1
        while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
            if p < alpha:
                bmax = b
            else:
                bmin = b
            b = (bmin+bmax)/2
            p = T3_skewed_lambdaM(b, m0, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
            nIter += 1
    return b
    
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
def getbS(L, k, n0, n1, arl, alpha, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda, bmin=3, bmax=5, asymp=False, dif=1e-10, nIterMax=1e2):
    m0 = arl*alpha
    pm = T3_lambdaS(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
    while pm < alpha:
        bmin -= 1
        pm = T3_lambdaS(bmin, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
    pM = T3_lambdaS(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
    while pM > alpha:
        bmax += 1
        pM = T3_lambdaS(bmax, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
    b = (bmin+bmax)/2
    p = T3_lambdaS(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
    nIter = 1
    while (np.abs(p-alpha) > dif) and (nIter < nIterMax):
        if p < alpha:
            bmax = b
        else:
            bmin = b
        b = (bmin+bmax)/2
        p = T3_lambdaS(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)*m0
        nIter += 1
    return b



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ LAMBDA █                                                                                                          ▐
# ▌ Purpose :                                                                                                           ▐
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
def T3_lambda(b, L, k, n0, n1, psum, qsum, psumk, qsumk):
    np.seterr(divide="ignore", invalid="ignore")
    C3 = C1_Z(np.arange((n0+1), (n1+2)), L, k, psum, qsum)
    C4 = C2_Z(np.arange((n0+1), (n1+2)), L, k, psum, qsum, psumk, qsumk)
    return norm.pdf(b)*b**3*((C3*C4*Nu(np.sqrt(2*C3*b**2))*Nu(np.sqrt(2*C4*b**2))).sum())
    
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
def T3_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp):
    np.seterr(divide="ignore", invalid="ignore")
    if asymp == True:
        C1 = C1_W_asymp(np.arange((n0+1), (n1+2))/L)
        C2 = C2_W_asymp(np.arange((n0+1), (n1+2))/L, k, psum, psumk1)
        C2[C2<0] = 1e-8
    else:
        C1 = C1_W(np.arange((n0+1), (n1+2)), L, k, psum, qsum)
        C2 = C2_W(np.arange((n0+1), (n1+2)), L, k, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2)
        C2[C2<0] = 1e-8
    return norm.pdf(b)*b**3*((C1*C2*Nu(np.sqrt(2*C1*b**2))*Nu(np.sqrt(2*C2*b**2))).sum())

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
def T3_lambdaZd(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp):
    if asymp == True:
        C1 = C1_D_asymp(np.arange((n0+1), (n1+2))/L)
        C2 = C2_D_asymp(np.arange((n0+1), (n1+2))/L, k, qsum, qsumk1)
        C2[C2<0] = 1e-8
    else:
        C1 = C1_D(np.arange((n0+1), (n1+2)), L, k, qsum)
        C2 = C2_D(np.arange((n0+1), (n1+2)), L, k, psum, qsum, qsumk, psumk1, qsumk1, psumk2, qsumk2)
        C2[C2<0] = 1e-8
    nu1 = np.nan_to_num(Nu(np.sqrt(2*C1*b**2)), nan=0, posinf=0, neginf=0)
    nu2 = np.nan_to_num(Nu(np.sqrt(2*C2*b**2)), nan=0, posinf=0, neginf=0)
    return norm.pdf(b)*b**3*((C1*C2*Nu(np.sqrt(2*C1*b**2))*Nu(np.sqrt(2*C2*b**2))).sum())
    
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
def T3_lambdaM(b, D, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp):
    pval_Zd = T3_lambdaZd(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
    pval_Zw = T3_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp)
    return (1-(1-D*2*pval_Zd)*(1-D*pval_Zw))
    
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
def T3_lambdaS(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, asymp):
    def integrandS(t, w):
        if asymp == True:
            C1w = np.array(C1_W_asymp(t/L))
            C2w = np.array(C2_W_asymp(t/L, k, psum, psumk1))
            C1d = np.array(C1_D_asymp(t/L))
            C2d = np.array(C2_D_asymp(t/L, k, qsum, qsumk1))
            C1w[C1w<0] = 0
            C2w[C2w<0] = 0
            C1d[C1d<0] = 0
            C2d[C2d<0] = 0
        else:
            C1w = np.array(C1_W(t, L, k, psum, qsum))
            C2w = np.array(C2_W(t, L, k, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2))
            C1d = np.array(C1_D(t, L, k, qsum))
            C2d = np.array(C2_D(t, L, k, psum, qsum, qsumk, psumk1, qsumk1, psumk2, qsumk2))
            C1w[C1w<0] = 0
            C2w[C2w<0] = 0
            C1d[C1d<0] = 0
            C2d[C2d<0] = 0
        nu1 = Nu(np.sqrt(2*b*(C1d*np.cos(w)**2+C1w*np.sin(w)**2)))
        nu2 = Nu(np.sqrt(2*b*(C2d*np.cos(w)**2+C2w*np.sin(w)**2)))
        return (4*(C1d*np.cos(w)**2+C1w*np.sin(w)**2)*(C2d*np.cos(w)**2+C2w*np.sin(w)**2)*b**2*nu1*nu2)/(2*np.pi)

    result = integrate.dblquad(integrandS, a=0, b=2*np.pi, gfun=(n0+1), hfun=(n1+1))[0]
    return chi2.pdf(b, 2)*result

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
def T3_skewed_lambda(b, L, k, n0, n1, psum, qsum, psumk, qsumk, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda):
    C3 = C1_Z(np.arange((n0+1), (n1+2)), L, k, psum, qsum)
    C4 = C2_Z(np.arange((n0+1), (n1+2)), L, k, psum, qsum, psumk, qsumk)
    n = L
    ts = np.arange(1, n)
    vn = psum/k
    EX = 4*k*ts*(n-ts)/(n-1)
    EX2 = 4*k*(1+vn)*2*ts*(n-ts)/(n-1)+4*(3*k**2*n+deg_sumsq-2*k*n*(1+vn))*ts*(n-ts)/n/(n-1)+(4*k**2*n**2-4*(3*k**2*n+deg_sumsq)+4*k*n*(1+vn))*4*ts*(ts-1)*(n-ts)*(n-ts-1)/(n*(n-1)*(n-2)*(n-3))
    EX3 = EX3_f(n, ts, k, vn, deg_sumsq, deg_sum3, daa, dda, aaa1, aaa2)
    VX = EX2-EX**2
    gamma = -(EX3-3*EX*VX-EX**3)/(VX**(3/2))
    theta = np.zeros((n-1))
    pos = (np.where((1+2*gamma*b)>0))[0]
    theta[pos] = (np.sqrt((1+2*gamma*b)[pos])-1)/gamma[pos]
    S = (1+gamma*theta)**(-1/2)*np.exp((b-theta)**2/2+gamma*theta**3/6)
    nn = n-(pos.size)
    if nn > .75*n:
        return 0
    if nn >= 2*(n0+1):
        neg = np.where((1+2*gamma*b)<=0)[0]
        dif = neg[1:(nn-1)]-neg[:(nn-2)]
        id1 = dif.argmax()
        if nn < n:
            id2 = id1+np.int64(np.ceil(.02*n))
            id3 = id2+np.int64(np.ceil(.02*n))
            inc = (S[id3]-S[id2])/(id3-id2)
            S[id2::-1] = S[(id2+1)]-inc*np.arange(1, (id2+2))
            S[(n-id2-2):(n-1)] = S[id2::-1]
        else:
            ymax = S[np.int64(np.ceil(n/2-1))]
            ind = np.int64(id1+.05*n)
            a = (ymax-S[ind])/(ind-n/2)**2
            S[:(ind+1)] = ymax-a*(np.arange(1, (ind+2))-n/2)**2
            S[(n-ind-2):(n-1)] = S[ind::-1]
        neg2 = np.where(S<0)[0]
        S[neg2] = 0
    return norm.pdf(b)*b**3*((S[np.arange((n-n0-2), (n-n1-3), -1)]*C3*C4*Nu(np.sqrt(2*C3*b**2))*Nu(np.sqrt(2*C4*b**2))).sum())
    
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
def T3_skewed_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda):
    C1 = C1_W(np.arange((n0+1), (n1+2)), L, k, psum, qsum)
    C2 = C2_W(np.arange((n0+1), (n1+2)), L, k, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2)
    C2[C2<0] = 1e-8
    n = L
    ts = np.arange(1, n)
    EX = ERw(L, k, ts)
    EX3 = EX3_newf(L, k, ts, psum, deg_sumsq, deg_sum3, daa, dda, aaa1, aaa2)["ERw3"]
    VX = VarRw(L, k, ts, psum, deg_sumsq)
    gamma = (EX3-3*EX*VX-EX**3)/(VX**(3/2))
    theta = np.zeros(n-1)
    pos = np.where((1+2*gamma*b)>0)[0]
    theta[pos] = (np.sqrt((1+2*gamma*b)[pos])-1)/gamma[pos]
    S = (1+gamma*theta)**(-1/2)*np.exp((b-theta)**2/2+gamma*theta**3/6)
    nn = n-(pos.size)
    if nn > .75*n:
        return 0
    if nn >= (n0+(n-n0+1)):
        neg = np.where((1+2*gamma*b)<=0)[0]
        dif = neg[1:(nn-1)]-neg[:(nn-2)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*n))
        id3 = id2+np.int64(np.ceil(.09*n))
        inc = (S[id3]-S[id2])/np.ceil(.09*n)
        S[id2::-1] = S[(id2+1)]-inc*np.arange(1, (id2+2))
        S[np.int64(n/2):n] = S[np.int64(n/2-1*(n%2==0))::-1]
        neg2 = np.where(S<0)[0]
        S[neg2] = 0
    return norm.pdf(b)*b**3*((S[np.arange((n-n0-2), (n-n1-3), -1)]*C1*C2*Nu(np.sqrt(2*C1*b**2))*Nu(np.sqrt(2*C2*b**2))).sum())
    
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
def T3_skewed_lambdaZd(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda):
    C1 = C1_D(np.arange((n0+1), (n1+2)), L, k, qsum)
    C2 = C2_D(np.arange((n0+1), (n1+2)), L, k, psum, qsum, qsumk, psumk1, qsumk1, psumk2, qsumk2)
    C1[C1<0] = 0
    C2[C2<0] = 0
    n = L
    ts = np.arange(1, n)
    EX = ERd(L, k, ts)
    EX3 = EX3_newf(L, k, ts, psum, deg_sumsq, deg_sum3, daa, dda, aaa1, aaa2)["ERd3"]
    VX = VarRd(L, k, ts, psum, deg_sumsq)
    gamma = (EX3-3*EX*VX-EX**3)/(VX**(3/2))
    theta = np.zeros((n-1))
    pos = np.where((1+2*gamma*b)>0)[0]
    theta[pos] = (np.sqrt((1+2*gamma*b)[pos])-1)/gamma[pos]
    S = (1+gamma*theta)**(-1/2)*np.exp((b-theta)**2/2+gamma*theta**3/6)
    S[np.int64(n/2-1)] = S[np.int64(n/2-2)]
    nn = n-(pos.size)
    nn_l = np.ceil(n/2)-((np.where((1+2*gamma[:np.int64(np.ceil(n/2))]*b)>0)[0]).size)
    nn_r = np.ceil(n/2)-((np.where((1+2*gamma[np.int64(np.ceil(n/2)):n]*b)>0)[0]).size)
    if nn > .75*n:
        return 0
    if nn_r >= (n-n1+1):
        neg = np.where((1+2*gamma[np.int64(np.ceil(n/2)):(n-1)]*b)<-0)[0]
        id1 = neg[0]+np.int64(np.ceil(n/2)-1)
        id2 = id1-np.int64(np.ceil(.06*n))
        id3 = id2-np.int64(np.ceil(.06*n))
        inc = (S[id3]-S[id2])/(id3-id2)
        S[id2:(n-1)] = S[id2]+inc*(np.arange((id2+1), n)-(id2+1))+inc**2*(np.arange((id2+1), n)-(id2+1))+inc**3*(np.arange((id2+1), n)-(id2+1))
        if (n0+1) <= .15*200:
            S[S<0] = 0
    return norm.pdf(b)*b**3*((S[np.arange((n-n0-2), (n-n1-3), -1)]*C1*C2*Nu(np.sqrt(2*C1*b**2))*Nu(np.sqrt(2*C2*b**2))).sum())
    
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
def T3_skewed_lambdaM(b, D, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda):
    pval_Zd = T3_skewed_lambdaZd(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
    pval_Zw = T3_skewed_lambdaZw(b, L, k, n0, n1, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2, deg_sumsq, deg_sum3, aaa1, aaa2, daa, dda)
    return (1-(1-D*2*pval_Zd)*(1-D*pval_Zw))



    
# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ CRITICAL VALUES █                                                                                                 ▐
# ▌ Purpose :                                                                                                           ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : cast_inputs_to_npfloat64                                                                              ║
# ║ Purpose     : acts as a decorator for the functions that compute the critical values; this decorator casts all      ║
# ║               inputs to type np.float64 if scalar, or np.asarray(, dtype=np.float64) otherwise                      ║
# ║ Arguments   :                                                                                                       ║
# ║    - func (function) : function whose input will be cast to np.float64                                              ║
# ║ Returns     : wrapper, a function that casts all input arguments to type np.float64                                 ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def cast_inputs_to_npfloat64(func):
    @wraps(func) # from the functools module, modifies wrapper so that it looks like func to introspection tools
    # wrapper casts all input arguments to np.float64
    def wrapper(*args, **kwargs):
        def cast(inputarg):
            return np.float64(inputarg) if np.isscalar(inputarg) else np.asarray(inputarg, dtype=np.float64)
        cast_args = tuple(cast(arg) for arg in args)
        cast_kwargs = {k : cast(v) for k, v in kwargs.items()}
        return func(*cast_args, **cast_kwargs)
    return wrapper
    
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
@cast_inputs_to_npfloat64
def C1_Z(x, L, k, psum, qsum):
    np.seterr(divide="ignore", invalid="ignore")
    return ((16*(k+2*psum-psum)*(2*L-2*x-1)*(x**2-x))/(L**3-6*L**2+11*L-6)-(16*k**2*x**2*(L-x))/ \
            (L-1)**2+(16*k**2*x*(L-x)**2)/(L-1)**2+(4*x*(3*k**2+k+2*qsum-qsum)*(3*L**2-10*L*x-3*L+8*x**2+2*x+ \
            2))/(L**3-6*L**2+11*L-6)+(16*L*k**2*x*(3*L*x-L**2+L-2*x**2-2*x+1))/((L-1)*(L-2)*(L-3)))/(4*((k**2* \
            (((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k+qsum-k**2))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L* \
            psum+2*k**2))/(k*(L-1)*(L-2)*(L-3)))**2*x**2*(L-x)**2)/(L-1)**2)**(1/2))-(((16*k**2*(((((-x+1)* \
            (4*L-4*x-4))/((L-2)*(L-3))+1)*(k-k**2+qsum))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L*psum+2*k**2))/ \
            (k*(L-1)*(L-2)*(L-3)))**2*x**2*(L-x))/(L-1)**2-(16*k**2*(((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)* \
            (k-k**2+qsum))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L*psum+2*k**2))/(k*(L-1)*(L-2)*(L-3)))**2*x* \
            (L-x)**2)/(L-1)**2+(64*k*(((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k-k**2+qsum))/k+((-x+1)*(4*L-4*x-4)* \
            (k+psum-L*k-L*psum+2*k**2))/(k*(L-1)*(L-2)*(L-3)))*x**2*(L-2*x)*(L-x)**2*(psum-qsum-L*psum+L*qsum-L*k**2+ \
            3*k**2))/((L-1)**2*(L**3-6*L**2+11*L-6)))*(L*((4*x*(L-x))/(L*(L-1))-(16*(L-x)*(L-x-1)*(x**2-x))/(L* \
            (L-1)*(L-2)*(L-3)))*(3*k**2+k+2*qsum-qsum)+(16*(k+2*psum-psum)*(x**2-x)*(L**2-2*L*x-L+x**2+x))/(L**3- \
            6*L**2+11*L-6)-(16*k**2*x**2*(L-x)**2)/(L-1)**2+(16*L*k**2*(L-x)*(L-x-1)*(x**2-x))/((L-1)*(L-2)*(L-3))))/ \
            (128*((k**2*(((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k+qsum-k**2))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L* \
            k-L*psum+2*k**2))/(k*(L-1)*(L-2)*(L-3)))**2*x**2*(L-x)**2)/(L-1)**2)**(3/2))

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
@cast_inputs_to_npfloat64
def C2_Z(x, L, k, psum, qsum, psumk, qsumk):
    np.seterr(divide="ignore", invalid="ignore")
    return ((((4*x*(L-x))/(L*(L-1))-(16*(L-x)*(L-x-1)*(x**2-x))/(L*(L-1)*(L-2)*(L-3)))*(2*k+4*qsum-2*qsum- \
            7*L*k-8*L*qsum+7*L*qsum-2*L*qsumk-9*L*k**2+3*L**2*k+2*L**2*qsum-3*L**2*qsum+2*L**2*qsumk+6*k**2+3* \
            L**2*k**2))/(L**2-3*L+2)+L*(3*k**2+k+2*qsum-qsum)*((4*(L**2-2*L*x-2*L+x**2+2*x))/(L*(L-1)*(L-2))- \
            (4*x*(L-x))/(L**2*(L-1))+(4*x*(L-x))/(L*(L-1)**2*(L-2))+(16*(-x+1)*(L**2-3*L*x-L+2*x**2+x))/(L*(L- \
            1)*(L-2)*(L-3))-(16*(4*L**2-12*L+6)*(L-x)*(L-x-1)*(x**2-x))/(L**2*(L-1)**2*(L-2)**2*(L-3)**2))+(16* \
            k**2*x**2*(L-x))/(L-1)**2-(16*k**2*x*(L-x)**2)/(L-1)**2-(16*(k+2*psum-psum)*(-x+1)*(L**6-3*L**5*x-7* \
            L**5+2*L**4*x**2+23*L**4*x+17*L**4-20*L**3*x**2-55*L**3*x-17*L**3+4*L**2*x**3+50*L**2*x**2+47*L**2*x+ \
            6*L**2-12*L*x**3-36*L*x**2-12*L*x+6*x**3+6*x**2))/(L*(11*L-6*L**2+L**3-6)**2)+(16*(x**2-x)*(L**2-2*L* \
            x-L+x**2+x)*(2*k+4*psum-2*psum-7*L*k-10*L*psum+7*L*psum-2*L*psumk+3*L**2*k+4*L**2*psum-3*L**2*psum+2* \
            L**2*psumk))/(L*(L**2-3*L+2)*(L**3-6*L**2+11*L-6))-(16*L*k**2*(-x+1)*(L**2-3*L*x-L+2*x**2+x))/((L-1)* \
            (L-2)*(L-3))+(16*k**2*(4*L**2-12*L+6)*(L-x)*(L-x-1)*(x**2-x))/((L-1)**2*(L-2)**2*(L-3)**2))/(4*((k**2* \
            (((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k+qsum-k**2))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L*psum+2* \
            k**2))/(k*(L-1)*(L-2)*(L-3)))**2*x**2*(L-x)**2)/(L-1)**2)**(1/2))+(((16*k**2*(((((-x+1)*(4*L-4*x-4))/ \
            ((L-2)*(L-3))+1)*(k-k**2+qsum))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L*psum+2*k**2))/(k*(L-1)*(L-2)*(L- \
            3)))**2*x**2*(L-x))/(L-1)**2-(16*k**2*(((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k-k**2+qsum))/k+((-x+ \
            1)*(4*L-4*x-4)*(k+psum-L*k-L*psum+2*k**2))/(k*(L-1)*(L-2)*(L-3)))**2*x*(L-x)**2)/(L-1)**2+(64*k* \
            (((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k-k**2+qsum))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L*psum+2* \
            k**2))/(k*(L-1)*(L-2)*(L-3)))*x**2*(L-2*x)*(L-x)**2*(psum-qsum-L*psum+L*qsum-L*k**2+3*k**2))/((L-1)**2* \
            (L**3-6*L**2+11*L-6)))*(L*((4*x*(L-x))/(L*(L-1))-(16*(L-x)*(L-x-1)*(x**2-x))/(L*(L-1)*(L-2)*(L-3)))* \
            (3*k**2+k+2*qsum-qsum)+(16*(k+2*psum-psum)*(x**2-x)*(L**2-2*L*x-L+x**2+x))/(L**3-6*L**2+11*L-6)- \
            (16*k**2*x**2*(L-x)**2)/(L-1)**2+(16*L*k**2*(L-x)*(L-x-1)*(x**2-x))/((L-1)*(L-2)*(L-3))))/(128* \
            ((k**2*(((((-x+1)*(4*L-4*x-4))/((L-2)*(L-3))+1)*(k+qsum-k**2))/k+((-x+1)*(4*L-4*x-4)*(k+psum-L*k-L* \
            psum+2*k**2))/(k*(L-1)*(L-2)*(L-3)))**2*x**2*(L-x)**2)/(L-1)**2)**(3/2))

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
@cast_inputs_to_npfloat64
def C1_W_asymp(x):
    np.seterr(divide="ignore", invalid="ignore")
    return 1/(2*x*(1-x))

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
@cast_inputs_to_npfloat64
def C2_W_asymp(x, k, psum, psumk1):
    np.seterr(divide="ignore", invalid="ignore")
    return (x**2-x+1)/(x*(1-x))-(2*k*psumk1)/(k+psum)

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
@cast_inputs_to_npfloat64
def C1_W(x, L, k, psum, qsum):
    np.seterr(divide="ignore", invalid="ignore")
    result = None
    if k == 1:
        result = -(x**2*(x-1)**2*(2*x**2-2*L*x+L)*(L**2-2*L*x-L+x**2+x)**2*(3*k+2*psum+qsum-4*L*k-3* \
                 L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2*(2*psum-4*L+qsum-3*L*psum-L*qsum-L*k**2+ \
                 L**2*psum+L**2+3*k**2+3))/(2*(L-1)**5*(L-2)**6*(L-3)**3*((x**2*(x-1)**2*(L**2-2*L*x-L+x**2+x)**2* \
                 (3*k+2*psum+qsum-4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+ \
                 2)**4))**(3/2))
    elif k == 5:
        result = -(x**2*(x-1)**2*(2*x**2-2*L*x+L)*(L**2-2*L*x-L+x**2+x)**2*(3*k+2*psum+qsum-4*L*k-3*L*psum-L* \
                 qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2*(2*psum-20*L+qsum-3*L*psum-L*qsum-L*k**2+L**2*psum+5*L**2+ \
                 3*k**2+15))/(2*(L-1)**5*(L-2)**6*(L-3)**3*((x**2*(x-1)**2*(L**2-2*L*x-L+x**2+x)**2*(3*k+2*psum+qsum- \
                 4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(3/2))
    else:
        result = -(x**2*(x-1)**2*(2*x**2-2*L*x+L)*(L**2-2*L*x-L+x**2+x)**2*(3*k+2*psum+qsum-4*L*k-3*L*psum-L* \
                 qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**3)/(2*(L-1)**5*(L-2)**6*(L-3)**3*((x**2*(x-1)**2* \
                 (L**2-2*L*x-L+x**2+x)**2*(3*k+2*psum+qsum-4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2* \
                 psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(3/2))
    return result

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
@cast_inputs_to_npfloat64
def C2_W(x, L, k, psum, qsum, psumk, qsumk, psumk1, qsumk1, psumk2, qsumk2):
    np.seterr(divide="ignore", invalid="ignore")
    num = None
    den = None
    part2 = None
    if k == 1:
        num = -((x-L+1)*(36*L-72*x+72*k**2*x**2+72*k**2*x**3+24*L*psum+12*L*qsum+66*L*x-48*psum*x-24*qsum* \
              x+36*L*k**2-74*L**2*psum+85*L**3*psum-45*L**4*psum+11*L**5*psum-L**6*psum-31*L**2*qsum+27*L**3* \
              qsum-9*L**4*qsum+L**5*qsum-312*L*x**2+88*L**2*x-36*L*x**3-169*L**3*x+96*L**4*x-23*L**5*x+2*L**6*x- \
              72*k**2*x+88*psum*x**2+8*psum*x**3-48*psumk1*x**2+48*psumk1*x**3-16*psumk2*x**2+16*psumk2*x**3+56* \
              qsum*x**2-8*qsum*x**3-16*qsumk1*x**2+16*qsumk1*x**3-4*qsumk2*x**2+4*qsumk2*x**3-105*L**2+112*L**3- \
              54*L**4+12*L**5-L**6-69*L**2*k**2+43*L**3*k**2-11*L**4*k**2+L**5*k**2+144*x**2+313*L**2*x**2+33*L**2* \
              x**3-153*L**3*x**2-10*L**3*x**3+35*L**4*x**2+L**4*x**3-3*L**5*x**2-210*L*k**2*x**2+52*L**2*k**2*x-66*L* \
              k**2*x**3-64*L**3*k**2*x+20*L**4*k**2*x-2*L**5*k**2*x+245*L**2*psum*x**2+33*L**2*psum*x**3-133*L**3* \
              psum*x**2-10*L**3*psum*x**3+33*L**4*psum*x**2+L**4*psum*x**3-3*L**5*psum*x**2+30*L**2*psumk1*x**2+70* \
              L**2*psumk1*x**3+14*L**2*psumk2*x**2-50*L**3*psumk1*x**2+14*L**2*psumk2*x**3-20*L**3*psumk1*x**3-12* \
              L**3*psumk2*x**2+18*L**4*psumk1*x**2-2*L**3*psumk2*x**3+2*L**4*psumk1*x**3+2*L**4*psumk2*x**2-2*L**5* \
              psumk1*x**2+68*L**2*qsum*x**2-20*L**3*qsum*x**2+2*L**4*qsum*x**2+14*L**2*qsumk1*x**2+14*L**2*qsumk1*x**3+ \
              4*L**2*qsumk2*x**2-12*L**3*qsumk1*x**2+2*L**2*qsumk2*x**3-2*L**3*qsumk1*x**3-2*L**3*qsumk2*x**2+2*L**4* \
              qsumk1*x**2+60*L*psum*x+48*L*psumk1*x+16*L*psumk2*x+6*L*qsum*x+16*L*qsumk1*x+4*L*qsumk2*x+152*L**2*k**2* \
              x**2+20*L**2*k**2*x**3-42*L**3*k**2*x**2-2*L**3*k**2*x**3+4*L**4*k**2*x**2+66*L*k**2*x-218*L*psum*x**2+ \
              40*L**2*psum*x-38*L*psum*x**3-117*L**3*psum*x+78*L**4*psum*x-21*L**5*psum*x+2*L**6*psum*x+52*L*psumk1* \
              x**2-100*L**2*psumk1*x-100*L*psumk1*x**3+12*L*psumk2*x**2-28*L**2*psumk2*x+70*L**3*psumk1*x-28*L*psumk2* \
              x**3+14*L**3*psumk2*x-20*L**4*psumk1*x-2*L**4*psumk2*x+2*L**5*psumk1*x-94*L*qsum*x**2+48*L**2*qsum*x+2* \
              L*qsum*x**3-52*L**3*qsum*x+18*L**4*qsum*x-2*L**5*qsum*x+12*L*qsumk1*x**2-28*L**2*qsumk1*x-28*L*qsumk1* \
              x**3+2*L*qsumk2*x**2-6*L**2*qsumk2*x+14*L**3*qsumk1*x-6*L*qsumk2*x**3+2*L**3*qsumk2*x-2*L**4*qsumk1* \
              x))/((L-2)**3*(L-4)*(L**2-4*L+3)**2*((x**2*(x-1)**2*(x-L-2*L*x+L**2+x**2)**2*(3*k+2*psum+qsum-4*L*k-3* \
              L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(1/2))-(x**2*(x-1)**2* \
              (L**2-2*L*x-L+x**2+x)**2*(2*L**2*x-L**2-6*L*x**2+2*L*x+L+4*x**3-2*x)*(3*k+2*psum+qsum-4*L*k-3*L*psum- \
              L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2*(2*psum-4*L+qsum-3*L*psum-L*qsum-L*k**2+L**2*psum+L**2+3*k**2+ \
              3))/(2*(L-3)**3*(L**2-3*L+2)**6*((x**2*(x-1)**2*(x-L-2*L*x+L**2+x**2)**2*(3*k+2*psum+qsum-4*L*k-3*L* \
              psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(3/2))
        den = 1
        part2 = 0
    elif k == 5:
        num = ((x-L+1)*(75600*L-151200*x+30240*k**2*x**2+30240*k**2*x**3+10080*L*psum+5040*L*qsum+196740*L*x-20160* \
              psum*x-10080*qsum*x+15120*L*k**2-34956*L**2*psum+48188*L**3*psum-34305*L**4*psum+13857*L**5*psum-3282* \
              L**6*psum+450*L**7*psum-33*L**8*psum+L**9*psum-14958*L**2*qsum+16615*L**3*qsum-8845*L**4*qsum+2506*L**5* \
              qsum-388*L**6*qsum+31*L**7*qsum-L**8*qsum-771480*L*x**2+123450*L**2*x-75600*L*x**3-418250*L**3*x+347605* \
              L**4*x-145120*L**5*x+34290*L**6*x-4640*L**7*x+335*L**8*x-10*L**9*x-30240*k**2*x+36960*psum*x**2+3360* \
              psum*x**3-47040*psumk1*x**2+47040*psumk1*x**3-14400*psumk2*x**2+14400*psumk2*x**3+23520*qsum*x**2-3360* \
              qsum*x**3-6720*qsumk1*x**2+6720*qsumk1*x**3-1800*qsumk2*x**2+1800*qsumk2*x**3-249570*L**2+324015*L**3- \
              215750*L**4+81815*L**5-18350*L**6+2405*L**7-170*L**8+5*L**9-34794*L**2*k**2+30009*L**3*k**2-13141*L**4* \
              k**2+3222*L**5*k**2-448*L**6*k**2+33*L**7*k**2-L**8*k**2+302400*x**2+925350*L**2*x**2+98370*L**2*x**3- \
              609605*L**3*x**2-51675*L**3*x**3+233495*L**4*x**2+14030*L**4*x**3-53130*L**5*x**2-2080*L**5*x**3+7060* \
              L**6*x**2+160*L**6*x**3-505*L**7*x**2-5*L**7*x**3+15*L**8*x**2-99828*L*k**2*x**2+9570*L**2*k**2*x-39348* \
              L*k**2*x**3-33736*L**3*k**2*x+19838*L**4*k**2*x-5548*L**5*k**2*x+830*L**6*k**2*x-64*L**7*k**2*x+2*L**8* \
              k**2*x+140076*L**2*psum*x**2+20176*L**2*psum*x**3-100385*L**3*psum*x**2-10387*L**3*psum*x**3+41021*L**4* \
              psum*x**2+2808*L**4*psum*x**3-9792*L**5*psum*x**2-416*L**5*psum*x**3+1348*L**6*psum*x**2+32*L**6*psum* \
              x**3-99*L**7*psum*x**2-L**7*psum*x**3+3*L**8*psum*x**2-8800*L**2*psumk1*x**2+137880*L**2*psumk1*x**3- \
              600*L**2*psumk2*x**2-63210*L**3*psumk1*x**2+38880*L**2*psumk2*x**3-74670*L**3*psumk1*x**3-19470*L**3* \
              psumk2*x**2+52530*L**4*psumk1*x**2-19410*L**3*psumk2*x**3+22140*L**4*psumk1*x**3+14400*L**4*psumk2*x**2- \
              18540*L**5*psumk1*x**2+5010*L**4*psumk2*x**3-3600*L**5*psumk1*x**3-4380*L**5*psumk2*x**2+3300*L**6* \
              psumk1*x**2-630*L**5*psumk2*x**3+300*L**6*psumk1*x**3+600*L**6*psumk2*x**2-290*L**7*psumk1*x**2+30*L**6* \
              psumk2*x**3-10*L**7*psumk1*x**3-30*L**7*psumk2*x**2+10*L**8*psumk1*x**2+44994*L**2*qsum*x**2-502*L**2* \
              qsum*x**3-21536*L**3*qsum*x**2+52*L**3*qsum*x**3+5678*L**4*qsum*x**2-2*L**4*qsum*x**3-834*L**5*qsum*x**2+ \
              64*L**6*qsum*x**2-2*L**7*qsum*x**2+280*L**2*qsumk1*x**2+17200*L**2*qsumk1*x**3+270*L**2*qsumk2*x**2-8990* \
              L**3*qsumk1*x**2+4290*L**2*qsumk2*x**3-8210*L**3*qsumk1*x**3-2400*L**3*qsumk2*x**2+6220*L**4*qsumk1*x**2- \
              1890*L**3*qsumk2*x**3+1990*L**4*qsumk1*x**3+1500*L**4*qsumk2*x**2-1760*L**5*qsumk1*x**2+390*L**4*qsumk2* \
              x**3-230*L**5*qsumk1*x**3-360*L**5*qsumk2*x**2+220*L**6*qsumk1*x**2-30*L**5*qsumk2*x**3+10*L**6*qsumk1* \
              x**3+30*L**6*qsumk2*x**2-10*L**7*qsumk1*x**2+32952*L*psum*x+47040*L*psumk1*x+14400*L*psumk2*x+6396*L* \
              qsum*x+6720*L*qsumk1*x+1800*L*qsumk2*x+99366*L**2*k**2*x**2+20670*L**2*k**2*x**3-46952*L**3*k**2*x**2- \
              5612*L**3*k**2*x**3+12056*L**4*k**2*x**2+832*L**4*k**2*x**3-1728*L**5*k**2*x**2-64*L**5*k**2*x**3+130* \
              L**6*k**2*x**2+2*L**6*k**2*x**3-4*L**7*k**2*x**2+39348*L*k**2*x-105772*L*psum*x**2+6036*L**2*psum*x- \
              17252*L*psum*x**3-54214*L**3*psum*x+52495*L**4*psum*x-24070*L**5*psum*x+6084*L**6*psum*x-866*L**7* \
              psum*x+65*L**8*psum*x-2*L**9*psum*x+82040*L*psumk1*x**2-129080*L**2*psumk1*x-129080*L*psumk1*x**3+23880* \
              L*psumk2*x**2-38280*L**2*psumk2*x+137880*L**3*psumk1*x-38280*L*psumk2*x**3+38880*L**3*psumk2*x-74670* \
              L**4*psumk1*x-19410*L**4*psumk2*x+22140*L**5*psumk1*x+5010*L**5*psumk2*x-3600*L**6*psumk1*x-630*L**6* \
              psumk2*x+300*L**7*psumk1*x+30*L**7*psumk2*x-10*L**8*psumk1*x-48524*L*qsum*x**2+18654*L**2*qsum*x+2132* \
              L*qsum*x**3-29436*L**3*qsum*x+17026*L**4*qsum*x-4954*L**5*qsum*x+774*L**6*qsum*x-62*L**7*qsum*x+2*L**8* \
              qsum*x+10760*L*qsumk1*x**2-17480*L**2*qsumk1*x-17480*L*qsumk1*x**3+2760*L*qsumk2*x**2-4560*L**2*qsumk2* \
              x+17200*L**3*qsumk1*x-4560*L*qsumk2*x**3+4290*L**3*qsumk2*x-8210*L**4*qsumk1*x-1890*L**4*qsumk2*x+1990* \
              L**5*qsumk1*x+390*L**5*qsumk2*x-230*L**6*qsumk1*x-30*L**6*qsumk2*x+10*L**7*qsumk1*x))
        den = ((L-2)**3*(L**2-4*L+3)**2*(L**4-26*L**3+251*L**2-1066*L+1680)*((x**2*(x-1)**2*(x-L-2*L*x+L**2+x**2)**2* \
              (3*k+2*psum+qsum-4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+ \
              2)**4))**(1/2))
        part2 = (x**2*(x-1)**2*(L**2-2*L*x-L+x**2+x)**2*(2*L**2*x-L**2-6*L*x**2+2*L*x+L+4*x**3-2*x)*(3*k+2*psum+qsum- \
                4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2*(2*psum-20*L+qsum-3*L*psum-L*qsum-L*k**2+ \
                L**2*psum+5*L**2+3*k**2+15))/(2*(L-3)**3*(L**2-3*L+2)**6*((x**2*(x-1)**2*(x-L-2*L*x+L**2+x**2)**2*(3*k+ \
                2*psum+qsum-4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(3/2))
    else:
        num = -((x-L+1)*(18*k*x-18*k**2*x**3-9*L*k-6*L*psum-3*L*qsum-18*k**2*x**2+12*psum*x+6*qsum*x-9*L*k**2+24*L**2* \
              k-22*L**3*k+8*L**4*k-L**5*k+17*L**2*psum-17*L**3*psum+7*L**4*psum-L**5*psum+7*L**2*qsum-5*L**3*qsum+L**4 \
              *qsum-36*k*x**2+18*k**2*x+2*psum*x**2-26*psum*x**3-12*psumk*x**2+12*psumk*x**3+4*qsum*x**2-16*qsum*x**3- \
              6*qsumk*x**2+6*qsumk*x**3+15*L**2*k**2-7*L**3*k**2+L**4*k**2+48*L*k**2*x**2-61*L**2*k*x**2-16*L**2*k**2* \
              x+12*L*k**2*x**3-6*L**2*k*x**3+23*L**3*k*x**2+12*L**3*k**2*x+L**3*k*x**3-3*L**4*k*x**2-2*L**4*k**2*x-67* \
              L**2*psum*x**2-20*L**2*psum*x**3+33*L**3*psum*x**2+3*L**3*psum*x**3-5*L**4*psum*x**2+10*L**2*psumk*x**2+ \
              12*L**2*psumk*x**3-10*L**3*psumk*x**2-2*L**3*psumk*x**3+2*L**4*psumk*x**2-26*L**2*qsum*x**2-4*L**2*qsum* \
              x**3+6*L**3*qsum*x**2+6*L**2*qsumk*x**2+2*L**2*qsumk*x**3-2*L**3*qsumk*x**2-12*L*k*x-36*L*psum*x+12*L* \
              psumk*x-18*L*qsum*x+6*L*qsumk*x-26*L**2*k**2*x**2-2*L**2*k**2*x**3+4*L**3*k**2*x**2+69*L*k*x**2-12*L* \
              k**2*x-25*L**2*k*x+9*L*k*x**3+36*L**3*k*x-15*L**4*k*x+2*L**5*k*x+41*L*psum*x**2+19*L**2*psum*x+41*L* \
              psum*x**3+12*L**3*psum*x-11*L**4*psum*x+2*L**5*psum*x+10*L*psumk*x**2-22*L**2*psumk*x-22*L*psumk*x**3+ \
              12*L**3*psumk*x-2*L**4*psumk*x+20*L*qsum*x**2+6*L**2*qsum*x+18*L*qsum*x**3+6*L**3*qsum*x-2*L**4*qsum*x+ \
              2*L*qsumk*x**2-8*L**2*qsumk*x-8*L*qsumk*x**3+2*L**3*qsumk*x))
        den = ((L-2)**3*(L**2-4*L+3)**2*((x**2*(x-1)**2*(x-L-2*L*x+L**2+x**2)**2*(3*k+2*psum+qsum-4*L*k-3*L*psum-L* \
              qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(1/2))
        part2 = (x**2*(x-1)**2*(L**2-2*L*x-L+x**2+x)**2*(2*L**2*x-L**2-6*L*x**2+2*L*x+L+4*x**3-2*x)*(3*k+2*psum+ \
                qsum-4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3*k**2)**3)/(2*(L-3)**3*(L**2-3*L+2)**6*((x**2* \
                (x-1)**2*(x-L-2*L*x+L**2+x**2)**2*(3*k+2*psum+qsum-4*L*k-3*L*psum-L*qsum-L*k**2+L**2*k+L**2*psum+3* \
                k**2)**2)/((L-3)**2*(L**2-3*L+2)**4))**(3/2))
    return ((num/den)-part2)

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
@cast_inputs_to_npfloat64
def C1_D_asymp(x):
    return 1/(x*(1-x))

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
@cast_inputs_to_npfloat64
def C2_D_asymp(x, k, qsum, qsumk1):
    return (10*qsum-4*k*qsumk1-(6*k**2-10*k))/(2*(qsum-k**2+k))-1/(2*x*(1-x))

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
@cast_inputs_to_npfloat64
def C1_D(x, L, k, qsum):
    result = None
    if k == 1:
        result = (L*x**2*(L-x)**2*(-k**2+k+qsum)**2*(-k**2+qsum+1))/(2*(L-1)**3*((x**2*(L-x)**2*(-k**2+ \
                 k+qsum)**2)/(L-1)**2)**(3/2))
    elif k == 5:
        result = (L*x**2*(L-x)**2*(-k**2+k+qsum)**2*(-k**2+qsum+5))/(2*(L-1)**3*((x**2*(L-x)**2*(-k**2+k+ \
                 qsum)**2)/(L-1)**2)**(3/2))
    else:
        result = (L*x**2*(L-x)**2*(-k**2+k+qsum)**3)/(2*(L-1)**3*((x**2*(L-x)**2*(-k**2+k+qsum)**2)/(L-1)**2)**(3/2))
    return result

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
@cast_inputs_to_npfloat64
def C2_D(x, L, k, psum, qsum, qsumk, psumk1, qsumk1, psumk2, qsumk2):
    result = None
    if k == 1:
        result = -(x**2*(L-x)**2*(-k**2+k+qsum)**2*(48*k**2*x**2+144*L*x-12*L**2*qsum+19*L**3*qsum-8*L**4*qsum+ \
                 L**5*qsum+204*L*x**2-204*L**2*x+82*L**3*x-10*L**4*x-144*qsum*x**2+32*qsumk1*x**2+8*qsumk2*x**2- \
                 12*L**2+19*L**3-8*L**4+L**5+12*L**2*k**2-19*L**3*k**2+8*L**4*k**2-L**5*k**2-144*x**2-82*L**2*x**2+ \
                 10*L**3*x**2-100*L*k**2*x**2+100*L**2*k**2*x-46*L**3*k**2*x+6*L**4*k**2*x-82*L**2*qsum*x**2+10*L**3* \
                 qsum*x**2+28*L**2*qsumk1*x**2+4*L**2*qsumk2*x**2-4*L**3*qsumk1*x**2+144*L*qsum*x-32*L*qsumk1*x-8*L* \
                 qsumk2*x+46*L**2*k**2*x**2-6*L**3*k**2*x**2-48*L*k**2*x+204*L*qsum*x**2-204*L**2*qsum*x+82*L**3*qsum* \
                 x-10*L**4*qsum*x-56*L*qsumk1*x**2+56*L**2*qsumk1*x-12*L*qsumk2*x**2+12*L**2*qsumk2*x-28*L**3*qsumk1*x- \
                 4*L**3*qsumk2*x+4*L**4*qsumk1*x))/(2*(L-1)**4*(L**3-9*L**2+26*L-24)*((x**2*(L-x)**2*(-k**2+k+ \
                 qsum)**2)/(L-1)**2)**(3/2))
    elif k == 5:
        result = -(x**2*(L-x)**2*(-k**2+k+qsum)**2*(6720*k**2*x**2+100800*L*x-1680*L**2*qsum+2746*L**3*qsum-1317*L**4* \
                 qsum+277*L**5*qsum-27*L**6*qsum+L**7*qsum+147960*L*x**2-147960*L**2*x+68360*L**3*x-14110*L**4*x+1360* \
                 L**5*x-50*L**6*x-20160*qsum*x**2+4480*qsumk1*x**2+1200*qsumk2*x**2-8400*L**2+13730*L**3-6585*L**4+1385* \
                 L**5-135*L**6+5*L**7+1680*L**2*k**2-2746*L**3*k**2+1317*L**4*k**2-277*L**5*k**2+27*L**6*k**2-L**7* \
                 k**2-100800*x**2-68360*L**2*x**2+14110*L**3*x**2-1360*L**4*x**2+50*L**5*x**2-14344*L*k**2*x**2+14344* \
                 L**2*k**2*x-7400*L**3*k**2*x+1610*L**4*k**2*x-160*L**5*k**2*x+6*L**6*k**2*x-13672*L**2*qsum*x**2+2822* \
                 L**3*qsum*x**2-272*L**4*qsum*x**2+10*L**5*qsum*x**2+8080*L**2*qsumk1*x**2+1980*L**2*qsumk2*x**2-2780* \
                 L**3*qsumk1*x**2-600*L**3*qsumk2*x**2+400*L**4*qsumk1*x**2+60*L**4*qsumk2*x**2-20*L**5*qsumk1*x**2+ \
                 20160*L*qsum*x-4480*L*qsumk1*x-1200*L*qsumk2*x+7400*L**2*k**2*x**2-1610*L**3*k**2*x**2+160*L**4*k**2* \
                 x**2-6*L**5*k**2*x**2-6720*L*k**2*x+29592*L*qsum*x**2-29592*L**2*qsum*x+13672*L**3*qsum*x-2822*L**4* \
                 qsum*x+272*L**5*qsum*x-10*L**6*qsum*x-10160*L*qsumk1*x**2+10160*L**2*qsumk1*x-2640*L*qsumk2*x**2+2640* \
                 L**2*qsumk2*x-8080*L**3*qsumk1*x-1980*L**3*qsumk2*x+2780*L**4*qsumk1*x+600*L**4*qsumk2*x-400*L**5* \
                 qsumk1*x-60*L**5*qsumk2*x+20*L**6*qsumk1*x))/(2*(L-1)**4*((x**2*(L-x)**2*(-k**2+k+qsum)**2)/(L- \
                 1)**2)**(3/2)*(L**5-28*L**4+303*L**3-1568*L**2+3812*L-3360))
    else:
        result = -(x**2*(L-x)**2*(-k**2+k+qsum)**2*(4*k**2*x**2-L**2*k+L**3*k-L**2*qsum+L**3*qsum-12*k*x**2-4* \
                 qsumk*x**2+L**2*k**2-L**3*k**2-6*L*k**2*x**2+6*L**2*k**2*x+12*L*k*x+4*L*qsumk*x+10*L*k*x**2- \
                 4*L*k**2*x-10*L**2*k*x+2*L*qsum*x**2-2*L**2*qsum*x+4*L*qsumk*x**2-4*L**2*qsumk*x))/(2*(L-1)**4* \
                 (L-2)*((x**2*(L-x)**2*(-k**2+k+qsum)**2)/(L-1)**2)**(3/2))
    return result



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ EXPECTATIONS, VARIANCES, AND AUTOCORRELATIONS █                                                                   ▐
# ▌ Purpose :                                                                                                           ▐
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
@cast_inputs_to_npfloat64
def Nu(x):
    y = x/2
    return (1/y)*(norm.cdf(y)-.5)/(y*norm.cdf(y)+norm.pdf(y))

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
@cast_inputs_to_npfloat64
def ERw(n, k, t):
    q = (n-t-1)/(n-2)
    p = (t-1)/(n-2)
    ER1 = 2*k*t*(t-1)/(n-1)
    ER2 = 2*k*(n-t)*(n-t-1)/(n-1)
    return q*ER1+p*ER2

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
@cast_inputs_to_npfloat64
def VarRw(n, k, t, psum, deg_sumsq):
    vn = psum/k
    config1 = (2*k*n+2*k*n*vn)
    config2 = (3*k**2*n+deg_sumsq-2*k*n-2*k*n*vn)
    config3 = (4*n**2*k**2+4*k*n+4*k*n*vn-12*k**2*n-4*deg_sumsq)
    f11 = 2*(t)*(t-1)/n/(n-1)
    f21 = 4*(t)*(t-1)*(t-2)/n/(n-1)/(n-2)
    f31 = (t)*(t-1)*(t-2)*(t-3)/n/(n-1)/(n-2)/(n-3)
    V1 = config1*f11+config2*f21+config3*f31-(2*k*t*(t-1)/(n-1))**2
    f12 = 2*(n-t)*(n-t-1)/n/(n-1)
    f22 = 4*(n-t)*(n-t-1)*(n-t-2)/n/(n-1)/(n-2)
    f32 = (n-t)*(n-t-1)*(n-t-2)*(n-t-3)/n/(n-1)/(n-2)/(n-3)
    V2 = config1*f12+config2*f22+config3*f32-(2*k*(n-t)*(n-t-1)/(n-1))**2
    P3 = (t*(t-1)*(n-t)*(n-t-1))/(n*(n-1)*(n-2)*(n-3))
    V12 = config3*P3-(2*k*t*(t-1)/(n-1))*(2*k*(n-t)*(n-t-1)/(n-1))
    q = (n-t-1)/(n-2)
    p = (t-1)/(n-2)
    return q**2*V1+p**2*V2+2*p*q*V12
    
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
@cast_inputs_to_npfloat64
def ERd(n, k, t):
    return 2*k*t*(t-1)/(n-1)-2*k*(n-t)*(n-t-1)/(n-1)

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
@cast_inputs_to_npfloat64
def VarRd(n, k, t, psum, deg_sumsq):
    vn = psum/k
    config1 = (2*k*n+2*k*n*vn)
    config2 = (3*k**2*n+deg_sumsq-2*k*n-2*k*n*vn)
    config3 = (4*n**2*k**2+4*k*n+4*k*n*vn-12*k**2*n-4*deg_sumsq)
    f11 = 2*(t)*(t-1)/n/(n-1)
    f21 = 4*(t)*(t-1)*(t-2)/n/(n-1)/(n-2)
    f31 = (t)*(t-1)*(t-2)*(t-3)/n/(n-1)/(n-2)/(n-3)
    V1 = config1*f11+config2*f21+config3*f31-(2*k*t*(t-1)/(n-1))**2
    f12 = 2*(n-t)*(n-t-1)/n/(n-1)
    f22 = 4*(n-t)*(n-t-1)*(n-t-2)/n/(n-1)/(n-2)
    f32 = (n-t)*(n-t-1)*(n-t-2)*(n-t-3)/n/(n-1)/(n-2)/(n-3)
    V2 = config1*f12+config2*f22+config3*f32-(2*k*(n-t)*(n-t-1)/(n-1))**2
    P3 = (t*(t-1)*(n-t)*(n-t-1))/(n*(n-1)*(n-2)*(n-3))
    V12 = config3*P3-(2*k*t*(t-1)/(n-1))*(2*k*(n-t)*(n-t-1)/(n-1))
    q = (n-t-1)/(n-2)
    p = (t-1)/(n-2)
    return V1+V2-2*V12

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
@cast_inputs_to_npfloat64
def EX3_f(n, t, k, vn, deg_sumsq, deg_sum3, daa, dda, aaa1, aaa2):
    x1 = 2*k*n+6*k*n*vn
    x2 = 3*k**2*n+deg_sumsq+2*k**2*n*vn+2*daa-x1
    x3 = 4*k**2*n**2*(1+vn)-4*(3*k**2*n+deg_sumsq+2*k**2*n*vn+2*daa)+2*(2*k*n+6*k*n*vn)
    x4 = 4*k**3*n+3*k*deg_sumsq+deg_sum3-3*(3*k**2*n+deg_sumsq+2*k**2*n*vn+2*daa)+2*(2*k*n+6*k*n*vn)
    x5 = 4*k**3*n+2*k*deg_sumsq+2*dda-2*(3*k**2*n+deg_sumsq+2*k**2*n*vn+2*daa)+x1-2*aaa1-6*aaa2
    x6 = 2*aaa1+6*aaa2
    x7 = 2*k*n*(3*k**2*n+deg_sumsq)-4*k**2*n**2*(1+vn)-4*(4*k**3*n+2*k*deg_sumsq+2*dda-(3*k**2*n+deg_sumsq+ \
         2*k**2*n*vn+2*daa))-2*(4*k**3*n+3*k*deg_sumsq+deg_sum3-(3*k**2*n+deg_sumsq+2*k**2*n*vn+2*daa))+4*((3* \
         k**2*n+deg_sumsq+2*k**2*n*vn+2*daa)-(2*k*n+6*k*n*vn))+2*x6
    x8 = 8*k**3*n**3-4*x1-24*x2-6*x3-8*x4-24*x5-8*x6-12*x7
    p1 = 2*t*(n-t)/n/(n-1)
    p2 = p1/2
    p3 = 4*t*(t-1)*(n-t)*(n-t-1)/(n*(n-1)*(n-2)*(n-3))
    p4 = t*(n-t)*((n-t-1)*(n-t-2)+(t-1)*(t-2))/(n*(n-1)*(n-2)*(n-3))
    p5 = p7 = p3/2
    p8 = 8*t*(t-1)*(t-2)*(n-t)*(n-t-1)*(n-t-2)/(n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5))
    return 4*x1*p1+24*x2*p2+6*x3*p3+8*x4*p4+24*x5*p5+12*x7*p7+x8*p8

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
@cast_inputs_to_npfloat64
def EX3_newf(n, k, t, psum, deg_sumsq, deg_sum3, daa, dda, aaa1, aaa2):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    x1 = 2*k*n+6*k*n*psum/k
    x2 = 3*k**2*n+deg_sumsq+2*k*n*psum+2*daa-x1
    x3 = 4*k**2*n**2*(1+psum/k)-4*(3*k**2*n+deg_sumsq+2*k**2*n*psum/k+2*daa)+2*(2*k*n+6*k*n*psum/k)
    x4 = 4*k**3*n+3*k*deg_sumsq+deg_sum3-3*(3*k**2*n+deg_sumsq+2*k**2*n*psum/k+2*daa)+2*(2*k*n+6*k*n*psum/k)
    x5 = 4*k**3*n+2*k*deg_sumsq+2*dda-2*(3*k**2*n+deg_sumsq+2*k**2*n*psum/k+2*daa)+x1-2*aaa1-6*aaa2
    x6 = 2*aaa1+6*aaa2
    x7 = 2*k*n*(3*k**2*n+deg_sumsq)-4*k**2*n**2*(1+psum/k)-4*(4*k**3*n+2*k*deg_sumsq+2*dda-(3*k**2*n+deg_sumsq+ \
         2*k**2*n*psum/k+2*daa))-2*(4*k**3*n+3*k*deg_sumsq+deg_sum3-(3*k**2*n+deg_sumsq+2*k**2*n*psum/k+2*daa))+ \
        4*((3*k**2*n+deg_sumsq+2*k**2*n*psum/k+2*daa)-(2*k*n+6*k*n*psum/k))+2*x6
    x8 = 8*k**3*n**3-4*x1-24*x2-6*x3-8*x4-24*x5-8*x6-12*x7
    p1 = t*(t-1)/n/(n-1)
    p2 = t*(t-1)*(t-2)/n/(n-1)/(n-2)
    p3 = t*(t-1)*(t-2)*(t-3)/(n*(n-1)*(n-2)*(n-3))
    p4 = p5 = t*(t-1)*(t-2)*(t-3)/(n*(n-1)*(n-2)*(n-3))
    p6 = p2
    p7 = t*(t-1)*(t-2)*(t-3)*(t-4)/(n*(n-1)*(n-2)*(n-3)*(n-4))
    p8 = t*(t-1)*(t-2)*(t-3)*(t-4)*(t-5)/(n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5))
    q1 = (n-t)*(n-t-1)/n/(n-1)
    q2 = (n-t)*(n-t-1)*(n-t-2)/n/(n-1)/(n-2)
    q3 = (n-t)*(n-t-1)*(n-t-2)*(n-t-3)/(n*(n-1)*(n-2)*(n-3))
    q4 = q5 = q3
    q6 = q2
    q7 = (n-t)*(n-t-1)*(n-t-2)*(n-t-3)*(n-t-4)/(n*(n-1)*(n-2)*(n-3)*(n-4))
    q8 = (n-t)*(n-t-1)*(n-t-2)*(n-t-3)*(n-t-4)*(n-t-5)/(n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5))
    A11 = 4*x1*p1+24*x2*p2+6*x3*p3+8*x4*p4+24*x5*p5+8*x6*p6+12*x7*p7+x8*p8
    A12 = 2*x3*t*(t-1)*(n-t)*(n-t-1)/(n*(n-1)*(n-2)*(n-3))+4*x7*t*(t-1)*(t-2)*(n-t)*(n-t-1)/(n*(n-1)*(n-2)*(n-3)* \
          (n-4))+x8*t*(t-1)*(t-2)*(t-3)*(n-t)*(n-t-1)/(n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5))
    A21 = 2*x3*t*(t-1)*(n-t)*(n-t-1)/(n*(n-1)*(n-2)*(n-3))+4*x7*(n-t)*(n-t-1)*(n-t-2)*(t)*(t-1)/(n*(n-1)*(n-2)*(n-3)* \
          (n-4))+x8*(n-t)*(n-t-1)*(n-t-2)*(n-t-3)*(t)*(t-1)/(n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5))
    A22 = 4*x1*q1+24*x2*q2+6*x3*q3+8*x4*q4+24*x5*q5+8*x6*q6+12*x7*q7+x8*q8
    q = (n-t-1)/(n-2)
    p = (t-1)/(n-2)
    ERw3 = q**3*A11+3*q**2*p*A12+3*q*p**2*A21+p**3*A22
    ERd3 = A11-3*A12+3*A21-A22
    return {"A11" : A11, "A12" : A12, "A21" : A21, "A22" : A22, "ERw3" : ERw3, "ERd3" : ERd3}


    
# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ PRINTING AND PLOTTING █                                                                                           ▐
# ▌ Purpose : prints and plots the results of gstream for quick and easy reading and visualization                      ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FORMATFLOAT() METADATA                                                                                              ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : formatfloat                                                                                           ║
# ║ Purpose     : formats a given float value to the specified number of decimals places as an f-string                 ║
# ║                  - if given value is None, return f"NONE"                                                           ║
# ║ Arguments   :                                                                                                       ║
# ║    - fval (float or NoneType) :                                                                                     ║
# ║    - decimal_places (integer) :                                                                                     ║
# ║ Returns     : an f-string of the formatted, passed in float value                                                   ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def formatfloat(fval, decimal_places=4):
    return f"{fval:.{decimal_places}f}" if fval is not None else f"NONE"

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ CON_PRINT() METADATA                                                                                                ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : con_print                                                                                             ║
# ║ Purpose     : summarizes and prints the results of gstream in a succinct and readable format                        ║
# ║ Arguments   :                                                                                                       ║
# ║    - results (boolean)    : a dictionary containing the results of gstream                                          ║
# ║    - printStops (boolean) : print the estimated stopping indexes (n-N0)? True if yes, False if no                   ║
# ║    - printScans (boolean) : print the scan statistics? True if yes, False if no                                     ║
# ║    - printbs (boolean)    : print the thresholds? True if yes, False if no                                          ║
# ║    - decimal_places (integer) : the number of decimal places to use when printing the scan statistics and           ║
# ║                                 thresholds                                                                          ║
# ║ Returns     : nothing, this function is just for printing to the console                                            ║
# ║ Author      : written by Alex Wold                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def con_print(results, printStops=True, printScans=True, printbs=True, decimal_places=4):
    # print original test results
    if ("original" in results["stops"]) or ("original" in results["scanZ"]) or ("original" in results["b"]):
        print(f"\n\nORIGINAL TEST RESULTS")
        print(f"---------------------")
        # print stopping indices
        if ("original" in results["stops"]) and (printStops==True):
            print(f"Stopping indices (n-N0):\n{results['stops']['original']}\n")

        # print scan statistics
        if ("original" in results["scanZ"]) and (printScans==True):
            print(f"Scan statistics (ZL|y):\n{np.array2string(results['scanZ']['original'], precision=decimal_places, suppress_small=True)}\n")

        # print threshold
        if ("original" in results["b"]) and (printbs==True):
            print(f"Threshold (bZ): {formatfloat(results['b']['original'], decimal_places)}\n")
        
    # print weighted test results
    if ("weighted" in results["stops"]) or ("weighted" in results["scanZ"]) or ("weighted" in results["b"]):
        print(f"\n\nWEIGHTED TEST RESULTS")
        print(f"---------------------")
        # print stopping indices
        if ("weighted" in results["stops"]) and (printStops==True):
            print(f"Stopping indices (n-N0):\n{results['stops']['weighted']}\n")

        # print scan statistics
        if ("weighted" in results["scanZ"]) and (printScans==True):
            print(f"Scan statistics (WL|y):\n{np.array2string(results['scanZ']['weighted'], precision=decimal_places, suppress_small=True)}\n")

        # print threshold
        if ("weighted" in results["b"]) and (printbs==True):
            print(f"Threshold (bW): {formatfloat(results['b']['weighted'], decimal_places)}\n")

    # print max type test results
    if ("max_type" in results["stops"]) or ("max_type" in results["scanZ"]) or ("max_type" in results["b"]):
        print(f"\n\nMAX-TYPE TEST RESULTS")
        print(f"---------------------")
        # print stopping indices
        if ("max_type" in results["stops"]) and (printStops==True):
            print(f"Stopping indices (n-N0):\n{results['stops']['max_type']}\n")

        # print scan statistics
        if ("max_type" in results["scanZ"]) and (printScans==True):
            print(f"Scan statistics (ML|y):\n{np.array2string(results['scanZ']['max_type'], precision=decimal_places, suppress_small=True)}\n")

        # print threshold
        if ("max_type" in results["b"]) and (printbs==True):
            print(f"Threshold (bM): {formatfloat(results['b']['max_type'], decimal_places)}\n")

    # print generalized test results
    if ("generalized" in results["stops"]) or ("generalized" in results["scanZ"]) or ("generalized" in results["b"]):
        print(f"\n\nGENERALIZED TEST RESULTS")
        print(f"------------------------")
        # print stopping indices
        if ("generalized" in results["stops"]) and (printStops==True):
            print(f"Stopping indices (n-N0):\n{results['stops']['generalized']}\n")

        # print scan statistics
        if ("generalized" in results["scanZ"]) and (printScans==True):
            print(f"Scan statistics (SL|y):\n{np.array2string(results['scanZ']['generalized'], precision=decimal_places, suppress_small=True)}\n")

        # print threshold
        if ("generalized" in results["b"]) and (printbs==True):
            print(f"Threshold (bS): {formatfloat(results['b']['generalized'], decimal_places)}\n")
    