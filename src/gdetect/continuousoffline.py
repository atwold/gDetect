# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █ ██████ continuousoffline ██████                                                                                     █
# █ Purpose : provides the two key functions for detecting change-points and change-intervals in the continuous setting █
# █              - gchangepoint: used for detecting single change-points                                                █
# █              - gchangeinterval: used for detecting change-intervals                                                 █
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
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.io as pio
pio.renderers.default = "notebook"



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ CORRECTION FACTORS AND AUTOCORRELATION FUNCTIONS █                                                                ▐
# ▌ Purpose : defines finite-sample versions of the tail probability correction factor nu(x) and the autocorrelation    ▐
# ▌           functions h0 (rho_one) and hw (rho_one_Rw); these are used in both the single change-point setting and    ▐
# ▌           the change-interval setting                                                                               ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ NU() METADATA                                                                                                       ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : Nu                                                                                                    ║
# ║ Purpose     : provides a practical (finite-sample) approximation for calculating the correction factor related to   ║
# ║               the tail probabilities of the maximum of the scan statistics time series                              ║
# ║                  - this refines p-value approximation (improving accuracy for extreme values of the scan statistic) ║
# ║                  - helps approximate the expected number of threshold (b) exceedances                               ║
# ║ Arguments   :                                                                                                       ║
# ║    - x (float) : the normalized boundary height. more specifically, x comprises the distance (in standard           ║
# ║                  deviations) that a scan statistic must exceed under the null hypothesis to be considered           ║
# ║                  significant                                                                                        ║
# ║                     - this is a positive scaling of the threshold level (b, the maximum test statistic of the       ║
# ║                       observed time series)                                                                         ║
# ║                     - it is the standardized expected overshoot of the test statistic near the boundary             ║
# ║                     - this is often passed in as x=b*sqrt(2*hstar(t/N)/N)                                           ║
# ║                     - N is the number of observations in the time series (number of nodes in the graph)             ║
# ║                     - b is some threshold                                                                           ║
# ║                     - hstar is the autocorrelation function that captures the local variability of the scan         ║
# ║                       statistic time series at position t/N                                                         ║
# ║ Returns     : a float. namely, the positive scalar correction factor that adjusts the tail probability              ║
# ║               approximation of the maximum scan statistic time series when given the scaled boundary height         ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def Nu(x):
    np.seterr(divide="ignore", invalid="ignore", over="ignore") # ignore floating-point errors
    y = x/2 # (1/y) = 2/x
    # this practical approximation is defined by Siegmund et al (2007)
    return (1/y)*(norm.cdf(y)-.5)/(y*norm.cdf(y)+norm.pdf(y))

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ RHO_ONE() METADATA                                                                                                  ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : rho_one                                                                                               ║
# ║ Purpose     :                                                                                                       ║
# ║ Arguments   :                                                                                                       ║
# ║    - N (<type>)       :                                                                                             ║
# ║    - s (<type>)       :                                                                                             ║
# ║    - sumE (<type>)    :                                                                                             ║
# ║    - sumEisq (<type>) :                                                                                             ║
# ║ Returns     : a float. namely, the                                                                                  ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# finite sample version of h0, the autocorrelation between the scan statistics time series
def rho_one(N, s, sumE, sumEisq):
    np.seterr(divide="ignore", invalid="ignore", over="ignore") # ignore floating-point errors
    f1 = 4*(N-1)*(2*s*(N-s)-N)
    f2 = ((N+1)*(N-2*s)**2-2*N*(N-1))
    f3 = 4*((N-2*s)**2-N)
    f4 = 4*N*(s-1)*(N-1)*(N-s-1)
    f5 = N*(N-1)*((N-2*s)**2-(N-2))
    f6 = 4*((N-2)*(N-2*s)**2-2*s*(N-s)+N)
    return N*(N-1)*(f1*sumE+f2*sumEisq-f3*sumE**2)/(2*s*(N-s)*(f4*sumE+f5*sumEisq-f6*sumE**2))

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
def rho_one_Rw(N, t):
    np.seterr(divide="ignore", invalid="ignore", over="ignore") # ignore floating-point errors
    return -((2*t**2-2*N*t+N)*(N**2-3*N+2)**4)/(2*t*(N-1)**3*(N-2)**4*(t-1)*(N**2-2*N*t-N+t**2+t))



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ CONTINUOUS SINGLE CHANGE-POINT DETECTION █                                                                        ▐
# ▌ Purpose : defines functions for detecting single change-points in the continuous setting                            ▐
# ▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ GCHANGEPOINT() METADATA                                                                                             ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : gchangepoint                                                                                          ║
# ║ Purpose     : user-facing access point for calculating single changepoints in the continuous setting (no repeated   ║
# ║               observations)                                                                                         ║
# ║ Arguments   :                                                                                                       ║
# ║    - E (ndarray, shape: (N, N))    : symmetric N by N adjacency matrix                                              ║
# ║                                         - N is the number of observations (nodes)                                   ║
# ║                                         - edges between two nodes are denoted with an entry of 1, 0 otherwise       ║
# ║    - statistic (set, optional)     : a set of string values specifying which scan statistics to compute             ║
# ║                                         - Default: {"all"}                                                          ║
# ║                                         - can be any subset of "all", "weighted", "max", or "generalized"           ║
# ║                                         - can also be any subset of the abbreviations "wei", "gen"                  ║
# ║                                         - can also be any subset of the one-letter abbreviations "w", "m", or "g"   ║
# ║    - n0 (integer, optional)        : lower boundary (of the index) for a changepoint                                ║
# ║                                         - earliest index in the sequence allowable for a changepoint                ║
# ║                                         - specifies where to start searching for a changepoint                      ║
# ║                                         - Default: None                                                             ║
# ║                                         - if None, n0 is set to the integer ceiling of (.05*N)-1                    ║
# ║                                         - n0 must be greater than or equal to 1, if not, n0 is set to 1             ║
# ║                                           that is, when searching for a changepoint, there must always be at least  ║
# ║                                           one observation in the first sample when the scan statistics are computed ║
# ║                                           this means that the lower boundary of the search cannot be less than      ║
# ║                                           index 1, as the specified index is not included in the first sample       ║
# ║                                         - n0 must be less than or equal to n1, if not, n0 is set to 1 and n1 is set ║
# ║                                           to (N-3)                                                                  ║
# ║    - n1 (integer, optional)        : upper boundary (of the index) for the changepoint                              ║
# ║                                         - latest index in the sequence allowable for a changepoint                  ║
# ║                                         - specifies where to stop searching for a changepoint                       ║
# ║                                         - Default: None                                                             ║
# ║                                         - if None, n1 is set to the integer floor of (.95*N)-1                      ║
# ║                                         - n1 must be less than or equal to (N-3), if not, n1 is set to (N-3)        ║
# ║                                           that is, when searching for a changepoint, there must always be at least  ║
# ║                                           one observation in the second sample when the scan statistics are         ║
# ║                                           computed                                                                  ║
# ║                                           this means that the upper boundary of the search cannot be greater than   ║
# ║                                           index (N-3), as the specified index is not included in the second sample  ║
# ║                                         - n1 must be greater than or equal to n0, if not, n1 is set to (N-3) and n0 ║
# ║                                           is set to 1                                                               ║
# ║    - pval_asym (boolean, optional) : should the asymptotic p-values be computed and returned?                       ║
# ║                                         - Default: True                                                             ║
# ║                                         - if True, compute and append the asymptotic p-values to the result         ║
# ║                                           dictionary with key "pval_asym"                                           ║
# ║                                         - if False, the asymptotic p-values are not computed and not returned       ║
# ║    - skew_corr (boolean, optional) : should skew correction be applied when computing the asymptoic p-values?       ║
# ║                                         - Default: True                                                             ║
# ║                                         - if True, apply skew correction when computing the asymptotic p-values     ║
# ║                                           note, argument pval_asym must be set to True                              ║
# ║                                           skew corrected p-values are appended to the "pval_asym" dictionary with   ║
# ║                                           key "skew"                                                                ║
# ║                                         - if False, skew correction is not applied and only the non-skew p-values   ║
# ║                                           are appended to the "pval_asym" dictionary under the key "no_skew"        ║                 
# ║                                         - non-skew corrected p-values are always returned as part of the            ║
# ║                                           "pval_asym" dictionary                                                    ║
# ║    - pval_perm (integer, optional) : the number of iterations to run when computing the p-values under permutation  ║
# ║ Returns     : a dictionary containing the specified scan statistics, and if requested, the asymptotic and or        ║
# ║               permutation p-values                                                                                  ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def gchangepoint(E, statistic={"all"}, n0=None, n1=None, pval_asym=True, skew_corr=True, pval_perm=0):
    N = (E.shape)[0]
    
    # define default values for n0 and n1
    if n0 is None: n0 = (.05*N)-1
    if n1 is None: n1 = (.95*N)-1
    if n0 < 1: n0 = 1
    if n1 > (N-3): n1 = N-3
    if n0 > n1 or n1 < n0:
        n0 = 1
        n1 = N-3
    n0 = np.int64(np.ceil(n0))
    n1 = np.int64(np.floor(n1))

    # edgelist is an ndarray of shape (edgenum, 2) that contains all of the undirected edges represented in the adjacency matrix E
    # that is, it is a matrix representing the entire set of eij
    # the name is a misnomer, it isn't a python list, it is just a matrix with 2 columns
    # the first columns of this matrix is i, the first observation (node)
    # the second column of this matrix is j, the second observation (node) which is connected to the first observation (node) by an edge
    # since this is an undirected graph, [i, j] = [j, i] and only the first ([i, j]) will be a row in edgelist 
    # that is, eij = eji is an edge in the graph if [i, j] is a row in edgelist
    # that is, we convert the passed in adjacency matrix (E) to a matrix 
    # edgelist is the E argument from gseg1() in the R package gSeg
    # every edge pair
    edgelist = gu.edgematrix_to_edgelist(E)

    # ebynode: for each node, the observations that node is connected to (edges listed by node)
    # this is a python list of python lists <- make sure it isn't a python list of ndarrays
    ebynode = gu.edgematrix_to_edgebynode(E)
    r1 = {}
    r1["scanZ"] = changepoint(N, ebynode, statistic, n0, n1)

    # compute asymptotic p-values
    if pval_asym == True:
        r1["pval_asym"] = pval1(N, edgelist, ebynode, r1["scanZ"], statistic, n0, n1, skew_corr)

    # compute permutation p-values
    pval_perm = round(pval_perm)
    if pval_perm > 0:
        r1["pval_perm"] = permpval1(N, ebynode, r1["scanZ"], statistic, n0, n1, pval_perm)
    
    return r1

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : changepoint                                                                                           ║
# ║ Purpose     : calculates the scan statistics for the standardized original, weighted, max-type, and generalized     ║
# ║               statistic for the single changepoint alternative                                                      ║
# ║ Arguments   :                                                                                                       ║
# ║    - N (<type>) :    total number of nodes                                                                          ║
# ║    - ebynode (<type>) :                                                                                             ║
# ║    - statistic (<type>) :                                                                                           ║
# ║    - n0 (integer, optional):                                                                                        ║
# ║    - n1 (integer, optional):                                                                                        ║
# ║ Returns     : a dictionary whose key-value pairs are the user-specified scan statistics                             ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# ebynode[i]: the array of nodes that are connected to i by an edge
def changepoint(N, ebynode, statistic={"all"}, n0=None, n1=None):
    # ignores calculation errors
    np.seterr(divide="ignore", invalid="ignore", over="ignore")

    # calculate specific graph-based attributes
    # these don't depend on the samples or alternative
    # attribute 1: calculated the degree of each node
    # denoted as the absolute value of G_{i}: |G_{i}|
    node_deg = np.zeros(N)
    for i in np.arange(N):
        node_deg[i] = (ebynode[i]).size

    # attribute 2: the summation of the degrees squared
    # denoted as sum_{i} |G_{i}|^{2}
    sumEisq = ((node_deg**2).sum())

    # attribute 3: count the number of edges in the graph
    # denoted as |G|
    edgenum = ((node_deg).sum())/2

    # calculate specific graph-based statistics
    g = np.ones(N) 
    R = np.zeros(N) # the number of between sample edges
    R1 = np.zeros(N) # the number of within-sample 1 edges
    R2 = np.zeros(N) # the number of within-sample 2 edges

    # calculate R, R1, and R2 for every time point 1 through N (indexed by 0 through N-1)
    for i in np.arange((N-1)):
        g[i] = 0
        links = ebynode[i]
        if i == 0:
            if (links.size) > 0:
                R[i] = ((np.tile(g[i], (links.size)) != g[links]).sum())
            else:
                R[i] = 0
            R1[i] = 0
            R2[i] = edgenum-(links.size)
        else:
            if (links.size) > 0:
                add = ((np.tile(g[i], (links.size)) != g[links]).sum())
                subtract = (links.size)-add
                R[i] = R[i-1]+add-subtract
                R1[i] = R1[i-1]+subtract
            else:
                R[i] = R[i-1]
                R1[i] = R1[i-1]
        R2[i] = edgenum-R[i]-R1[i]

    # calculate the mean and standard deviation for each time point 1 through N
    # tt is a sequence from 1 through N (represents each time point, not index)
    tt = np.arange(1, (N+1))
    temp = np.arange(n0, (n1+1))
    scanZ = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        mu_t = edgenum*2*tt*(N-tt)/(N*(N-1))
        p1_tt = 2*tt*(N-tt)/(N*(N-1))
        p2_tt = tt*(N-tt)*(N-2)/(N*(N-1)*(N-2))
        p3_tt = 4*tt*(N-tt)*(tt-1)*(N-tt-1)/(N*(N-1)*(N-2)*(N-3))
        A_tt = (p1_tt-2*p2_tt+p3_tt)*edgenum+(p2_tt-p3_tt)*sumEisq+p3_tt*edgenum**2
        # standardize the R statistic
        # we reject for unusually small values of R, so if R is smaller than expected (mu_t), then -R-mu_t/sqrt(var(R)) becomes large and positive
        # we reject for large values of Z, so we put a negative in front of R which motivates (mu_t-R) instead of (R-mu_t)
        Z = (mu_t-R)/np.sqrt(A_tt-mu_t**2)
        Z[(N-1)] = 0
        # estimate tauhat, the index of the changepoint, to be the time point where the maximum original scan statistic occurs
        tauhat = temp[(Z[n0:(n1+1)]).argmax()]
        scanZ["original"] = {"tauhat" : tauhat, "Zmax" : Z[tauhat], "Z" : Z, "R" : R}
        
    if gu.anyin({"all", "weighted", "wei", "w", "max", "m", "generalized", "gen", "g"}, statistic):
        # the weighted linear combination of R1 and R2
        Rw = ((N-tt-1)*R1+(tt-1)*R2)/(N-2)
        # calculate the mean and standard deviation for Rw at each time point
        mu_Rw = edgenum*((N-tt-1)*tt*(tt-1)+(tt-1)*(N-tt)*(N-tt-1))/(N*(N-1)*(N-2))
        mu_R1 = edgenum*tt*(tt-1)/(N*(N-1))
        mu_R2 = edgenum*(N-tt)*(N-tt-1)/(N*(N-1))
        v11 = mu_R1*(1-mu_R1)+2*(0.5*sumEisq-edgenum)*(tt*(tt-1)*(tt-2))/(N*(N-1)*(N-2))+(edgenum*(edgenum-1)-2* \
              (0.5*sumEisq-edgenum))*(tt*(tt-1)*(tt-2)*(tt-3))/(N*(N-1)*(N-2)*(N-3))
        v22 = mu_R2*(1-mu_R2)+2*(0.5*sumEisq-edgenum)*((N-tt)*(N-tt-1)*(N-tt-2))/(N*(N-1)*(N-2))+(edgenum*(edgenum-1)- \
              2*(0.5*sumEisq-edgenum))*((N-tt)*(N-tt-1)*(N-tt-2)*(N-tt-3))/(N*(N-1)*(N-2)*(N-3))
        v12 = (edgenum*(edgenum-1)-2*(0.5*sumEisq-edgenum))*tt*(N-tt)*(tt-1)*(N-tt-1)/(N*(N-1)*(N-2)*(N-3))-mu_R1*mu_R2
        var_Rw = ((N-tt-1)/(N-2))**2*v11+2*((N-tt-1)/(N-2))*((tt-1)/(N-2))*v12+((tt-1)/(N-2))**2*v22
        # standardize Rw
        # Zw is the weighted scan statistic
        Zw = -(mu_Rw-Rw)/np.sqrt(np.stack((var_Rw, np.zeros(N)), axis=1).max(axis=1))

        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            # estimate tauhat, the index of the changepoint, to be the time point where the maximum weighted scan statistic occurs
            tauhat = temp[(Zw[n0:(n1+1)]).argmax()]
            scanZ["weighted"] = {"tauhat" : tauhat, "Zmax" : Zw[tauhat], "Zw" : Zw, "Rw" : Rw}

        if gu.anyin({"all", "max", "m", "generalized", "gen", "g"}, statistic):
            # Rd is the difference between R1 and R2
            Rd = R1-R2
            # Zd is the difference scan statistic
            Zd = (Rd-(mu_R1-mu_R2))/np.sqrt(np.stack(((v11+v22-2*v12), np.zeros(N)), axis=1).max(axis=1))

            if gu.anyin({"all", "max", "m"}, statistic):
                # M is the max-type scan statistic
                # this is the maximum between |Zd| and Zw
                M = (np.stack((np.abs(Zd), Zw), axis=1)).max(axis=1)
                # estimate tauhat, the index of the changepoint, to be the time point where the maximum max-type scan statistic occurs
                tauhat = temp[(M[n0:(n1+1)]).argmax()]
                scanZ["max_type"] = {"tauhat" : tauhat, "Zmax" : M[tauhat], "M" : M}

            if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
                # S is the generalized scan statistic
                S = Zw**2+Zd**2
                # estimate tauhat, the index of the changepoint, to be the time point where the maximum generalized scan statistic occurs
                tauhat = temp[(S[n0:(n1+1)]).argmax()]
                scanZ["generalized"] = {"tauhat" : tauhat, "Zmax" : S[tauhat], "S" : S}

    return scanZ


# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ PVAL1() METADATA                                                                                                    ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : pval1                                                                                                 ║
# ║ Purpose     : computes the asymptotic analystical p-value approximations of tail probabilities                      ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def pval1(N, edgelist, ebynode, scanZ, statistic={"all"}, n0=None, n1=None, skew_corr=True):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    lower = n0+1
    upper = n1+1
    output = {}
    deg = np.zeros(N, dtype=np.int64)
    for i in np.arange(N):
        deg[i] = ((ebynode[i]).size)
    sumE = ((deg).sum())/2
    sumEisq = ((deg**2).sum())


    # always return the no_skew 
    # equations 5, 6, 7, and 8 in the graph-based change point review
    output["no_skew"] = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        b = scanZ["original"]["Zmax"]
        if b > 0:
            def integrandO(s):
                # autocorrelation between the scan statistic processes (h0) in the paper
                # (the scan statistics are a time series) Z0(t)
                x = rho_one(N, s, sumE, sumEisq) # finite sample version of asymptotic autocorrelation of process Z0(t)
                return x*Nu(np.sqrt(2*b**2*x))
            pval_ori = norm.pdf(b)*b*integrate.quad(integrandO, a=lower, b=upper, limit=3000)[0]
        else:
            pval_ori = 1
        output["no_skew"]["original"] = np.array([pval_ori, 1]).min()
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        b = scanZ["weighted"]["Zmax"]
        if b > 0:
            def integrandW(t):
                x = rho_one_Rw(N, t)
                return x*Nu(np.sqrt(2*b**2*x))
            pval_wei = norm.pdf(b)*b*integrate.quad(integrandW, a=lower, b=upper, limit=3000)[0]
        else:
            pval_wei = 1
        output["no_skew"]["weighted"] = np.array([pval_wei, 1]).min()
    if gu.anyin({"all", "max", "m",}, statistic):
        b = scanZ["max_type"]["Zmax"]
        if b > 0:
            def integrandM1(t):
                x1 = N/(2*t*(N-t))
                return x1*Nu(np.sqrt(2*b**2*x1))
            def integrandM2(t):
                x2 = rho_one_Rw(N, t)
                return x2*Nu(np.sqrt(2*b**2*x2))
            pval_u1 = 2*norm.pdf(b)*b*integrate.quad(integrandM1, a=lower, b=upper, limit=3000)[0]
            pval_u2 = norm.pdf(b)*b*integrate.quad(integrandM2, a=lower, b=upper, limit=3000)[0]
            pval_max = 1-(1-(np.array([pval_u1, 1]).min()))*(1-(np.array([pval_u2, 1]).min()))
        else:
            pval_max = 1
        output["no_skew"]["max_type"] = pval_max
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        b = scanZ["generalized"]["Zmax"]
        if b > 0:
            def integrandG(t, w):
                x1 = N/(2*t*(N-t))
                x2 = rho_one_Rw(N, t)
                return 2*(x1*np.cos(w)**2+x2*np.sin(w)**2)*b*Nu(np.sqrt(2*b*(x1*np.cos(w)**2+x2*np.sin(w)**2)))/(2*np.pi)
            pval_gen = chi2.pdf(b, 2)*integrate.dblquad(integrandG, a=0, b=2*np.pi, gfun=lower, hfun=upper)[0]
        else:
            pval_gen = 1
        output["no_skew"]["generalized"] = np.array([pval_gen, 1]).min()

    if skew_corr == True:
        output["skew"] = {}
        x1 = ((deg*(deg-1)).sum())
        x2 = ((deg*(deg-1)*(deg-2)).sum())
        x3 = 0
        for i in np.arange((edgelist.shape)[0]):
            x3 += (deg[edgelist[i, 0]]-1)*(deg[edgelist[i, 1]]-1)
        x4 = ((deg*(deg-1)*(sumE-deg)).sum())
        x5 = 0
        for i in np.arange((edgelist.shape)[0]):
            j = edgelist[i, 0]
            k = edgelist[i, 1]
            x5 += ((np.isin(ebynode[j], ebynode[k])).sum())
        
        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            b = scanZ["original"]["Zmax"]
            if b > 0:
                s = np.arange(1, (N+1))
                x = rho_one(N, s, sumE, sumEisq)
                p1 = 2*s*(N-s)/(N*(N-1))
                p2 = 4*s*(s-1)*(N-s)*(N-s-1)/(N*(N-1)*(N-2)*(N-3))
                p3 = s*(N-s)*((N-s-1)*(N-s-2)+(s-1)*(s-2))/(N*(N-1)*(N-2)*(N-3))
                p4 = 8*s*(s-1)*(s-2)*(N-s)*(N-s-1)*(N-s-2)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
                mu = p1*sumE
                sig = np.sqrt(((np.stack(((p2*sumE+(p1/2-p2)*sumEisq+(p2-p1**2)*sumE**2), (np.zeros(N))), axis=1)).max(axis=1)))
                ER3 = p1*sumE+p1/2*3*x1+p2*(3*sumE*(sumE-1)-3*x1)+p3*x2+p2/2*(3*x4-6*x3)+p4*(sumE*(sumE-1)*(sumE-2)- \
                      x2-3*x4+6*x3)-2*p4*x5
                r = (mu**3+3*mu*sig**2-ER3)/sig**3
                pval_ori = pval1_sub0(N, b, r, x, n0, n1)
                if pval_ori is not None:
                    pval_ori = np.array([pval_ori, 1]).min()
            else:
                pval_ori = 1
            output["skew"]["original"] = pval_ori

        if gu.anyin({"all", "weighted", "wei", "w", "max", "m"}, statistic):
            t = np.arange(1, N, dtype=np.float64)
            # expectation of R_1^3
            A1 = sumE*t*(t-1)/(N*(N-1))+3*x1*t*(t-1)*(t-2)/(N*(N-1)*(N-2))+(3*sumE*(sumE-1)-3*x1)*t*(t-1)* \
                 (t-2)*(t-3)/(N*(N-1)*(N-2)*(N-3))+x2*t*(t-1)*(t-2)*(t-3)/(N*(N-1)*(N-2)*(N-3))+(6*x3-6*x5)* \
                 (t*(t-1)*(t-2)*(t-3))/(N*(N-1)*(N-2)*(N-3))+2*x5*(t*(t-1)*(t-2))/(N*(N-1)*(N-2))+(3*x4+6*x5-12*x3)* \
                 t*(t-1)*(t-2)*(t-3)*(t-4)/(N*(N-1)*(N-2)*(N-3)*(N-4))+(sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*t* \
                 (t-1)*(t-2)*(t-3)*(t-4)*(t-5)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
            # expectation of R_1^2 * R_2
            B1 = (sumE*(sumE-1)-x1)*(t*(t-1)*(N-t)*(N-t-1))/(N*(N-1)*(N-2)*(N-3))+(x4+2*x5-4*x3)*(t*(t-1)*(t-2)*(N-t)* \
                 (N-t-1))/(N*(N-1)*(N-2)*(N-3)*(N-4))+(sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*t*(t-1)*(t-2)*(t-3)* \
                 (N-t)*(N-t-1)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
            # expectation of R_1 * R_2^2
            C1 = (sumE*(sumE-1)-x1)*(N-t)*(N-t-1)*t*(t-1)/(N*(N-1)*(N-2)*(N-3))+(x4+2*x5-4*x3)*(N-t)*(N-t-1)*(N-t-2)* \
                 t*(t-1)/(N*(N-1)*(N-2)*(N-3)*(N-4))+(sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*t*(t-1)*(N-t)*(N-t-1)* \
                 (N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
            # expectation of R_2^3
            D1 = sumE*(N-t)*(N-t-1)/(N*(N-1))+3*x1*(N-t)*(N-t-1)*(N-t-2)/(N*(N-1)*(N-2))+(3*sumE*(sumE-1)-3*x1)*(N-t)* \
                 (N-t-1)*(N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3))+x2*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3))+ \
                 (6*x3-6*x5)*((N-t)*(N-t-1)*(N-t-2)*(N-t-3))/(N*(N-1)*(N-2)*(N-3))+2*x5*((N-t)*(N-t-1)*(N-t-2))/(N* \
                 (N-1)*(N-2))+(3*x4+6*x5-12*x3)*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)*(N-t-4)/(N*(N-1)*(N-2)*(N-3)*(N-4))+ \
                 (sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)*(N-t-4)*(N-t-5)/(N*(N-1)* \
                 (N-2)*(N-3)*(N-4)*(N-5))
            r1 = sumE*(t*(t-1)/(N*(N-1)))+2*(0.5*sumEisq-sumE)*t*(t-1)*(t-2)/(N*(N-1)*(N-2))+(sumE*(sumE-1)-(2*(0.5* \
                 sumEisq-sumE)))*t*(t-1)*(t-2)*(t-3)/(N*(N-1)*(N-2)*(N-3))
            r2 = sumE*((N-t)*(N-t-1)/(N*(N-1)))+2*(0.5*sumEisq-sumE)*(N-t)*(N-t-1)*(N-t-2)/(N*(N-1)*(N-2))+(sumE* \
                 (sumE-1)-(2*(0.5*sumEisq-sumE)))*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3))
            r12 = (sumE*(sumE-1)-(2*(0.5*sumEisq-sumE)))*t*(t-1)*(N-t)*(N-t-1)/(N*(N-1)*(N-2)*(N-3))
            x = rho_one_Rw(N, t)
            q = (N-t-1)/(N-2)
            p = (t-1)/(N-2)
            mu = sumE*(q*t*(t-1)+p*(N-t)*(N-t-1))/(N*(N-1))
            sig1 = q**2*r1+2*q*p*r12+p**2*r2-mu**2
            sig = np.sqrt(sig1)
            ER3 = q**3*A1+3*q**2*p*B1+3*q*p**2*C1+p**3*D1
            r = (ER3-3*mu*sig**2-mu**3)/sig**3
            b = scanZ["weighted"]["Zmax"]
            result_u2 = pval1_sub2(N, b, r, x, n0, n1)
            r_Rw = r
            x_Rw = x

            if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
                if (result_u2 is not None) and (result_u2 > 0):
                    pval_wei = np.array([result_u2, 1]).min()
                else:
                    pval_wei = None
                output["skew"]["weighted"] = pval_wei

            if gu.anyin({"all", "max", "m"}, statistic):
                b = scanZ["max_type"]["Zmax"]
                t = np.arange(1, N, dtype=np.float64)
                x = N/(2*t*(N-t))
                q = 1
                p = -1
                mu = sumE*(q*t*(t-1)+p*(N-t)*(N-t-1))/(N*(N-1))
                sig1 = q**2*r1+2*q*p*r12+p**2*r2-mu**2
                sig = np.sqrt(((np.stack(((sig1), (np.zeros((N-1)))), axis=1)).max(axis=1)))
                ER3 = q**3*A1+3*q**2*p*B1+3*q*p**2*C1+p**3*D1
                r = (ER3-3*mu*sig**2-mu**3)/sig**3
                result_u1 = pval1_sub1(N, b, r, x, n0, n1) # p-value for Zdiff
                result_u2 = pval1_sub2(N, b, r_Rw, x_Rw, n0, n1) # p-value for Zw
                if (result_u1 is not None) and (result_u2 is not None) and (result_u1 != 0) and (result_u2 != 0):
                    pval_max = 1-(1-((np.array([result_u1, 1])).min()))*(1-((np.array([result_u2, 1])).min()))
                else:
                    pval_max = None
                output["skew"]["max_type"] = pval_max
                
        # generalized is the same for both skew and no skew correction       
        if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
            output["skew"]["generalized"] = pval_gen
            
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
def pval1_sub0(N, b, r, x, n0, n1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    lower = n0+1
    upper = n1+1
    theta_b = np.zeros(N)
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = (np.sqrt((1+2*r*b)[pos])-1)/r[pos]
    ratio = (np.exp((b-theta_b)**2/2+r*theta_b**3/6))/(np.sqrt(1+r*theta_b))
    a = x*Nu(np.sqrt(2*b**2*x))*ratio
    NN = N-(pos.size)
    if NN > .75*N:
        return None
    if (NN >= (lower-1)+(N-upper)):
        neg = (np.where((1+2*r*b)<=0))[0]
        ns = (neg.size)
        dif = neg[1:ns]-neg[:(ns-1)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
        a[np.int64(N/2):N] = a[np.int64(N/2-1*(N%2==0))::-1]
        a[a<0] = 0
    
    def integrand(s):
        return a[np.int64(s)]
    
    result = None
    try:
        result = norm.pdf(b)*b*integrate.quad(integrand, n0, n1, limit=3000)[0]
    except:
        result = None

    return result

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : pval1_sub1                                                                                            ║
# ║ Purpose     : calculating asymptotic p-value for Zdiff; includes extrapolation                                      ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# extrapolation: calculates the slope
def pval1_sub1(N, b, r, x, n0, n1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    if b < 0:
        return 1
    lower = n0+1
    upper = n1+1
    theta_b = np.zeros((N-1))
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = np.nan_to_num(((np.sqrt((1+2*r*b)[pos])-1)/r[pos]), nan=0, posinf=0, neginf=0)
    ratio = (np.exp((b-theta_b)**2/2+r*theta_b**3/6))/(np.sqrt(1+r*theta_b))
    a = x*Nu(np.sqrt(2*b**2*x))*ratio
    nn_l = np.ceil(N/2)-(((np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)>0))[0]).size)
    nn_r = np.ceil(N/2)-(((np.where((1+2*r[np.int64(np.ceil(N/2-1)):(N-1)]*b)>0))[0]).size)
    if (nn_l > .35*N) or (nn_r > .35*N):
        return 0
    if (nn_l >= lower):
        neg = (np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)<=0))[0]
        dif = np.append(np.diff(neg), (N/2-nn_l))
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
    if nn_r >= (N-upper):
        neg = (np.where((1+2*r[np.int64(np.ceil(N/2-1)):(N-1)]*b)<=0))[0]
        id1 = ((np.append(neg+np.int64(np.ceil(N/2))-1, np.int64(np.ceil(N/2))-2)).min())
        id2 = id1-np.int64(np.ceil(.03*N))
        id3 = id2-np.int64(np.ceil(.09*N))
        inc = (ratio[id3]-ratio[id2])/(id3-id2)
        ratio[id2:(N-1)] = ratio[(id2-1)]+inc*(np.arange((id2+1), N)-(id2+1))
        ratio[ratio<0] = 0
        a[np.int64(N/2-1):(N-1)] = (x*Nu(np.sqrt(2*b**2*x))*ratio)[np.int64(N/2-1):(N-1)]
    a[a<0] = 0

    def integrand(s):
        return a[np.int64(s)]

    result = None
    try:
        result = 2*norm.pdf(b)*b*integrate.quad(integrand, n0, n1, limit=3000)[0]
    except:
        result = None

    return result

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ FUNCTION() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : pval1_sub2                                                                                            ║
# ║ Purpose     : calculating asymptotic p-value for Zw; includes extrapolation                                         ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝    
def pval1_sub2(N, b, r, x, n0, n1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    if b < 0:
        return 0
    lower = n0+1
    upper = n1+1
    theta_b = np.zeros((N-1))
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = (np.sqrt((1+2*r*b)[pos])-1)/r[pos]
    ratio = (np.exp((b-theta_b)**2/2+r*theta_b**3/6))/(np.sqrt(1+r*theta_b))
    a = np.nan_to_num((x*Nu(np.sqrt(2*b**2*x))*ratio), nan=0, posinf=0, neginf=0)
    NN = N-1-(pos.size)
    if NN > .75*N:
        return 0
    if (NN >= (lower-1)+(N-upper)):
        neg = (np.where((1+2*r*b)<=0))[0]
        ns = (neg.size)
        dif = neg[1:ns]-neg[:(ns-1)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
        a[np.int64(N/2):N] = a[np.int64(N/2-1*(N%2==0))::-1]
        a = np.nan_to_num(a, nan=0, posinf=0, neginf=0)
        a[a<0] = 0
    
    def integrand(s):
        return a[np.int64(s)]
    
    result = None
    try:
        result = norm.pdf(b)*b*integrate.quad(integrand, n0, n1, limit=3000)[0]
    except:
        result = None

    return result

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ PERMPVAL1() METADATA                                                                                                ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : permpval1                                                                                             ║
# ║ Purpose     : computes the single change-point p-values by permutation (very slow)                                  ║
# ║ Arguments   :                                                                                                       ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║    - <arg_name> (<type>) :                                                                                          ║
# ║ Returns     :                                                                                                       ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# p-value from permutation for single change point
def permpval1(N, ebynode, scanZ, statistic={"all"}, n0=None, n1=None, B=100):
    Z_ori = np.zeros((B, N))
    Z_wei = np.zeros((B, N))
    Z_max = np.zeros((B, N))
    Z_gen = np.zeros((B, N))
    for b in np.arange(B):
        perm = np.random.choice(N, size=N, replace=False)
        permmatch = np.zeros(N, dtype=np.int64)
        for i in np.arange(N):
            permmatch[perm[i]] = i
        ebnstar = [[] for _ in np.arange(N)]
        for i in np.arange(N):
            oldlinks = ebynode[permmatch[i]]
            ebnstar[i] = perm[oldlinks]
        gcpstar = changepoint(N, ebnstar, statistic, n0, n1)
        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            Z_ori[b, :] = gcpstar["original"]["Z"]
        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            Z_wei[b, :] = gcpstar["weighted"]["Zw"]
        if gu.anyin({"all", "max", "m"}, statistic):
            Z_max[b, :] = gcpstar["max_type"]["M"]
        if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
            Z_gen[b, :] = gcpstar["generalized"]["S"]
    output = {}
    p = 1-(np.arange(B))/B
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        maxZ = (Z_ori[:, n0:(n1+1)]).max(axis=1)
        maxZs = np.sort(maxZ)
        output["original"] = {"pval" : ((maxZs >= scanZ["original"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Z" : Z_ori}
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        maxZ = (Z_wei[:, n0:(n1+1)]).max(axis=1)
        maxZs = np.sort(maxZ)
        output["weighted"] = {"pval" : ((maxZs >= scanZ["weighted"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Z" : Z_wei}
    if gu.anyin({"all", "max", "m"}, statistic):
        maxZ = (Z_max[:, n0:(n1+1)]).max(axis=1)
        maxZs = np.sort(maxZ)
        output["max_type"] = {"pval" : ((maxZs >= scanZ["max_type"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Z" : Z_max}
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        maxZ = (Z_gen[:, n0:(n1+1)]).max(axis=1)
        maxZs = np.sort(maxZ)
        output["generalized"] = {"pval" : ((maxZs >= scanZ["generalized"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Z" : Z_gen}
        
    return output



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ CONTINUOUS CHANGE-INTERVAL DETECTION █                                                                            ▐
# ▌ Purpose : defines functions for detecting change-intervals in the the continuous setting                            ▐
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
def gchangeinterval(E, statistic={"all"}, l0=None, l1=None, pval_asym=True, skew_corr=True, pval_perm=0):
    N = (E.shape)[0]

    # define default values for l0 and l1
    if l0 is None: l0 = .05*N
    if l1 is None: l1 = .95*N
    if l0 <= 1: l0 = 2
    if l1 >= (N-1): l1 = N-2
    if l0 > l1 or l1 < l0:
        l0 = 2
        l1 = N-2
    l0 = np.int64(np.ceil(l0))
    l1 = np.int64(np.floor(l1))

    edgelist = gu.edgematrix_to_edgelist(E)
    ebynode = gu.edgematrix_to_edgebynode(E)
    r1 = {}

    # compute scan statistics
    r1["scanZ"] = changeinterval(N, ebynode, statistic, l0, l1)

    # compute asymptotic p-values
    if pval_asym == True:
        r1["pval_asym"] = pval2(N, edgelist, ebynode, r1["scanZ"], statistic, l0, l1, skew_corr)

    # compute permutation p_values
    pval_perm = round(pval_perm)
    if pval_perm > 0:
        r1["pval_perm"] = permpval2(N, ebynode, r1["scanZ"], statistic, l0, l1, pval_perm)
        
    return r1

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
def changeinterval(N, ebynode, statistic, l0, l1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    node_deg = np.zeros(N)
    for i in np.arange(N):
        node_deg[i] = ((ebynode[i]).size)
    sumEisq = ((node_deg**2).sum())
    edgenum = ((node_deg).sum())/2
    Rtmp = np.zeros((N, N))
    R1 = np.zeros((N, N))
    R2 = np.zeros((N, N))
    Rw = np.zeros((N, N))

    for i in np.arange((N-1)):
        g = np.zeros(N)
        for j in np.arange((i+1), N):
            g[j] = 1
            links = ebynode[j]
            if j == (i+1):
                if (links.size) > 0:
                    Rtmp[i, j] = ((np.tile(g[j], (links.size))!=g[links]).sum())
                    R1[i, j] = 0
                    R2[i, j] = (edgenum-(links.size))
            else:
                if (links.size) > 0:
                    add = ((np.tile(g[j], (links.size))!=g[links]).sum())
                    subtract = (links.size)-add
                    Rtmp[i, j] = Rtmp[i, (j-1)]+add-subtract
                    R1[i, j] = R1[i, (j-1)]+subtract
                else:
                    Rtmp[i, j] = Rtmp[i, (j-1)]
                    R1[i, j] = R1[i, (j-1)]
            R2[i, j] = edgenum-Rtmp[i, j]-R1[i, j]
            Rw[i, j] = ((N-j+i-1)/(N-2))*R1[i, j]+(j-i-1)/(N-2)*R2[i, j]
    dif = np.zeros((N, N), dtype=np.int64)
    for i in np.arange(N):
        for j in np.arange(N):
            dif[i, j] = j-i
    difv = (dif.flatten())
    ids = (np.where(difv>0))[0]
    ids2 = (np.where((difv>=l0)&(difv<=l1)))[0]
    
    scanZ = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        tt = np.arange(1, (N+1))
        mu_t = edgenum*2*tt*(N-tt)/(N*(N-1))
        p1_tt = 2*tt*(N-tt)/(N*(N-1))
        p2_tt = 4*tt*(N-tt)*(tt-1)*(N-tt-1)/(N*(N-1)*(N-2)*(N-3))
        V_tt = p2_tt*edgenum+(p1_tt/2-p2_tt)*sumEisq+(p2_tt-p1_tt**2)*edgenum**2
        Rv = (Rtmp.flatten())
        Zv = np.zeros(N*N)
        Zv[ids] = (mu_t[(difv[ids]-1)]-Rv[ids])/(np.sqrt(V_tt[(difv[ids]-1)]))
        Zmax = ((Zv[ids2]).max())
        Z = Zv.reshape((N, N))
        tauhat = np.concatenate(np.where(Z==Zmax))
        scanZ["original"] = {"tauhat" : tauhat, "Zmax" : Zmax, "Z" : Z, "R" : Rtmp, "Zv" : Zv}
        
    if gu.anyin({"all", "weighted", "wei", "w", "max", "m", "generalized", "gen", "g"}, statistic):
        mu_r1 = edgenum*tt*(tt-1)/(N*(N-1))
        mu_r2 = edgenum*(N-tt)*(N-tt-1)/(N*(N-1))
        sig11 = mu_r1*(1-mu_r1)+2*(0.5*sumEisq-edgenum)*(tt*(tt-1)*(tt-2))/(N*(N-1)*(N-2))+(edgenum*(edgenum-1)- \
                2*(0.5*sumEisq-edgenum))*(tt*(tt-1)*(tt-2)*(tt-3))/(N*(N-1)*(N-2)*(N-3))
        sig22 = mu_r2*(1-mu_r2)+2*(0.5*sumEisq-edgenum)*((N-tt)*(N-tt-1)*(N-tt-2))/(N*(N-1)*(N-2))+ \
                (edgenum*(edgenum-1)-2*(0.5*sumEisq-edgenum))*((N-tt)*(N-tt-1)*(N-tt-2)*(N-tt-3))/(N* \
                (N-1)*(N-2)*(N-3))
        sig12 = (edgenum*(edgenum-1)-2*(0.5*sumEisq-edgenum))*tt*(N-tt)*(tt-1)*(N-tt-1)/(N*(N-1)* \
                (N-2)*(N-3))-mu_r1*mu_r2
        sig21 = (edgenum*(edgenum-1)-2*(0.5*sumEisq-edgenum))*tt*(N-tt)*(tt-1)*(N-tt-1)/(N*(N-1)* \
                (N-2)*(N-3))-mu_r1*mu_r2
        p = (tt-1)/(N-2)
        q = 1-p
        muRw_tt=q*mu_r1+p*mu_r2
        sigRw=q**2*sig11+p**2*sig22+2*p*q*sig12
        Rw_v = (Rw.flatten())
        Zwv = np.zeros(N*N)
        Zwv[ids] = -(muRw_tt[(difv[ids]-1)]-Rw_v[ids])/(np.sqrt(sigRw[(difv[ids]-1)]))
        
        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            Zw = Zwv.reshape((N, N))
            Zmax = ((Zwv[ids2]).max())
            tauhat = np.concatenate(np.where(Zw==Zmax))
            scanZ["weighted"] = {"tauhat" : tauhat, "Zmax" : Zmax, "Zw" : Zw, "Rw" : Rw_v, "Zwv" : Zwv}
            
        if gu.anyin({"all", "max", "m", "generalized", "gen", "g"}, statistic):
            Rsub = R1-R2
            Rsub_v = (Rsub.flatten())
            mu1_tt = (mu_r1-mu_r2)
            sig1 = sig11+sig22-2*sig12
            Zv1 = np.zeros(N*N)
            Zv1[ids] = -(mu1_tt[(difv[ids]-1)]-Rsub_v[ids])/(np.sqrt(sig1[(difv[ids]-1)]))
            
            if gu.anyin({"all", "max", "m"}, statistic):
                Mv = np.zeros(N*N)
                Mv[ids] = ((np.stack((np.abs(Zv1[ids]), Zwv[ids]), axis=1)).max(axis=1))
                Zmax = ((Mv[ids2]).max())
                M = Mv.reshape((N, N))
                tauhat = np.concatenate(np.where(M==Zmax))
                scanZ["max_type"] = {"tauhat" : tauhat, "Zmax" : Zmax, "M" : M, "Mv" : Mv}
                
            if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
                Sv = np.zeros(N*N)
                Sv[ids] = (Zv1[ids])**2+(Zwv[ids])**2
                Zmax = ((Sv[ids2]).max())
                S = Sv.reshape((N, N))
                tauhat = np.concatenate(np.where(S==Zmax))
                scanZ["generalized"] = {"tauhat" : tauhat, "Zmax" : Zmax, "S" : S, "Sv" : Sv}
                
    return scanZ

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
def pval2(N, edgelist, ebynode, scanZ, statistic, l0, l1, skew_corr):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    output = {}
    deg = np.zeros(N, dtype=np.int64)
    for i in np.arange(N):
        deg[i] = ((ebynode[i]).size)
    sumE = ((deg).sum())/2
    sumEisq = ((deg**2).sum())

    output["no_skew"] = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        b = scanZ["original"]["Zmax"]
        if b > 0:
            def integrandO(s):
                x = rho_one(N, s, sumE, sumEisq)
                return (b**2*x*Nu(np.sqrt(2*b**2*x)))**2*(N-s)
            pval_ori = norm.pdf(b)/b*integrate.quad(integrandO, a=l0, b=l1, limit=3000)[0]
        else:
            pval_ori = 1
        output["no_skew"]["original"] = (np.array([pval_ori, 1]).min())
        
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        b = scanZ["weighted"]["Zmax"]
        if b > 0:
            def integrandW(t):
                x = rho_one_Rw(N, t)
                return (b**2*x*Nu(np.sqrt(2*b**2*x)))**2*(N-t)
            pval_wei = norm.pdf(b)/b*integrate.quad(integrandW, a=l0, b=l1, limit=3000)[0]
        else:
            pval_wei = 1
        output["no_skew"]["weighted"] = (np.array([pval_wei, 1]).min())
        
    if gu.anyin({"all", "max", "m"}, statistic):
        b = scanZ["max_type"]["Zmax"]
        if b > 0:
            def integrandM1(t):
                x1 = N/(2*t*(N-t)) 
                return (b**2*x1*Nu(np.sqrt(2*b**2*x1)))**2*(N-t)
                
            def integrandM2(t):
                x2 = rho_one_Rw(N, t)
                return (b**2*x2*Nu(np.sqrt(2*b**2*x2)))**2*(N-t)
            pval_u1 = 2*norm.pdf(b)/b*integrate.quad(integrandM1, a=l0, b=l1, limit=3000)[0]
            pval_u2 = norm.pdf(b)/b*integrate.quad(integrandM2, a=l0, b=l1, limit=3000)[0]
            pval_max = (1-(1-((np.array([pval_u1, 1])).min()))*(1-((np.array([pval_u2, 1])).min())))
        else:
            pval_max = 1
        output["no_skew"]["max_type"] = pval_max
        
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        b = scanZ["generalized"]["Zmax"]
        if b > 0:
            def integrandG(t, w):
                x1 = N/(2*t*(N-t))
                x2 = rho_one_Rw(N, t)
                return (N-t)*(2*(x1*np.cos(w)**2+x2*np.sin(w)**2)*b*Nu(np.sqrt(2*b*(x1*np.cos(w)**2+x2*np.sin(w)**2))))**2/(2*np.pi)
            pval_gen = chi2.pdf(b, 2)*integrate.dblquad(integrandG, a=0, b=2*np.pi, gfun=l0, hfun=l1)[0]
        else:
            pval_gen = 1
        output["no_skew"]["generalized"] = (np.array([pval_gen, 1]).min())

    if skew_corr == True:
        output["skew"] = {}
        x1 = ((deg*(deg-1)).sum())
        x2 = ((deg*(deg-1)*(deg-2)).sum())
        x3 = 0
        for i in np.arange(((edgelist.shape)[0])):
            x3 += (deg[edgelist[i, 0]]-1)*(deg[edgelist[i, 1]]-1)
        x4 = ((deg*(deg-1)*(sumE-deg)).sum())
        x5 = 0
        for i in np.arange(((edgelist.shape)[0])):
            j = edgelist[i, 0]
            k = edgelist[i, 1]
            x5 += ((np.isin(ebynode[j], ebynode[k])).sum())

        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            b = scanZ["original"]["Zmax"]
            if b > 0:
                s = np.arange(1, (N+1))
                x = rho_one(N, s, sumE, sumEisq)
                p1 = 2*s*(N-s)/(N*(N-1))
                p2 = 4*s*(s-1)*(N-s)*(N-s-1)/(N*(N-1)*(N-2)*(N-3))
                p3 = s*(N-s)*((N-s-1)*(N-s-2)+(s-1)*(s-2))/(N*(N-1)*(N-2)*(N-3))
                p4 = 8*s*(s-1)*(s-2)*(N-s)*(N-s-1)*(N-s-2)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
                mu = p1*sumE
                sig = np.sqrt(p2*sumE+(p1/2-p2)*sumEisq+(p2-p1**2)*sumE**2) # sigma
                ER3 = p1*sumE+p1/2*3*x1+p2*(3*sumE*(sumE-1)-3*x1)+p3*x2+p2/2*(3*x4-6*x3)+p4*(sumE*(sumE-1)* \
                      (sumE-2)-x2-3*x4+6*x3)-2*p4*x5
                r = (mu**3+3*mu*sig**2-ER3)/sig**3
                pval_ori = pval2_sub0(N, b, r, x, l0, l1)
                if pval_ori is not None:
                    pval_ori = (np.array([pval_ori, 1]).min())
            else:
                pval_ori = 1
            output["skew"]["original"] = pval_ori
            
        if gu.anyin({"all", "weighted", "wei", "w", "max", "m"}, statistic):
            t = np.arange(1, N, dtype=np.float64)
            A1 = sumE*t*(t-1)/(N*(N-1))+3*x1*t*(t-1)*(t-2)/(N*(N-1)*(N-2))+(3*sumE*(sumE-1)-3*x1)*t*(t-1)*(t-2)* \
                 (t-3)/(N*(N-1)*(N-2)*(N-3))+x2*t*(t-1)*(t-2)*(t-3)/(N*(N-1)*(N-2)*(N-3))+(6*x3-6*x5)*(t*(t-1)*(t-2)* \
                 (t-3))/(N*(N-1)*(N-2)*(N-3))+2*x5*(t*(t-1)*(t-2))/(N*(N-1)*(N-2))+(3*x4+6*x5-12*x3)*t*(t-1)*(t-2)* \
                 (t-3)*(t-4)/(N*(N-1)*(N-2)*(N-3)*(N-4))+(sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*t*(t-1)*(t-2)* \
                 (t-3)*(t-4)*(t-5)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
            B1 = (sumE*(sumE-1)-x1)*(t*(t-1)*(N-t)*(N-t-1))/(N*(N-1)*(N-2)*(N-3))+(x4+2*x5-4*x3)*(t*(t-1)*(t-2)*(N-t)* \
                 (N-t-1))/(N*(N-1)*(N-2)*(N-3)*(N-4))+(sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*t*(t-1)*(t-2)*(t-3)* \
                 (N-t)*(N-t-1)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
            C1 = (sumE*(sumE-1)-x1)*(N-t)*(N-t-1)*t*(t-1)/(N*(N-1)*(N-2)*(N-3))+(x4+2*x5-4*x3)*(N-t)*(N-t-1)*(N-t-2)* \
                 t*(t-1)/(N*(N-1)*(N-2)*(N-3)*(N-4))+(sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*t*(t-1)*(N-t)*(N-t-1)* \
                 (N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5))
            D1 = sumE*(N-t)*(N-t-1)/(N*(N-1))+3*x1*(N-t)*(N-t-1)*(N-t-2)/(N*(N-1)*(N-2))+(3*sumE*(sumE-1)-3*x1)*(N-t)* \
                 (N-t-1)*(N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3))+x2*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3))+ \
                 (6*x3-6*x5)*((N-t)*(N-t-1)*(N-t-2)*(N-t-3))/(N*(N-1)*(N-2)*(N-3))+2*x5*((N-t)*(N-t-1)*(N-t-2))/(N* \
                 (N-1)*(N-2))+(3*x4+6*x5-12*x3)*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)*(N-t-4)/(N*(N-1)*(N-2)*(N-3)*(N-4))+ \
                 (sumE*(sumE-1)*(sumE-2)+6*x3-2*x5-x2-3*x4)*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)*(N-t-4)*(N-t-5)/(N*(N-1)* \
                 (N-2)*(N-3)*(N-4)*(N-5))
            r1 = sumE*(t*(t-1)/(N*(N-1)))+2*(0.5*sumEisq-sumE)*t*(t-1)*(t-2)/(N*(N-1)*(N-2))+(sumE*(sumE-1)- \
                 (2*(0.5*sumEisq-sumE)))*t*(t-1)*(t-2)*(t-3)/(N*(N-1)*(N-2)*(N-3))
            r2 = sumE*((N-t)*(N-t-1)/(N*(N-1)))+2*(0.5*sumEisq-sumE)*(N-t)*(N-t-1)*(N-t-2)/(N*(N-1)*(N-2))+ \
                 (sumE*(sumE-1)-(2*(0.5*sumEisq-sumE)))*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)/(N*(N-1)*(N-2)*(N-3))
            r12 = (sumE*(sumE-1)-(2*(0.5*sumEisq-sumE)))*t*(t-1)*(N-t)*(N-t-1)/(N*(N-1)*(N-2)*(N-3))
            x = rho_one_Rw(N, t)
            q =(N-t-1)/(N-2)
            p = (t-1)/(N-2)
            mu = sumE*(q*t*(t-1)+p*(N-t)*(N-t-1))/(N*(N-1))
            sig1 = q**2*r1+2*q*p*r12+p**2*r2-mu**2
            sig = np.sqrt(sig1) #sigma
            ER3 = q**3*A1+3*q**2*p*B1+3*q*p**2*C1+p**3*D1
            r = (ER3-3*mu*sig**2-mu**3)/sig**3
            b = scanZ["weighted"]["Zmax"]
            result_u2 = pval2_sub2(N, b, r, x, l0, l1)
            r_Rw = r
            x_Rw = x
            
            if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
                if (result_u2 is not None) and (result_u2 > 0):
                    pval_wei = (np.array([result_u2, 1]).min())
                else:
                    pval_wei = None
                output["skew"]["weighted"] = pval_wei
                
            if gu.anyin({"all", "max", "m"}, statistic):
                b = scanZ["max_type"]["Zmax"]
                t = np.arange(1, N)
                x = N/(2*t*(N-t))
                q = 1
                p = -1
                mu = sumE*(q*t*(t-1)+p*(N-t)*(N-t-1))/(N*(N-1))
                sig1 = q**2*r1+2*q*p*r12+p**2*r2-mu**2
                sig = np.sqrt((np.stack((sig1, np.zeros((N-1))), axis=1).max(axis=1))) # sigma
                ER3 = q**3*A1+3*q**2*p*B1+3*q*p**2*C1+p**3*D1
                r = (ER3-3*mu*sig**2-mu**3)/sig**3
                result_u1 = pval2_sub1(N, b, r, x, l0, l1)
                result_u2 = pval2_sub2(N, b, r_Rw, x_Rw, l0, l1)
                if (result_u1 is not None) and (result_u2 is not None) and (result_u1 != 0) and (result_u2 != 0):
                    pval_max = 1-(1-((np.array([result_u1, 1])).min()))*(1-((np.array([result_u2, 1])).min()))
                else:
                    pval_max = None
                output["skew"]["max_type"] = pval_max
                
        if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
            output["skew"]["generalized"] = pval_gen
    
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
def pval2_sub0(N, b, r, x, l0, l1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    theta_b = np.zeros(N)
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = (np.sqrt((1+2*r*b)[pos])-1)/(r[pos])
    ratio = (np.exp((b-theta_b)**2/2+r*theta_b**3/6))/(np.sqrt(1+r*theta_b))
    a = (((b**2*x*Nu(np.sqrt(2*b**2*x)))**2)*ratio)
    NN = N-(pos.size)
    if NN > .75*N:
        return None
    if NN >= (2*l0-1):
        neg = (np.where((1+2*r*b)<=0))[0]
        ns = (neg.size)
        dif = neg[1:ns]-neg[:(ns-1)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
        a[np.int64(N/2):N] = a[np.int64(N/2-1*(N%2==0))::-1]
        a[a<0] = 0

    def integrand(s):
        return a[np.int64(s-1)]*(N-s)

    result = None
    try:
        result = norm.pdf(b)/b*integrate.quad(integrand, a=l0, b=l1, limit=3000)[0]
    except:
        result = None

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
def pval2_sub1(N, b, r, x, l0, l1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    if b < 0:
        return 1
    theta_b = np.zeros((N-1))
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = np.nan_to_num(((np.sqrt((1+2*r*b)[pos])-1)/(r[pos])), nan=0, posinf=0, neginf=0)
    ratio = (np.exp((b-theta_b)**2/2+r*theta_b**3/6))/(np.sqrt(1+r*theta_b))
    a = (((b**2*x*Nu(np.sqrt(2*b**2*x)))**2)*ratio)
    nn_l = np.ceil(N/2)-(((np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)>0))[0]).size)
    nn_r = np.ceil(N/2)-(((np.where((1+2*r[np.int64(np.ceil(N/2-1)):(N-1)]*b)>0))[0]).size)
    if (nn_l > .35*N) or (nn_r > .35*N):
        return 0
    if nn_l >= l0:
        neg = (np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)<=0))[0]
        dif = np.append(np.dif(neg), (N/2-nn_l))
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
    if nn_r >= (N-l1):
        neg = (np.where((1+2*r[np.int64(np.ceil(N/2-1)):N]*b)<=0))[0]
        id1 = ((np.append(neg+np.int64(np.ceil(N/2))-1, np.int64(np.ceil(N/2))-2)).min())
        id2 = id1-np.int64(np.ceil(.03*N))
        id3 = id2-np.int64(np.ceil(.09*N))
        inc = (ratio[id3]-ratio[id2])/(id3-id2)
        ratio[id2:(N-1)] = ratio[(id2-1)]+inc*(np.arange((id2+1), N)-(id2+1))
        ratio[ratio<0] = 0
        a = (((b**2*x*Nu(np.sqrt(2*b**2*x)))**2)*ratio)
    a[a<0] = 0
    
    def integrand(s):
        return a[np.int64(s-1)]*(N-s)
        
    result = None
    try:
        result = 2*norm.pdf(b)/b*integrate.quad(integrand, a=l0, b=l1, limit=3000)[0]
    except:
        result = None

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
def pval2_sub2(N, b, r, x, l0, l1):
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    if b < 0:
        return 1
    theta_b = np.zeros((N-1))
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = (np.sqrt((1+2*r*b)[pos])-1)/(r[pos])
    ratio = (np.exp((b-theta_b)**2/2+r*theta_b**3/6))/(np.sqrt(1+r*theta_b))
    a = np.nan_to_num((((b**2*x*Nu(np.sqrt(2*b**2*x)))**2)*ratio), nan=0, posinf=0, neginf=0)
    NN = N-1-(pos.size)
    if NN > .75*N:
        return 0
    if NN >= ((l0-1)+(N-l1)):
        neg = (np.where((1+2*r*b)<=0))[0]
        ns = (neg.size)
        dif = neg[1:ns]-neg[:(ns-1)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
        a[np.int64(N/2):N] = a[np.int64(N/2-1*(N%2==0))::-1]
        a = np.nan_to_num(a, nan=0, posinf=0, neginf=0)
        a[a<0] = 0
        
    def integrand(s):
        return a[np.int64(s-1)]*(N-s)
        
    result = None
    try:
        result = norm.pdf(b)/b*integrate.quad(integrand, a=l0, b=l1, limit=3000)[0]
    except:
        result = None
        
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
def permpval2(N, ebynode, scanZ, statistic, l0, l1, B):
    Z_ori = np.zeros(B)
    Z_wei = np.zeros(B)
    Z_max = np.zeros(B)
    Z_gen = np.zeros(B)
    for b in np.arange(B):
        perm = np.random.choice(N, size=N, replace=False)
        permmatch = np.zeros(N, dtype=np.int64)
        for i in np.arange(N):
            permmatch[perm[i]] = i
        ebnstar = [[] for _ in np.arange(N)]
        for i in np.arange(N):
            oldlinks = ebynode[permmatch[i]]
            ebnstar[i] = perm[oldlinks]
        gcpstar = changeinterval(N, ebnstar, statistic, l0, l1)
        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            Z_ori[b] = gcpstar["original"]["Zmax"]
        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            Z_wei[b] = gcpstar["weighted"]["Zmax"]
        if gu.anyin({"all", "max", "m"}, statistic):
            Z_max[b] = gcpstar["max_type"]["Zmax"]
        if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
            Z_gen[b] = gcpstar["generalized"]["Zmax"]

    output = {}
    p = 1-(np.arange(B))/B
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        maxZ = ((Z_ori).max())
        maxZs = np.sort(Z_ori)
        output["original"] = {"pval" : ((maxZs >= scanZ["original"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Zmax" : Z_ori}
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        maxZ = ((Z_wei).max())
        maxZs = np.sort(Z_wei)
        output["weighted"] = {"pval" : ((maxZs >= scanZ["weighted"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Zmax" : Z_wei}
    if gu.anyin({"all", "max", "m"}, statistic):
        maxZ = ((Z_max).max())
        maxZs = np.sort(Z_max)
        output["max_type"] = {"pval" : ((maxZs >= scanZ["max_type"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Zmax" : Z_max}
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        maxZ = ((Z_gen).max())
        maxZs = np.sort(Z_gen)
        output["generalized"] = {"pval" : ((maxZs >= scanZ["generalized"]["Zmax"]).sum())/B, "curve" : np.stack((maxZs, p), axis=1), "maxZs" : maxZs, "Zmax" : Z_gen}

    return output



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ PRINTING AND PLOTTING █                                                                                           ▐
# ▌ Purpose : prints and plots the results of gchangepoint or gchangeinterval for quick and easy reading and            ▐
# ▌           visualization                                                                                             ▐
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
# ║ CO_PRINT() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : co_print                                                                                              ║
# ║ Purpose     : summarizes and prints the results of gchangepoint or gchangeinterval in a succinct and readable       ║
# ║               format                                                                                                ║
# ║ Arguments   :                                                                                                       ║
# ║    - results (boolean)        : a dictionary containing the results of gchangepoint or gchangeinterval              ║
# ║    - printEsts (boolean)      : print the estimated changepoints/changeintervals (tauhat)? True if yes, False if no ║
# ║    - printScans (boolean)     : print the scan statistics? True if yes, False if no                                 ║
# ║    - printAsyms (boolean)     : print the asymptotic p-values? True if yes, False if no                             ║
# ║    - printPerms (boolean)     : print the permutation p-values? True if yes, False if no                            ║
# ║    - decimal_places (integer) : the number of decimal places to print the scan statistics and p-values              ║
# ║ Returns     : nothing, this function is just for printing to the console                                            ║
# ║ Author      : written by Alex Wold                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def co_print(results, printEsts=True, printScans=True, printAsyms=True, printPerms=True, decimal_places=4):
    # print estimated change-points/change-intervals and max scan statistics
    if "scanZ" in results:
        # print information about the original test results
        if "original" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = "neither"
            if "tauhat" in results["scanZ"]["original"]:
               point_or_interval = "interval" if (np.asarray(results["scanZ"]["original"]["tauhat"]).size)>1 else "point" 
            else:
                print(f"NO ESTIMATED CHANGEPOINT FOR ORIGINAL TEST")
                return

            print(f"\n\nORIGINAL TEST RESULTS")
            print(f"---------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat" in results["scanZ"]["original"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i): {results['scanZ']['original']['tauhat']}")
                print(f"Estimated change-{point_or_interval} time (t): {results['scanZ']['original']['tauhat']+1}")
            elif ("tauhat" in results["scanZ"]["original"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j): {results['scanZ']['original']['tauhat']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2): {results['scanZ']['original']['tauhat']+1}")

            # print the maximum scan statistic for the original test
            if ("Zmax" in results["scanZ"]["original"]) and (printScans==True):
                print(f"Maximum original scan statistic: {formatfloat(results['scanZ']['original']['Zmax'], decimal_places)}")

            # print asymptotic p-value for the original test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "original" in results["pval_asym"]["no_skew"]:
                        print(f"Original asymptotic p-value, no skew correction: {formatfloat(results['pval_asym']['no_skew']['original'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "original" in results["pval_asym"]["skew"]:
                        print(f"Original asymptotic p-value, with skew correction: {formatfloat(results['pval_asym']['skew']['original'], decimal_places)}")

            # print permutation p-values for the original test
            if ("pval_perm" in results) and (printPerms==True):
                if "original" in results["pval_perm"]:
                    print(f"Original permutation p-value: {formatfloat(results['pval_perm']['original']['pval'], decimal_places)}")
                
                
        # print information about the weighted test results
        if "weighted" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = "neither"
            if "tauhat" in results["scanZ"]["weighted"]:
               point_or_interval = "interval" if (np.asarray(results["scanZ"]["weighted"]["tauhat"]).size)>1 else "point" 
            else:
                print(f"\n\nNO ESTIMATED CHANGEPOINT FOR WEIGHTED TEST")
                return

            print(f"\n\nWEIGHTED TEST RESULTS")
            print(f"---------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat" in results["scanZ"]["weighted"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i): {results['scanZ']['weighted']['tauhat']}")
                print(f"Estimated change-{point_or_interval} time (t): {results['scanZ']['weighted']['tauhat']+1}")
            elif ("tauhat" in results["scanZ"]["weighted"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j): {results['scanZ']['weighted']['tauhat']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2): {results['scanZ']['weighted']['tauhat']+1}")

            # print the maximum scan statistic for the weighted test
            if ("Zmax" in results["scanZ"]["weighted"]) and (printScans==True):
                print(f"Maximum weighted scan statistic: {formatfloat(results['scanZ']['weighted']['Zmax'], decimal_places)}")

            # print asymptotic p-value for the weighted test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "weighted" in results["pval_asym"]["no_skew"]:
                        print(f"Weighted asymptotic p-value, no skew correction: {formatfloat(results['pval_asym']['no_skew']['weighted'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "weighted" in results["pval_asym"]["skew"]:
                        print(f"Weighted asymptotic p-value, with skew correction: {formatfloat(results['pval_asym']['skew']['weighted'], decimal_places)}")

            # print permutation p-values for the weighted test
            if ("pval_perm" in results) and (printPerms==True):
                if "weighted" in results["pval_perm"]:
                    print(f"Weighted permutation p-value: {formatfloat(results['pval_perm']['weighted']['pval'], decimal_places)}")

        # print information about the weighted test results
        if "max_type" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = "neither"
            if "tauhat" in results["scanZ"]["max_type"]:
               point_or_interval = "interval" if (np.asarray(results["scanZ"]["max_type"]["tauhat"]).size)>1 else "point" 
            else:
                print(f"\n\nNO ESTIMATED CHANGEPOINT FOR MAX-TYPE TEST")
                return

            print(f"\n\nMAX-TYPE TEST RESULTS")
            print(f"---------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat" in results["scanZ"]["max_type"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i): {results['scanZ']['max_type']['tauhat']}")
                print(f"Estimated change-{point_or_interval} time (t): {results['scanZ']['max_type']['tauhat']+1}")
            elif ("tauhat" in results["scanZ"]["max_type"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j): {results['scanZ']['max_type']['tauhat']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2): {results['scanZ']['max_type']['tauhat']+1}")

            # print the maximum scan statistic for the max_type test
            if ("Zmax" in results["scanZ"]["max_type"]) and (printScans==True):
                print(f"Maximum max-type scan statistic: {formatfloat(results['scanZ']['max_type']['Zmax'], decimal_places)}")

            # print asymptotic p-value for the max_type test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "max_type" in results["pval_asym"]["no_skew"]:
                        print(f"Max-type asymptotic p-value, no skew correction: {formatfloat(results['pval_asym']['no_skew']['max_type'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "max_type" in results["pval_asym"]["skew"]:
                        print(f"Max-type asymptotic p-value, with skew correction: {formatfloat(results['pval_asym']['skew']['max_type'], decimal_places)}")

            # print permutation p-values for the max_type test
            if ("pval_perm" in results) and (printPerms==True):
                if "max_type" in results["pval_perm"]:
                    print(f"Max-type permutation p-value: {formatfloat(results['pval_perm']['max_type']['pval'], decimal_places)}")

        # print information about the weighted test results
        if "generalized" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = "neither"
            if "tauhat" in results["scanZ"]["generalized"]:
               point_or_interval = "interval" if (np.asarray(results["scanZ"]["generalized"]["tauhat"]).size)>1 else "point" 
            else:
                print(f"\n\nNO ESTIMATED CHANGEPOINT FOR GENERALIZED TEST")
                return

            print(f"\n\nGENERALIZED TEST RESULTS")
            print(f"------------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat" in results["scanZ"]["generalized"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i): {results['scanZ']['generalized']['tauhat']}")
                print(f"Estimated change-{point_or_interval} time (t): {results['scanZ']['generalized']['tauhat']+1}")
            elif ("tauhat" in results["scanZ"]["generalized"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j): {results['scanZ']['generalized']['tauhat']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2): {results['scanZ']['generalized']['tauhat']+1}")

            # print the maximum scan statistic for the generalized test
            if ("Zmax" in results["scanZ"]["generalized"]) and (printScans==True):
                print(f"Maximum generalized scan statistic: {formatfloat(results['scanZ']['generalized']['Zmax'], decimal_places)}")

            # print asymptotic p-value for the generalized test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "generalized" in results["pval_asym"]["no_skew"]:
                        print(f"Generalized asymptotic p-value, no skew correction: {formatfloat(results['pval_asym']['no_skew']['generalized'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "generalized" in results["pval_asym"]["skew"]:
                        print(f"Generalized asymptotic p-value, with skew correction: {formatfloat(results['pval_asym']['skew']['generalized'], decimal_places)}")

            # print permutation p-values for the generalized test
            if ("pval_perm" in results) and (printPerms==True):
                if "generalized" in results["pval_perm"]:
                    print(f"Generalized permutation p-value: {formatfloat(results['pval_perm']['generalized']['pval'], decimal_places)}")

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ CO_PLOT() METADATA                                                                                                  ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : co_plot                                                                                               ║
# ║ Purpose     : calls co_plot_test_statistic to plot a line graph (for gchangepoint results) or heatmap and surface   ║
# ║               plot (for gchangeinterval results) of the Z test statistics. no plotting is done in this function, it ║
# ║               just dispatches the co_plot_test_statistic function. the resulting plots will NOT be interactive. it  ║
# ║               uses matplotlib for static plots. the plots for each test in the results['scanZ'] dictionary will be  ║
# ║               drawn                                                                                                 ║
# ║ Arguments   :                                                                                                       ║
# ║    - results (dictionary) : the dictionary returned from either gchangepoint or gchangeinterval. this must have the ║
# ║                             'scanZ' key containing the test statistics for each performed test                      ║
# ║ Returns     : nothing                                                                                               ║
# ║ Author      : written by Alex Wold                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def co_plot(results):
    if "original" in results["scanZ"]:
        orowdim = results["scanZ"]["original"]["Z"].shape[0]
        originalZ = results["scanZ"]["original"]["Z"].reshape((orowdim, -1))
        co_plot_test_statistic(originalZ, r"$Z$", "Original")
    if "weighted" in results["scanZ"]:
        wrowdim = results["scanZ"]["weighted"]["Zw"].shape[0]
        weightedZ = results["scanZ"]["weighted"]["Zw"].reshape((wrowdim, -1))
        co_plot_test_statistic(weightedZ, r"$Z_{w}$", "Weighted")
    if "max_type" in results["scanZ"]:
        mrowdim = results["scanZ"]["max_type"]["M"].shape[0]
        max_typeZ = results["scanZ"]["max_type"]["M"].reshape((mrowdim, -1))
        co_plot_test_statistic(max_typeZ, r"$M$", "Max-type")
    if "generalized" in results["scanZ"]:
        growdim = results["scanZ"]["generalized"]["S"].shape[0]
        generalizedZ = results["scanZ"]["generalized"]["S"].reshape((growdim, -1))
        co_plot_test_statistic(generalizedZ, r"$S$", "Generalized")

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ CO_PLOT_TEST_STATISTIC() METADATA                                                                                   ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : co_plot_test_statistic                                                                                ║
# ║ Purpose     : uses matplotlib to plot the test statistics of gchangepoint or gchangeinterval results                ║
# ║                  - draws a static line plot for the single change-point setting (gchangepoint results)              ║
# ║                  - draws a static heatmap and surface plot for the change-interval setting (gchangeinterval         ║
# ║                    results                                                                                          ║
# ║ Arguments   :                                                                                                       ║
# ║    - dimarray (ndarray shape: (N, 1) or (N, N)) : the (N, 1) array or the (N, N) matrix of test statistics to plot  ║
# ║    - testtype (string, raw string literal)      : the raw string literal representing what test statistic to plot   ║
# ║                                                      - r"$Z$" for the original test statistic                       ║
# ║                                                      - r"$Z_{w}$" for the weighted test statistic                   ║
# ║                                                      - r"$M$" for the max-type test statistic                       ║
# ║                                                      - r"$S$" for the generalized test statistic                    ║
# ║                                                    this argument is mostly used for figure/axis/legend titles       ║
# ║    - testname (string)                          : the string value representing the name of the test. is one of:    ║
# ║                                                      - "Original"                                                   ║
# ║                                                      - "Weighted"                                                   ║
# ║                                                      - "Max-type"                                                   ║
# ║                                                      - "Generalized"                                                ║
# ║                                                   this argument is mostly used for figure/axis/legend titles        ║
# ║ Returns     : nothing: this function only draws plots                                                               ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def co_plot_test_statistic(dimarray, testtype, testname):
    dimarray = np.nan_to_num(dimarray, nan=0, posinf=0, neginf=0)
    stripped_type = testtype[1:-1] # strip out the $ characters
    if dimarray.shape[1] == 1:
        indices = np.arange(dimarray.shape[0])
        maxTSindex = dimarray.argmax() # TS stands for test statistic, get the index of the maximum test statistic
        maxTS = dimarray[maxTSindex] # get the maximum test statistic value
        fig, vax = plt.subplots()
        fig.suptitle(testname)
        vax.plot(indices, dimarray, color="blue", label=testtype, zorder=-1)
        vax.scatter((maxTSindex), maxTS, marker="o", color="red", alpha=1, zorder=1)
        vax.vlines((dimarray.argmax()), ymin=0, ymax=1, linestyles="dotted", color="red", transform=vax.get_xaxis_transform(),
                  label=(r"max " + testtype + r" index: " + str(maxTSindex)))
        plt.xlabel(r"$\text{obs index: } \left( i \right)$")
        plt.ylabel(r"$\text{test statistic: } \left(" + stripped_type + r"\right)$")
        plt.legend()
        plt.show()
    if dimarray.shape[1] > 1:
        # transpose so t2 (interval endpoint) is on the y-axis
        dimarray = np.transpose(dimarray)

        # mask the invalid lower triangle
        masked = np.ma.masked_where(np.triu(np.ones_like(dimarray)), dimarray)

        # get the indices of the maximum test statistic
        maxTSindices = np.unravel_index(dimarray.argmax(), dimarray.shape)

        # plot the 2D heatmap
        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(testname)
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(masked, origin="lower")
        cbar = fig.colorbar(im, ax=ax1, orientation="vertical", label=r"test statistic: " + testtype)
        ax1.scatter(maxTSindices[1], maxTSindices[0], marker=".", color="red", alpha=1)
        ax1.vlines(maxTSindices[1], ymin=0, ymax=1, linestyles="dotted", color="red", transform=ax1.get_xaxis_transform(),
                   label=r"max " + testtype + r" start index: $i=$" + str(maxTSindices[1]))
        ax1.hlines(maxTSindices[0], xmin=0, xmax=1, linestyles="dotted", color="red", transform=ax1.get_yaxis_transform(),
                   label=r"max " + testtype + r" end index: $j=$" + str(maxTSindices[0]))
        ax1.set_xlabel(r"interval start index: $i$")
        ax1.set_ylabel(r"interval end index: $j$")
        ax1.legend(loc="lower right")

        # plot the 3D contour plot
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        rows, cols = dimarray.shape # cols refers to x-coordinate, rows refers to y coordinates
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        surf = ax2.plot_surface(X, Y, masked, cmap=cm.viridis, vmin=np.min(dimarray), vmax=np.max(dimarray))
        cbar2 = ax2.figure.colorbar(surf, ax=ax2, orientation="vertical", label=r"test statistic: " + testtype)
        max_zval = np.max(np.abs(dimarray))
        ax2.set(xlim=(0, cols), ylim=(0, rows), zlim=(-max_zval, max_zval))
        ax2.contourf(X[0, :], Y[:, 0], masked, zdir="z", offset=-max_zval, cmap=cm.viridis, alpha=.4)
        ax2.scatter(maxTSindices[1], maxTSindices[0], masked[maxTSindices[0], maxTSindices[1]], marker="o", color="red")
        max_x = maxTSindices[1]
        max_y = maxTSindices[0]
        ax2.scatter(max_x, max_y, -max_zval, color="red", marker="o")
        ax2.plot([max_x, max_x], [max_y, max_y], [-max_zval, max_zval], linestyle="dotted", color="red") # vertical
        ax2.plot([max_x, max_x], [0, (dimarray.shape[1]-1)], [-max_zval, -max_zval], linestyle="dotted", color="red") # t1
        ax2.plot([0, (dimarray.shape[0]-1)], [max_y, max_y], [-max_zval, -max_zval], linestyle="dotted", color="red") # t2

        # set axis labels
        ax2.set_xlabel(r"interval start index: $i$") # X axis: interval start index (i)
        ax2.set_ylabel(r"interval end index: $j$") # Y axis: interval end index (j)
        ax2.set_zlabel("test statistic") # Z axis: 
        ax2.view_init(elev=30, azim=-90.5, roll=0)
    
        plt.tight_layout()
        plt.show()

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ CO_PLOT_INTERACTIVE() METADATA                                                                                      ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : co_plot_interactive                                                                                   ║
# ║ Purpose     : calls co_plot_test_statistic_interactive to plot a line graph (for gchangepoint results) or heatmap   ║
# ║               and surface plot (for gchangeinterval results) of the Z test statistics. no plotting is done in this  ║
# ║               function, it just dispatches the co_plot_test_statistic_interactive function. the resulting plots ARE ║
# ║               interactive (built from plotly) and are meant to be used in an interactive notebook setting or within ║
# ║               standalone html contexts. the plots for each test in the results['scanZ'] dictionary will be drawn.   ║
# ║ Arguments   :                                                                                                       ║
# ║    - results (dictionary) : the dictionary returned from either gchangepoint or gchangeinterval. this must have the ║
# ║                             'scanZ' key containing the test statistics for each performed test                      ║
# ║ Returns     : nothing                                                                                               ║
# ║ Author      : written by Alex Wold                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def co_plot_interactive(results):
    if "original" in results["scanZ"]:
        orowdim = results["scanZ"]["original"]["Z"].shape[0]
        originalZ = results["scanZ"]["original"]["Z"].reshape((orowdim, -1))
        co_plot_test_statistic_interactive(originalZ, r"$Z$", r"Original")
    if "weighted" in results["scanZ"]:
        wrowdim = results["scanZ"]["weighted"]["Zw"].shape[0]
        weightedZ = results["scanZ"]["weighted"]["Zw"].reshape((wrowdim, -1))
        co_plot_test_statistic_interactive(weightedZ, r"$Z_{w}$", r"Weighted")
    if "max_type" in results["scanZ"]:
        mrowdim = results["scanZ"]["max_type"]["M"].shape[0]
        max_typeZ = results["scanZ"]["max_type"]["M"].reshape((mrowdim, -1))
        co_plot_test_statistic_interactive(max_typeZ, r"$M$", r"Max-type")
    if "generalized" in results["scanZ"]:
        growdim = results["scanZ"]["generalized"]["S"].shape[0]
        generalizedZ = results["scanZ"]["generalized"]["S"].reshape((growdim, -1))
        co_plot_test_statistic_interactive(generalizedZ, r"$S$", r"Generalized")

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ CO_PLOT_TEST_STATISTIC_INTERACTIVE() METADATA                                                                       ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : co_plot_test_statistic_interactive                                                                    ║
# ║ Purpose     : uses plotly to plot the test statistics of gchangepoint or gchangeinterval results                    ║
# ║                  - draws an interactive line plot for the single change-point setting (gchangepoint results)        ║
# ║                  - draws an interactive heatmap and surface plot for the change-interval setting (gchangeinterval   ║
# ║                    results)                                                                                         ║
# ║ Arguments   :                                                                                                       ║
# ║    - dimarray (ndarray shape: (N, 1) or (N, N)) : the (N, 1) array or the (N, N) matrix of test statistics to plot  ║
# ║    - testtype (string, raw string literal)      : the raw string literal representing what test statistic to plot   ║
# ║                                                      - r"$Z$" for the original test statistic                       ║
# ║                                                      - r"$Z_{w}$" for the weighted test statistic                   ║
# ║                                                      - r"$M$" for the max-type test statistic                       ║
# ║                                                      - r"$S$" for the generalized test statistic                    ║
# ║                                                    this argument is mostly used for figure/axis/legend titles       ║
# ║    - testname (string)                          : the string value representing the name of the test. is one of:    ║
# ║                                                      - "Original"                                                   ║
# ║                                                      - "Weighted"                                                   ║
# ║                                                      - "Max-type"                                                   ║
# ║                                                      - "Generalized"                                                ║
# ║                                                   this argument is mostly used for figure/axis/legend titles        ║
# ║ Returns     : nothing: this function only draws plots                                                               ║
# ║ Author      : translated from the gSeg R package by Alex Wold                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def co_plot_test_statistic_interactive(dimarray, testtype, testname):
    dimarray = np.nan_to_num(dimarray, nan=0, posinf=0, neginf=0)
    stripped_type = testtype[1:-1] # strip out the $ characters
    
    if dimarray.shape[1] == 1:
        # get axis of observation indices (0 to N-1)
        indices = np.arange(dimarray.shape[0])

        # get the index and value of the maximum test statistic
        maxTSindex = dimarray.argmax()
        maxTS = dimarray[maxTSindex]
        
        # create a 1D line plot with a marker at the maximum
        fig = go.Figure()

        # line plot of test statistic vs observation index
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=dimarray[:, 0],
                mode="lines",
                name=testtype,
                line=dict(color="blue")
            )
        )

        # line marking index of the maximum test statistic
        fig.add_shape(
            type="line",
            x0=maxTSindex, x1=maxTSindex,
            y0=0, y1=1, # yref="paper", so y0=0 and y1=1 means span entire plot height
            xref="x",
            yref="paper",
            line=dict(color="red", dash="dash", width=2)
        )

        # marker at the maximum test statistic
        fig.add_trace(
            go.Scatter(
                x=[float(maxTSindex)],
                y=[float(maxTS[0])],
                mode="markers",
                name=r"$\text{max } " + stripped_type + r"$",
                marker=dict(color="red", size=10)
            )
        )

        # plot information: title, axis, legend, etc
        fig.update_layout(
            title=testname,
            xaxis_title=r"$\text{obs index: } \left(i\right)$",
            yaxis_title=r"$\text{test statistic: } \left(" + stripped_type + r"\right)$",
            legend=dict(
                x=.98,
                y=.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",   # semi-transparent white background
                bordercolor="black",
                borderwidth=1)
        )
        
        fig.show()

    if dimarray.shape[1] > 1:
        # Transpose so t2 is on the y-axis
        dimarray = np.transpose(dimarray)
        rows, cols = dimarray.shape # cols refers to the x-coordinates, rows refers to the y-coordinates
        
        # mask the invalid lower triangle
        # np.ones_like returns a matrix of all ones the size of dimarray
        # then np.triu returns only the upper triangle of this
        # np.where uses this as a condition to set the lower triangle to nan, and the upper triangle to dimarray
        masked = np.where(np.triu(np.ones_like(dimarray)), np.nan, dimarray)
        
        # find the maximum test statistic
        maxTSindices = np.unravel_index(dimarray.argmax(), dimarray.shape)
        max_x = maxTSindices[1]
        max_y = maxTSindices[0]
        zmax = np.max(dimarray)
        zmin = np.min(dimarray)
        z_floor = np.full_like(dimarray, zmin)
        z_floor[np.isnan(masked)] = np.nan
        rows, cols = dimarray.shape # rows are y-coordinates, cols are x-coordinates
        # meshrid used for plotting 3d surface
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

        # create subplots: 1 row, 2 columns: (1, 1) is heatmap, (1, 2) is surface plot
        fig = psub.make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "heatmap"}, {"type": "surface"}]]
        )
        
        # heatmap
        fig.add_trace(
            go.Heatmap(
                z=masked,
                x=np.arange(cols),
                y=np.arange(rows),
                colorscale="Viridis",
                showscale=False,
                name="contour"
            ),
            row=1,
            col=1
        )

        # add red line for max start index (vertical line) to heatmap
        fig.add_shape(
            type="line",
            x0=max_x, x1=max_x,
            y0=0, y1=1, # yref="paper", so y0=0 and y1=1 means span entire plot height
            xref="x",
            yref="paper",
            line=dict(color="red", dash="dash", width=2),
            name="start index"
        )

        # add red line for max end index (horizontal line) to heatmap
        fig.add_shape(
            type="line",
            x0=0, x1=(cols-1),
            y0=max_y, y1=max_y,
            xref="x",
            yref="y",
            line=dict(color="red", dash="dash", width=2),
            name="end index"
        )
        
        # red marker on heatmap
        fig.add_trace(
            go.Scatter(
                x=[max_x],
                y=[max_y],
                mode="markers",
                marker=dict(color="red", size=10),
                name="max test statistic",
                showlegend=False
            ),
            row=1,
            col=1
        )

        # change axis limits for heatmap
        colorbar=dict(showticklabels=False, showcolorbar=False)
        fig.update_xaxes(constrain='range', range=[0, (cols-1)], row=1, col=1, autorange=False,
                         title=r"$\text{interval start index: } i$")
        fig.update_yaxes(constrain='range', range=[0, (rows-1)], row=1, col=1, autorange=False,
                         title=r"$\text{interval end index: } j$")
        
        # surface plot
        fig.add_trace(
            go.Surface(
                z=masked,
                x=np.arange(cols),
                y=np.arange(rows),
                colorscale="Viridis",
                colorbar=dict(title="test statistic"),
                showscale=True,
                name="surface"
            ),
            row=1,
            col=2
        )

        # add filled contour plot to floor of surface plot
        fig.add_trace(
            go.Surface(
                z=z_floor,
                x=X,
                y=Y,
                surfacecolor=masked,
                colorscale="Viridis",
                cmin=zmin,
                cmax=zmax,
                showscale=False,
                opacity=.4,
                name="contour"
            ),
            row=1,
            col=2
        )
        
        # red line for interval start index
        fig.add_trace(
            go.Scatter3d(
                x=[max_x, max_x],
                y=[0, (rows-1)],
                z=[zmin, zmin],
                mode="lines",
                line=dict(
                    color="red",
                    width=5,
                    dash="dash"
                ),
                showlegend=False,
                name="start index"
            ),
            row=1,
            col=2
        )
        
        # red line for interval end index
        fig.add_trace(
            go.Scatter3d(
                x=[0, (cols-1)],
                y=[max_y, max_y],
                z=[zmin, zmin],
                mode="lines",
                line=dict(
                    color="red",
                    width=5,
                    dash="dash"
                ),
                showlegend=False,
                name="end index"
            ),
            row=1,
            col=2
        )

        # red vertical line up to max test statistic
        fig.add_trace(
            go.Scatter3d(
                x=[max_x, max_x],
                y=[max_y, max_y],
                z=[zmin, zmax],
                mode="lines",
                line=dict(
                    color="red",
                    width=5,
                    dash="dash"
                ),
                showlegend=False,
                name="floor to max"
            ),
            row=1,
            col=2
        )

        # marker at the maximum test statistic, duplicated on floor
        fig.add_trace(
            go.Scatter3d(
                x=[max_x, max_x],
                y=[max_y, max_y],
                z=[zmin, zmax],
                mode="markers",
                marker=dict(color="red", size=5),
                name="max test statistic",
                showlegend=False
            ),
            row=1,
            col=2
        )
        
        # plot information: title, axis, legend, etc
        fig.update_layout(
            title_text=testname,
            scene=dict(
                xaxis_title=r"interval start index: i",
                yaxis_title=r"interval end index: j",
                zaxis_title="test statistic",
                xaxis=dict(range=[0, (cols-1)]),
                yaxis=dict(range=[0, (rows-1)]),
                zaxis=dict(range=[zmin, zmax]),
                aspectmode="cube"
            ),
            height=600,
            width=1300,
            scene_camera=dict(
                eye=dict(
                x=0.0188,
                y=-2.165,
                z=1.25
                )
            )
        )

        fig.show()
