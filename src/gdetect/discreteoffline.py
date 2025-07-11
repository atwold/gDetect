# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █ ██████ discreteoffline ██████                                                                                       █
# █ Purpose : provides the two key functions for detecting change-points and change-intervals in the discrete setting   █
# █              - gchangepoint_discrete: used for detecting single change-points                                       █
# █              - gchangeinterval_discrete: used for detecting change-intervals                                        █
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



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ CORRECTION FACTORS, AUTOCORRELATION FUNCTIONS, AND OTHER SHARED FUNCTIONS █                                       ▐
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
def Nu(x):
    x = np.float64(x)
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
def rho_w(N, t):
    N = np.float64(N)
    t = np.float64(t)
    return ((2*t-1)*(N-t)*(N-t-1)-t*(t-1)*(2*t-2*N+1))/2/t/(t-1)/(N-t)/(N-t-1)

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
def rho_d(N, t):
    N = np.float64(N)
    t = np.float64(t)
    return N/2/t/(N-t)

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
def rho_one_discrete(N, t, one, two, three):
    N = np.float64(N)
    t = np.float64(t)
    one = np.float64(one)
    two = np.float64(two)
    three = np.float64(three)
    np.seterr(divide="ignore")
    f1 = 4*(N-1)*(2*t*(N-t)-N)
    f2 = ((N+1)*(N-2*t)**2-2*N*(N-1))
    f3 = 4*((N-2*t)**2-N)
    f4 = 4*N*(t-1)*(N-1)*(N-t-1)
    f5 = N*(N-1)*((N-2*t)**2-(N-2))
    f6 = 4*((N-2)*(N-2*t)**2-2*t*(N-t)+N)
    return N*(N-1)*(f1*one+f2*two-f3*three)/(2*t*(N-t)*(f4*one+f5*two-f6*three))

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
# this is a support function for calculating the asymptotic p-values
# that is, this is a support function for pval1_discrete() and pval2_discrete()
# this provides moments E(Rw_a^3), E(Rw_u^3), E(Rd_a^3), and E(Rd_u^3)
def skewcorr(edgelist, order):
    N = np.float64(order.size)
    K = np.float64((np.unique(order)).size)
    edgenum = np.float64((edgelist.shape)[0])
    mk = (np.unique(order, return_counts=True))[1]
    t = np.arange(1, (N+1), dtype=np.float64)

    p_hat = (t-1)/(N-2)
    q_hat = 1-p_hat
    p1 = t*(t-1)/N/(N-1)
    p2 = t*(t-1)*(t-2)/N/(N-1)/(N-2)
    p3 = t*(t-1)*(t-2)*(t-3)/N/(N-1)/(N-2)/(N-3)
    p4 = t*(t-1)*(t-2)*(t-3)*(t-4)/N/(N-1)/(N-2)/(N-3)/(N-4)
    p5 = t*(t-1)*(t-2)*(t-3)*(t-4)*(t-5)/N/(N-1)/(N-2)/(N-3)/(N-4)/(N-5)
    p6 = t*(t-1)*(N-t)*(N-t-1)/N/(N-1)/(N-2)/(N-3)
    p7 = t*(t-1)*(t-2)*(N-t)*(N-t-1)/N/(N-1)/(N-2)/(N-3)/(N-4)
    p8 = t*(t-1)*(t-2)*(t-3)*(N-t)*(N-t-1)/N/(N-1)/(N-2)/(N-3)/(N-4)/(N-5)
    q1 = (N-t)*(N-t-1)/N/(N-1)
    q2 = (N-t)*(N-t-1)*(N-t-2)/N/(N-1)/(N-2)
    q3 = (N-t)*(N-t-1)*(N-t-2)*(N-t-3)/N/(N-1)/(N-2)/(N-3)
    q4 = (N-t)*(N-t-1)*(N-t-2)*(N-t-3)*(N-t-4)/N/(N-1)/(N-2)/(N-3)/(N-4)
    q5 = (N-t)*(N-t-1)*(N-t-2)*(N-t-3)*(N-t-4)*(N-t-5)/N/(N-1)/(N-2)/(N-3)/(N-4)/(N-5)
    q7 = t*(t-1)*(N-t)*(N-t-1)*(N-t-2)/N/(N-1)/(N-2)/(N-3)/(N-4)
    q8 = t*(t-1)*(N-t)*(N-t-1)*(N-t-2)*(N-t-3)/N/(N-1)/(N-2)/(N-3)/(N-4)/(N-5)

    Ebynode = [[] for _ in np.arange(np.int64(K), dtype=np.int64)]
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        Ebynode[edgelist[i, 0]].append(edgelist[i, 1])
        Ebynode[edgelist[i, 1]].append(edgelist[i, 0])

    temp28 = temp29 = temp30 = temp31 = temp41 = temp45 = 0.0
    node_deg_a = np.zeros(np.int64(K), dtype=np.float64)
    node_deg_u = np.zeros(np.int64(K), dtype=np.float64)
    quan = quan1 = quan2 = quan3 = quan4 = quan5 = quan_u = 0.0
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        node_deg_a[e1] += 1
        node_deg_a[e2] += 1
        node_deg_u[e1] += mk[e2]
        node_deg_u[e2] += mk[e1]
        quan += 1/mk[e1]/mk[e2]
        quan_u += mk[e1]*mk[e2]
        quan1 += (mk[e1]+mk[e2])/mk[e1]**2/mk[e2]**2
        quan2 += (mk[e1]+mk[e2])**2/mk[e1]/mk[e2]
        quan3 += (mk[e1]**2+mk[e2]**2)/mk[e1]/mk[e2]
        quan4 += 1/mk[e1]/mk[e1]/mk[e2]/mk[e2]
        quan5 += (mk[e1]**2+mk[e2]**2)/mk[e1]/mk[e1]/mk[e2]/mk[e2]
        e3 = np.intersect1d(Ebynode[e1], Ebynode[e2])
        e4 = ((np.isin(Ebynode[e1], Ebynode[e2])).sum())
        temp28 += e4
        temp29 += ((1/mk[e3]/mk[e1]/mk[e2]).sum())
        temp30 += (((mk[e3]+mk[e1]+mk[e2])/mk[e3]/mk[e1]/mk[e2]).sum())
        temp31 += (((mk[e3]*mk[e1]+mk[e3]*mk[e2]+mk[e1]*mk[e2])/mk[e3]/mk[e1]/mk[e2]).sum())
        temp41 += (1/mk[e1]+1/mk[e2])*e4
        temp45 += (mk[e1]+mk[e2]-1)*e4/mk[e1]/mk[e2]
    G = (((mk*(mk-1)).sum())/2)+quan_u
    node_deg_u += mk-1
    temp47 = ((node_deg_u*(node_deg_u-1)*mk).sum())
    temp49 = ((node_deg_u*(node_deg_u-1)*(G-node_deg_u)*mk).sum())
    temp34 = temp35 = temp36 = temp37 = temp42 = temp51 = temp52 = 0
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        e3 = np.intersect1d(Ebynode[e1], Ebynode[e2])
        e4 = ((np.isin(Ebynode[e1], Ebynode[e2])).sum())
        temp34 = (node_deg_a[e1]-1)*(node_deg_a[e2]-1)
        temp35 += 6*(temp34-e4)/mk[e1]/mk[e2]
        temp36 += 6*(temp34-e4)*((mk[e1]+mk[e2]-2)/mk[e1]/mk[e2])
        temp37 += 6*(temp34-e4)*((mk[e1]*mk[e2]-mk[e1]-mk[e2]+1)/mk[e1]/mk[e2])
        temp42 += temp34
        temp51 += ((mk[e1]*mk[e2]*mk[e3]).sum())+3*mk[e1]*(mk[e1]-1)*mk[e2]/2+3*mk[e2]*(mk[e2]-1)*mk[e1]/2
        temp52 += mk[e1]*mk[e2]*(node_deg_u[e1]-1)*(node_deg_u[e2]-1)
    temp51 += 3*((mk*(mk-1)*(mk-2)/6).sum())
    temp52 += ((mk*(mk-1)*(node_deg_u-1)**2/2).sum())

    temp1 = ((node_deg_a/mk).sum())
    temp2 = ((node_deg_a/(mk**2)).sum())
    temp3 = ((1/mk).sum())
    temp4 = ((mk*node_deg_a).sum())
    temp5 = ((mk**2).sum())
    temp6 = ((1/(mk**2)).sum())
    temp7 = ((node_deg_a**2).sum())
    temp8 = ((node_deg_a**2/mk).sum())
    temp9 = ((node_deg_a**2*mk).sum())
    temp10 = ((node_deg_a**2*mk**2).sum())
    temp12 = ((node_deg_a**2/mk/mk).sum())
    temp13 = ((node_deg_a**3).sum())
    temp15 = ((node_deg_a**3/mk).sum())
    temp33 = ((node_deg_a**3/mk/mk).sum())
    temp43 = ((node_deg_a*(node_deg_a-1)*(3*edgenum-2*node_deg_a-2)).sum())

    temp16, temp19, temp20, temp23, temp38 = np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
    for i in np.arange(np.int64(K), dtype=np.int64):
        temp16 = np.append(temp16, ((mk[Ebynode[i]]).sum()))
        temp19 = np.append(temp19, (((node_deg_a[Ebynode[i]]-1)*(node_deg_a[Ebynode[i]]-2)).sum()))
        temp20 = np.append(temp20, ((node_deg_a[np.setdiff1d(np.delete(np.arange(np.int64(K), dtype=np.int64), i), Ebynode[i])]*(node_deg_a[np.setdiff1d(np.delete(np.arange(np.int64(K), dtype=np.int64), i), Ebynode[i])]-1)).sum()))
        temp23 = np.append(temp23, ((1/mk[Ebynode[i]]).sum()))
        temp38 = np.append(temp38, ((node_deg_a[Ebynode[i]]).sum()))
    temp17 = (((node_deg_a-1)*(1-1/mk)*((N-mk)*node_deg_a-2*temp16)).sum())
    temp18 = (((node_deg_a-1)*((N-mk)*node_deg_a-2*temp16)/mk).sum())
    temp21 = (((mk-1)*((edgenum-node_deg_a)*(edgenum-node_deg_a-1)-temp19-temp20)).sum())
    temp25 = (((node_deg_a-1)*temp23).sum())
    temp26 = (((node_deg_a-1)*temp23/mk).sum())
    temp27 = (((node_deg_a-1)*temp23/mk**2).sum())
    temp39 = (((node_deg_a-1)*temp38).sum())
    temp40 = (((node_deg_a-1)*temp38/mk).sum())

    temp22 = temp32 = temp46 = temp44 = w = w1 = w2 = w3 = 0
    for i in np.arange(np.int64(K), dtype=np.int64):
        for j in Ebynode[i]:
            temp22 += (node_deg_a[j]-1)*((4*p3-10*p4+6*p5)/mk[j]+(2*p4-2*p5)*mk[i]/mk[j]-(4*p3-8*p4+4*p5)/mk[i]/mk[j])
            temp32 += (edgenum-node_deg_a[i]-node_deg_a[j]+1)*(p5+(p3-2*p4+p5)/mk[i]/mk[j]+(p4-p5)*(1/mk[i]+1/mk[j]))
            temp44 += (node_deg_a[j]-1)*((18*p8-10*p7)/mk[j]+mk[i]*(2*p7-6*p8)/mk[j]+(8*p7-12*p8)/mk[i]/mk[j])
            temp46 += (edgenum-node_deg_a[i]-node_deg_a[j]+1)*(3*p8+(p6-2*p7+3*p8)/mk[i]/mk[j]+(p7-3*p8)*(1/mk[i]+1/mk[j]))
            w += (node_deg_a[j]-1)*((4*q3-10*q4+6*q5)/mk[j]+(2*q4-2*q5)*mk[i]/mk[j]-(4*q3-8*q4+4*q5)/mk[i]/mk[j])
            w1 += (edgenum-node_deg_a[i]-node_deg_a[j]+1)*(q5+(q3-2*q4+q5)/mk[i]/mk[j]+(q4-q5)*(1/mk[i]+1/mk[j]))
            w2 += (node_deg_a[j]-1)*((18*q8-10*q7)/mk[j]+mk[i]*(2*q7-6*q8)/mk[j]+(8*q7-12*q8)/mk[i]/mk[j])
            w3 += (edgenum-node_deg_a[i]-node_deg_a[j]+1)*(3*q8+(p6-2*q7+3*q8)/mk[i]/mk[j]+(q7-3*q8)*(1/mk[i]+1/mk[j]))
    
    R1_a3 = p1*(4*temp3-4*temp6+quan4)+p2*(32*K-96*temp3+64*temp6+12*temp1-12*temp2+24*quan-9*quan1-6*quan4+3*temp27+2*temp29)+p3*\
            (6*(N-K)*K+32*N-216*K+(6*K+96)*edgenum-(6*N-6*K+6*edgenum-412)*temp3-228*temp6+temp33-12*temp12+83*temp2+12*temp8-132*temp1+(3*N-3*K-75)*\
            quan+13*quan4+30*quan1+quan5-6*temp29+2*temp30+9*temp26-12*temp27+temp35)+p4*(12*(N-K)*(N-3*K)-72*(N-5*K)+24*edgenum**2+(36*N-60*K-216)*\
            edgenum-3*temp33+3*temp7-141*temp2+3*(edgenum-K-5)*temp8+30*temp12-(15*edgenum+9*N-12*K-240)*temp1+24*(N-K+edgenum-24)*temp3+288*temp6-6*(N-K-13)*quan-\
            12*quan4-33*quan1+3*quan3-3*quan2-3*quan5+6*temp29-4*temp30+2*temp31+3*temp18+3*temp25-18*temp26+15*temp27+temp36-6*temp40+3*temp41)+\
            p5*((N-K)**3-3*(N-K)*(4*N-10*K)+40*N-176*K-42*edgenum**2+(3*(N-K)**2-33*N+57*K+95)*edgenum-2*temp13+3*(edgenum-K+6)*temp7+2*temp33+70*temp2-3*(edgenum-K-1)*\
            temp8-18*temp12+(9*N-12*K+15*edgenum-120)*temp1-(18*N-18*K+18*edgenum-256)*temp3-120*temp6-3*temp9+(6*edgenum-3)*temp4+3*(N-K-9)*quan+4*quan4+12*\
            quan1-3*quan3+2*quan5+3*quan2+3*temp17+3*temp21-3*temp25+9*temp26-6*temp27-2*temp29+2*temp30-2*temp31+temp37-6*temp39+6*temp40+6*\
            temp28-3*temp41+6*temp42-temp43+edgenum*(edgenum-1)*(edgenum-2))+3*temp22+1.5*temp32
    R1_a2R2_a = p6*(2*(N-K)*K-8*K+2*K*edgenum-2*(edgenum+N-K-10)*temp3-12*temp6-4*temp1+4*temp2+(N-K-3)*quan+quan1+quan4+temp26-temp27)+p7*\
                (8*edgenum**2+(12*N-20*K-72)*edgenum+4*(N-K)*(N-3*K)-24*N+120*K+8*(edgenum+N-K-24)*temp3+96*temp6+temp7-temp33-(5*edgenum+3*N-4*K-80)*\
                temp1-47*temp2+(edgenum-K-5)*temp8+10*temp12-(2*N-2*K-26)*quan+quan3-11*quan1-quan2-quan5-4*quan4+2*temp29-temp30+temp45+\
                temp18+temp25-6*temp26+5*temp27+temp36/3-2*temp40+temp41)+p8*((N-K)**3-3*(N-K)*(4*N-10*K)+40*N-176*K-42*edgenum**2+\
                (3*(N-K)**2-33*N+57*K+95)*edgenum+2*temp33-2*temp13-3*temp9+(6*edgenum-3)*temp4-(18*N-18*K+18*edgenum-256)*temp3-120*temp6+3*(edgenum-K+6)*\
                temp7+(15*edgenum+9*N-12*K-120)*temp1+70*temp2+(3*K-3*edgenum+3)*temp8-18*temp12+3*(N-K-9)*quan+12*quan1-3*quan3+3*quan2+2*quan5+\
                4*quan4+3*temp17+3*temp21-3*temp25+9*temp26-6*temp27-2*temp29+2*temp30-2*temp31+temp37-6*temp39+6*temp40+6*temp28-3*\
                temp41+edgenum*(edgenum-1)*(edgenum-2)+6*temp42-temp43)+temp44+0.5*temp46
    R1_aR2_a2 = p6*(2*(N-K)*K-8*K+2*K*edgenum-2*(edgenum+N-K-10)*temp3-12*temp6-4*temp1+4*temp2+(N-K-3)*quan+quan1+quan4+temp26-temp27)+q7*\
                (8*edgenum**2+(12*N-20*K-72)*edgenum+4*(N-K)*(N-3*K)-24*N+120*K+8*(edgenum+N-K-24)*temp3+96*temp6+temp7-temp33-(5*edgenum+3*N-4*K-80)*temp1-\
                47*temp2+(edgenum-K-5)*temp8+10*temp12-(2*N-2*K-26)*quan+quan3-11*quan1-quan2-quan5-4*quan4+2*temp29-temp30+temp45+temp18+\
                temp25-6*temp26+5*temp27+temp36/3-2*temp40+temp41)+q8*((N-K)**3-3*(N-K)*(4*N-10*K)+40*N-176*K-42*edgenum**2+(3*(N-K)**2-33*N+57*K+95)*\
                edgenum+2*temp33-2*temp13-3*temp9+(6*edgenum-3)*temp4-(18*N-18*K+18*edgenum-256)*temp3-120*temp6+3*(edgenum-K+6)*temp7+(15*edgenum+9*N-12*K-120)*temp1+\
                70*temp2+(3*K-3*edgenum+3)*temp8-18*temp12+3*(N-K-9)*quan+12*quan1-3*quan3+3*quan2+2*quan5+4*quan4+3*temp17+3*temp21-3*temp25+9*\
                temp26-6*temp27-2*temp29+2*temp30-2*temp31+temp37-6*temp39+6*temp40+6*temp28-3*temp41+edgenum*(edgenum-1)*(edgenum-2)+6*temp42-temp43)+w2+0.5*w3
    R2_a3 = q1*(4*temp3-4*temp6+quan4)+q2*(32*K-96*temp3+64*temp6+12*temp1-12*temp2+24*quan-9*quan1-6*quan4+3*temp27+2*temp29)+q3*\
            (6*(N-K)*K+32*N-216*K+(6*K+96)*edgenum-(6*N-6*K+6*edgenum-412)*temp3-228*temp6+temp33-12*temp12+83*temp2+12*temp8-132*temp1+\
            (3*N-3*K-75)*quan+13*quan4+30*quan1+quan5-6*temp29+2*temp30+9*temp26-12*temp27+temp35)+q4*(12*(N-K)*(N-3*K)-72*(N-5*K)+24*\
            edgenum**2+(36*N-60*K-216)*edgenum-3*temp33+3*temp7-141*temp2+3*(edgenum-K-5)*temp8+30*temp12-(15*edgenum+9*N-12*K-240)*temp1+24*(N-K+edgenum-24)*temp3+\
            288*temp6-6*(N-K-13)*quan-12*quan4-33*quan1+3*quan3-3*quan2-3*quan5+6*temp29-4*temp30+2*temp31+3*temp18+3*temp25-18*temp26+\
            15*temp27+temp36-6*temp40+3*temp41)+q5*((N-K)**3-3*(N-K)*(4*N-10*K)+40*N-176*K-42*edgenum**2+(3*(N-K)**2-33*N+57*K+95)*edgenum-2*temp13+\
            3*(edgenum-K+6)*temp7+2*temp33+70*temp2-3*(edgenum-K-1)*temp8-18*temp12+(9*N-12*K+15*edgenum-120)*temp1-(18*N-18*K+18*edgenum-256)*temp3-120*temp6-\
            3*temp9+(6*edgenum-3)*temp4+3*(N-K-9)*quan+4*quan4+12*quan1-3*quan3+2*quan5+3*quan2+3*temp17+3*temp21-3*temp25+9*temp26-6*temp27-2*\
            temp29+2*temp30-2*temp31+temp37-6*temp39+6*temp40+6*temp28-3*temp41+6*temp42-temp43+edgenum*(edgenum-1)*(edgenum-2))+3*w+1.5*w1
    R1_u3 = p1*G+p2*(3*temp47+2*temp51)+p3*(3*G*(G-1)-3*temp47+np.sum(node_deg_u*(node_deg_u-1)*(node_deg_u-2)*mk)+6*temp52-6*temp51)+p4*\
            (3*temp49+6*temp51-12*temp52)+p5*(G*(G-1)*(G-2)+6*temp52-2*temp51-np.sum(node_deg_u*(node_deg_u-1)*(3*G-2*node_deg_u-2)*mk))
    R1_u2R2_u = p6*(G*(G-1)-temp47)+p7*(temp49+2*temp51-4*temp52)+p8*(G*(G-1)*(G-2)+6*temp52-2*temp51-np.sum(node_deg_u*(node_deg_u-1)*\
                (node_deg_u-2)*mk)-3*temp49)
    R1_uR2_u2 = p6*(G*(G-1)-temp47)+q7*(temp49+2*temp51-4*temp52)+q8*(G*(G-1)*(G-2)+6*temp52-2*temp51-np.sum(node_deg_u*(node_deg_u-1)*\
                (node_deg_u-2)*mk)-3*temp49)
    R2_u3 = q1*G+q2*(3*temp47+2*temp51)+q3*(3*G*(G-1)-3*temp47+np.sum(node_deg_u*(node_deg_u-1)*(node_deg_u-2)*mk)+6*temp52-6*temp51)+q4*\
            (3*temp49+6*temp51-12*temp52)+q5*(G*(G-1)*(G-2)+6*temp52-2*temp51-np.sum(node_deg_u*(node_deg_u-1)*(3*G-2*node_deg_u-2)*mk))

    p_hat = (t-1)/(N-2)
    q_hat = 1-p_hat
    ER3w_a = q_hat**3*R1_a3+3*q_hat**2*p_hat*R1_a2R2_a+3*q_hat*p_hat**2*R1_aR2_a2+p_hat**3*R2_a3 # E(Rw_a^3)
    ER3w_u = q_hat**3*R1_u3+3*q_hat**2*p_hat*R1_u2R2_u+3*q_hat*p_hat**2*R1_uR2_u2+p_hat**3*R2_u3 # E(Rw_u^3)
    
    q_hat = 1
    p_hat = -1
    ER3d_a = q_hat**3*R1_a3+3*q_hat**2*p_hat*R1_a2R2_a+3*q_hat*p_hat**2*R1_aR2_a2+p_hat**3*R2_a3 # E(Rd_a^3)
    ER3d_u = q_hat**3*R1_u3+3*q_hat**2*p_hat*R1_u2R2_u+3*q_hat*p_hat**2*R1_uR2_u2+p_hat**3*R2_u3 # E(Rd_u^3)
    
    q_hat = 1
    p_hat = 1
    ER3_a = q_hat**3*R1_a3+3*q_hat**2*p_hat*R1_a2R2_a+3*q_hat*p_hat**2*R1_aR2_a2+p_hat**3*R2_a3
    ER3_u = q_hat**3*R1_u3+3*q_hat**2*p_hat*R1_u2R2_u+3*q_hat*p_hat**2*R1_uR2_u2+p_hat**3*R2_u3

    return {"ER3w_a" : ER3w_a, "ER3w_u" : ER3w_u, "ER3d_a" : ER3d_a, "ER3d_u" : ER3d_u, "ER3_a" : ER3_a, "ER3_u" : ER3_u}



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ DISCRETE SINGLE CHANGE-POINT DETECTION █                                                                          ▐
# ▌ Purpose : defines functions for detecting single change-points in the discrete setting                              ▐
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
def gchangepoint_discrete(E, order, statistic={"all"}, n0=None, n1=None, pval_asym=True, skew_corr=True, pval_perm=0):
    N = np.float64(order.size) # the edge matrix should be NxN (square)
    edgelist = gu.edgematrix_to_edgelist(E) # convert the edge matrix to an edge list
    
    # define default values for n0 and n1, which are functions of N
    if n0 is None:
        n0 = np.int64(np.ceil(.05*N)-1)
    if n1 is None:
        n1 = np.int64(np.floor(.95*N)-1)
    
    if n0 < 1 or n0 >= n1:
        n0 = np.int64(1)
    else:
        n0 = np.int64(np.ceil(n0))
    if n1 > (N-3) or n1 <= n0:
        n1 = np.int64(N-3)
    else:
        n1 = np.int64(np.floor(n1))

    # create an empty dictionary that will hold the results
    r1 = {}
    r1["scanZ"] = changepoint1_discrete(edgelist, order, statistic, n0, n1)

    # compute the asymptotic p-values
    if pval_asym == True:
        r1["pval_asym"] = pval1_discrete(edgelist, order, r1["scanZ"], statistic, skew_corr, n0, n1)

    # compute the permuatation p-values
    pval_perm = round(pval_perm)
    if pval_perm > 0:
        r1["pval_perm"] = permpval1_discrete(edgelist, order, r1["scanZ"], statistic, pval_perm, n0, n1)

    # store meta information about the function call
    r1["meta"] = {"type" : "point", "start" : n0, "end" : n1}
    
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
# edgelist is the edgenum by 2 matrix that represents the similarity graph (MST, NNL, etc)
# order is the index of observations (which are numbered by their order in the sequence)
# order is a numpy array
def changepoint1_discrete(edgelist, order, statistic, n0=None, n1=None):
    np.seterr(divide="ignore", invalid="ignore")
    N = np.float64(order.size) # total number of nodes
    # define default values for n0 and n1, which are functions of N
    if n0 is None:
        n0 = np.int64(np.ceil(.05*N)-1)
    if n1 is None:
        n1 = np.int64(np.floor(.95*N)-1)
    
    if n0 < 1 or n0 >= n1:
        n0 = np.int64(1)
    else:
        n0 = np.int64(np.ceil(n0))
    if n1 > (N-3) or n1 <= n0:
        n1 = np.int64(N-3)
    else:
        n1 = np.int64(np.floor(n1))

    temp = getMeanVar_changepoint1_discrete(edgelist, order)
    muo_a = temp["muo_a"]
    muo_u = temp["muo_u"]
    varo_a = temp["varo_a"]
    varo_u = temp["varo_u"]
    mu1_a = temp["mu1_a"]
    mu2_a = temp["mu2_a"]
    var1_a = temp["var1_a"]
    var2_a = temp["var2_a"]
    var12_a = temp["var12_a"]
    mu1_u = temp["mu1_u"]
    mu2_u = temp["mu2_u"]
    var1_u = temp["var1_u"]
    var2_u = temp["var2_u"]
    var12_u = temp["var12_u"]

    t = np.arange(1, (N+1), dtype=np.float64) # from 1 to N
    p_hat = (t-1)/(N-2)
    q_hat = 1-p_hat

    # expectation and variance of the weighted edge-count test (average method)
    muw_a = q_hat*mu1_a+p_hat*mu2_a
    varw_a = q_hat**2*var1_a+p_hat**2*var2_a+2*q_hat*p_hat*var12_a
    
    # expectation and variance of the difference of two within group edge counts (average method)
    mud_a = mu1_a-mu2_a
    vard_a = (np.concatenate(((var1_a+var2_a-2*var12_a).reshape((np.int64(N), -1)), np.zeros((np.int64(N), 1), dtype=np.float64)), axis=1)).max(axis=1)

    # expectation and variance of the weighted edge count test (union method)
    muw_u = q_hat*mu1_u+p_hat*mu2_u 
    varw_u = q_hat**2*var1_u+p_hat**2*var2_u+2*q_hat*p_hat*var12_u

    # expectation and variance of the difference of two within group edge counts (union method)
    mud_u = mu1_u-mu2_u
    vard_u = (np.concatenate(((var1_u+var2_u-2*var12_u).reshape((np.int64(N), -1)), np.zeros((np.int64(N), 1), dtype=np.float64)), axis=1)).max(axis=1)

    temp = getR1R2_discrete1(edgelist, order)
    R1_a = temp["R1_a"]
    R2_a = temp["R2_a"]
    R1_u = temp["R1_u"]
    R2_u = temp["R2_u"]
    Ro_a = temp["Ro_a"]
    Ro_u = temp["Ro_u"]

    Rw_a = q_hat*R1_a+p_hat*R2_a
    Rd_a = R1_a-R2_a
    Zw_a = (Rw_a-muw_a)/np.sqrt(varw_a)
    Zd_a = (Rd_a-mud_a)/np.sqrt(vard_a)

    Rw_u = q_hat*R1_u+p_hat*R2_u
    Rd_u = R1_u-R2_u
    Zw_u = (Rw_u-muw_u)/np.sqrt(varw_u)
    Zd_u = (Rd_u-mud_u)/np.sqrt(vard_u)

    temp = np.arange(n0, (n1+1), dtype=np.int64)
    scanZ = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        Zo_a = -(Ro_a-muo_a)/np.sqrt(varo_a)
        Zo_u = -(Ro_u-muo_u)/np.sqrt(varo_u)
        tauhat_a = temp[((Zo_a[n0:(n1+1)]).argmax())]
        tauhat_u = temp[((Zo_u[n0:(n1+1)]).argmax())]
        scanZ["original"] = {"tauhat_a" : tauhat_a, "Zo_a_max" : Zo_a[tauhat_a], "Zo_a" : Zo_a, "Ro_a" : Ro_a,
                             "tauhat_u" : tauhat_u, "Zo_u_max" : Zo_u[tauhat_u], "Zo_u" : Zo_u, "Ro_u" : Ro_u}
        
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        tauhat_a = temp[((Zw_a[n0:(n1+1)]).argmax())]
        tauhat_u = temp[((Zw_u[n0:(n1+1)]).argmax())]
        scanZ["weighted"] = {"tauhat_a" : tauhat_a, "Zw_a_max" : Zw_a[tauhat_a], "Zw_a" : Zw_a, "Rw_a" : Rw_a,
                             "tauhat_u" : tauhat_u, "Zw_u_max" : Zw_u[tauhat_u], "Zw_u" : Zw_u, "Rw_u" : Rw_u}
        
    if gu.anyin({"all", "max", "m"}, statistic):
        M_a = (np.concatenate(((Zw_a).reshape((-1, 1)), (np.abs(Zd_a)).reshape((-1, 1))), axis=1)).max(axis=1)
        M_u = (np.concatenate(((Zw_u).reshape((-1, 1)), (np.abs(Zd_u)).reshape((-1, 1))), axis=1)).max(axis=1)
        tauhat_a = temp[((M_a[n0:(n1+1)]).argmax())]
        tauhat_u = temp[((M_u[n0:(n1+1)]).argmax())]
        scanZ["max_type"] = {"tauhat_a" : tauhat_a, "M_a_max" : M_a[tauhat_a], "M_a" : M_a,
                             "tauhat_u" : tauhat_u, "M_u_max" : M_u[tauhat_u], "M_u" : M_u}
        
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        S_a = Zw_a**2+Zd_a**2
        S_u = Zw_u**2+Zd_u**2
        tauhat_a = temp[((S_a[n0:(n1+1)]).argmax())]
        tauhat_u = temp[((S_u[n0:(n1+1)]).argmax())]
        scanZ["generalized"] = {"tauhat_a" : tauhat_a, "S_a_max" : S_a[tauhat_a], "S_a" : S_a,
                                "tauhat_u" : tauhat_u, "S_u_max" : S_u[tauhat_u], "S_u" : S_u}
    
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
# GETMEANVAR_CHANGEPOINT_DISCRETE()
# this is a supporting function of changepoint1_discrete()
# this function returns the mean, variance, or covariance of test statistics like E(R1_a(t))
# edgelist is the edgenum by 2 matrix of edges that defines the similarity matrix
# order is the order of observations (which are numbered by their order in the sequence)
def getMeanVar_changepoint1_discrete(edgelist, order):
    N = np.float64(order.size) # total number of nodes (number of obs)
    K = np.float64((np.unique(order)).size) # number of unique obs
    edgenum = np.float64((edgelist.shape)[0]) # number of edges
    
    t = np.arange(1, (N+1), dtype=np.float64) # from 1 to N
    p1 = t*(t-1)/N/(N-1)
    p2 = t*(t-1)*(t-2)/N/(N-1)/(N-2)
    p3 = t*(t-1)*(t-2)*(t-3)/N/(N-1)/(N-2)/(N-3)
    q1 = (N-t)*(N-t-1)/N/(N-1)
    q2 = (N-t)*(N-t-1)*(N-t-2)/N/(N-1)/(N-2)
    q3 = (N-t)*(N-t-1)*(N-t-2)*(N-t-3)/N/(N-1)/(N-2)/(N-3)
    f1 = t*(t-1)*(N-t)*(N-t-1)/N/(N-1)/(N-2)/(N-3)
    p0 = t*(N-t)/N/(N-1)

    muo_a = (N-K+edgenum)*2*p0
    mu1_a = (N-K+edgenum)*p1
    mu2_a = (N-K+edgenum)*q1

    node_deg_a = np.zeros((np.int64(K), ), dtype=np.float64)
    node_deg_u = np.zeros((np.int64(K), ), dtype=np.float64)
    quan = 0
    quan_u = 0
    mk = (np.unique(order, return_counts=True))[1]

    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        node_deg_a[e1] += 1
        node_deg_a[e2] += 1
        node_deg_u[e1] += mk[e2]
        node_deg_u[e2] += mk[e1]
        quan += 1/mk[e1]/mk[e2]
        quan_u += mk[e1]*mk[e2]

    temp1 = N-K+2*edgenum+((node_deg_a**2/4/mk).sum())-((node_deg_a/mk).sum())
    temp2 = K-((1/mk).sum())
    temp3 = quan
    temp4 = (N-K+edgenum)**2

    varo_a = 4*(p0-4*f1)*temp1+(24*f1-4*p0)*temp2+4*f1*temp3+(4*f1-4*p0**2)*temp4
    var1_a = 4*(p2-p3)*temp1+2*(p1-4*p2+3*p3)*temp2+(p1-2*p2+p3)*temp3+(p3-p1**2)*temp4
    var2_a = 4*(q2-q3)*temp1+2*(q1-4*q2+3*q3)*temp2+(q1-2*q2+q3)*temp3+(q3-q1**2)*temp4
    var12_a = f1*(-4*temp1+6*temp2+temp3)+(f1-p1*q1)*temp4

    G = (((mk*(mk-1)).sum())/2)+quan_u
    node_deg_u += mk-1

    muo_u = G*2*p0
    mu1_u = G*p1
    mu2_u = G*q1

    varo_u = (2*p0-4*f1)*G+(p0-4*f1)*((mk*node_deg_u*(node_deg_u-1)).sum())+4*(f1-p0**2)*G**2
    var1_u = (p1-p3)*G+(p2-p3)*((mk*node_deg_u*(node_deg_u-1)).sum())+(p3-p1**2)*G**2
    var2_u = (q1-q3)*G+(q2-q3)*((mk*node_deg_u*(node_deg_u-1)).sum())+(q3-q1**2)*G**2
    var12_u = f1*(G**2-G-((mk*node_deg_u*(node_deg_u-1)).sum()))-p1*q1*G**2

    return {"mu1_a" : mu1_a, "mu2_a" : mu2_a, "var1_a" : var1_a, "var2_a" : var2_a, "var12_a" : var12_a,
            "mu1_u" : mu1_u, "mu2_u" : mu2_u, "var1_u" : var1_u, "var2_u" : var2_u, "var12_u" : var12_u,
            "muo_a" : muo_a, "muo_u" : muo_u, "varo_a" : varo_a, "varo_u" : varo_u}

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
# GETR1R2_DISCRETE1()
# this is a supporting function of changepoint1_discrete()
# this function gives test statistics such as R1_a(t), R2_u(t), Ro_a(t)
# edgelist is the edgenum by 2 matrix of edges that defines the similarity matrix
def getR1R2_discrete1(edgelist, order):
    N = np.float64(order.size) # total number of nodes (number of obs)
    K = np.float64((np.unique(order)).size) # number of unique obs
    edgenum = np.float64((edgelist.shape)[0]) # number of edges
    mk = (np.unique(order, return_counts=True))[1]
    temp1 = ((mk**2-mk).sum())/2
    temp2 = 0
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        temp2 += mk[e1]*mk[e2]
    R1_a = np.zeros(np.int64(N), dtype=np.float64)
    R2_a = np.zeros(np.int64(N), dtype=np.float64)
    R1_u = np.zeros(np.int64(N), dtype=np.float64)
    R2_u = np.zeros(np.int64(N), dtype=np.float64)
    Ro_a = np.zeros(np.int64(N), dtype=np.float64)
    Ro_u = np.zeros(np.int64(N), dtype=np.float64)

    V = np.zeros((2, np.int64(K)), dtype=np.float64)
    for i in np.arange(np.int64(N), dtype=np.int64):
        if i == 0:
            V[1, :] = mk
            V[0, order[i]] = 1
            V[1, order[i]] -= 1
            So_a1 = (V[0, :]*V[1, :]*2/mk).sum()
            So_u1 = (V[0, :]*V[1, :]).sum()
            So_a2 = 0
            So_u2 = 0
            for k in np.arange(i, np.int64(edgenum), dtype=np.int64):
                So_a2 += (V[0, edgelist[k, 0]]*V[1, edgelist[k, 1]]+V[0, edgelist[k, 1]]*V[1, edgelist[k, 0]])/mk[edgelist[k, 0]]/mk[edgelist[k, 1]]
                So_u2 += V[0, edgelist[k, 0]]*V[1, edgelist[k, 1]]+V[0, edgelist[k, 1]]*V[1, edgelist[k, 0]]
            R1_a[i] = 0
            Ro_a[i] = So_a1+So_a2
            R2_a[i] = N-K+edgenum-R1_a[i]-Ro_a[i]
            R1_u[i] = 0
            Ro_u[i] = So_u1+So_u2
            R2_u[i] = temp1+temp2-R1_u[i]-Ro_u[i]
        else:
            V[0, order[i]] += 1
            V[1, order[i]] -= 1
            A = np.concatenate(np.where(edgelist==order[i]), dtype=np.int64).reshape((-1, 2), order="F")
            A[:, 1] = 1-A[:, 1]
            R1_a[i] = R1_a[i-1]+2*(V[0, order[i]]-1)/mk[order[i]]+((V[0, edgelist[A[:, 0], A[:, 1]]]/mk[edgelist[A[:, 0], A[:, 1]]]).sum())/mk[order[i]]
            Ro_a[i] = Ro_a[i-1]+(2-(4*V[0, order[i]]-2)/mk[order[i]])+(((V[1, edgelist[A[:, 0], A[:, 1]]]-V[0, edgelist[A[:, 0], A[:, 1]]])/mk[edgelist[A[:, 0], A[:, 1]]]).sum())/mk[order[i]]
            R1_u[i] = R1_u[i-1]+(V[0, order[i]]-1)+((V[0, edgelist[A[:, 0], A[:, 1]]]).sum())
            Ro_u[i] = Ro_u[i-1]+(mk[order[i]]-2*V[0, order[i]]+1)+((V[1, edgelist[A[:, 0], A[:, 1]]]-V[0, edgelist[A[:, 0], A[:, 1]]]).sum())
        R2_a[i] = N-K+edgenum-R1_a[i]-Ro_a[i]
        R2_u[i] = temp1+temp2-R1_u[i]-Ro_u[i]
    return {"R1_a" : R1_a, "R2_a" : R2_a, "R1_u" : R1_u, "R2_u" : R2_u, "Ro_a" : Ro_a, "Ro_u" : Ro_u}

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
def pval1_discrete(edgelist, order, scanZ, statistic={"all"}, skew_corr=True, lower=None, upper=None):
    N = np.float64(order.size) # total number of nodes
    if lower is None:
        lower = np.int64(np.ceil(.05*N)-1)
    if upper is None:
        upper = np.int64(np.floor(.95*N)-1)
    
    if lower < 1 or lower >= upper:
        lower = np.int64(1)
    else:
        lower = np.int64(np.ceil(lower))
    if upper > (N-3) or upper <= lower:
        upper = np.int64(N-3)
    else:
        upper = np.int64(np.floor(upper))

    # storage container for the results
    output = {}
    t = np.arange(1, (N+1), dtype=np.float64)
    K = np.float64((np.unique(order)).size)
    edgenum = np.float64((edgelist.shape)[0])
    mk = (np.unique(order, return_counts=True))[1]
    node_deg_a = np.zeros(np.int64(K), dtype=np.float64)
    node_deg_u = np.zeros(np.int64(K), dtype=np.float64)
    quan = 0.0
    quan_u = 0.0
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        node_deg_a[e1] += 1
        node_deg_a[e2] += 1
        node_deg_u[e1] += mk[e2]
        node_deg_u[e2] += mk[e1]
        quan += 1/mk[e1]/mk[e2]
        quan_u += mk[e1]*mk[e2]
    temp3 = ((1/mk).sum())
    temp1 = ((node_deg_a/mk).sum())
    temp8 = ((node_deg_a**2/mk).sum())

    G = (((mk*(mk-1)).sum())/2)+quan_u
    node_deg_u += mk-1
    one_a = 2*(2*K-2*temp3+quan)
    two_a = 4*(N-2*K+2*edgenum+temp8/4-temp1+temp3)
    three_a = (N-K+edgenum)**2
    one_u = G
    two_u = ((mk*node_deg_u**2).sum())
    three_u = G**2

    # asymptotic p-value: NO SKEW CORRECTION
    # always return no skew correction
    output["no_skew"] = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        b_a = scanZ["original"]["Zo_a_max"]
        b_u = scanZ["original"]["Zo_u_max"]
        
        def integrandO_a(x):
            c1 = rho_one_discrete(N, x, one_a, two_a, three_a)
            return c1*Nu(b_a*np.sqrt(2*c1))

        def integrandO_u(x):
            c1 = rho_one_discrete(N, x, one_u, two_u, three_u)
            return c1*Nu(b_u*np.sqrt(2*c1))
        
        pval_ori_a = b_a*norm.pdf(b_a)*integrate.quad(integrandO_a, a=lower, b=upper, limit=3000)[0]
        pval_ori_u = b_u*norm.pdf(b_u)*integrate.quad(integrandO_u, a=lower, b=upper, limit=3000)[0]
        output["no_skew"]["ori_a"] = np.array([pval_ori_a, 1]).min()
        output["no_skew"]["ori_u"] = np.array([pval_ori_u, 1]).min()
            
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        b_a = scanZ["weighted"]["Zw_a_max"]
        b_u = scanZ["weighted"]["Zw_u_max"]

        def integrandW_a(x):
            c1 = rho_w(N, x)
            return c1*Nu(b_a*np.sqrt(2*c1))

        def integrandW_u(x):
            c1 = rho_w(N, x)
            return c1*Nu(b_u*np.sqrt(2*c1))

        pval_wei_a = b_a*norm.pdf(b_a)*integrate.quad(integrandW_a, a=lower, b=upper, limit=3000)[0]
        pval_wei_u = b_u*norm.pdf(b_u)*integrate.quad(integrandW_u, a=lower, b=upper, limit=3000)[0]
        output["no_skew"]["wei_a"] = np.array([pval_wei_a, 1]).min()
        output["no_skew"]["wei_u"] = np.array([pval_wei_u, 1]).min()
            
    if gu.anyin({"all", "max", "m"}, statistic):
        b_a = scanZ["max_type"]["M_a_max"]
        b_u = scanZ["max_type"]["M_u_max"]

        def integrandD_a(x):
            c1 = rho_d(N, x)
            return c1*Nu(b_a*np.sqrt(2*c1))

        def integrandD_u(x):
            c1 = rho_d(N, x)
            return c1*Nu(b_u*np.sqrt(2*c1))

        def integrandW_a(x):
            c1 = rho_w(N, x)
            return c1*Nu(b_a*np.sqrt(2*c1))

        def integrandW_u(x):
            c1 = rho_w(N, x)
            return c1*Nu(b_u*np.sqrt(2*c1))

        pval_a1 = 2*b_a*norm.pdf(b_a)*integrate.quad(integrandD_a, a=lower, b=upper, limit=3000)[0]
        pval_a2 = b_a*norm.pdf(b_a)*integrate.quad(integrandW_a, a=lower, b=upper, limit=3000)[0]
        pval_max_a = 1-(1-np.array([pval_a1, 1]).min())*(1-np.array([pval_a2, 1]).min())
        pval_u1 = 2*b_u*norm.pdf(b_u)*integrate.quad(integrandD_u, a=lower, b=upper, limit=3000)[0]
        pval_u2 = b_u*norm.pdf(b_u)*integrate.quad(integrandW_u, a=lower, b=upper, limit=3000)[0]
        pval_max_u = 1-(1-np.array([pval_u1, 1]).min())*(1-np.array([pval_u2, 1]).min())
        output["no_skew"]["max_a"] = pval_max_a
        output["no_skew"]["max_u"] = pval_max_u
        
    # asymptotic p-value: SKEW CORRECTION
    if skew_corr == True:
        output["skew"] = {}
        temp = skewcorr(edgelist, order)
        ER3w_a = temp["ER3w_a"]
        ER3w_u = temp["ER3w_u"]
        ER3d_a = temp["ER3d_a"]
        ER3d_u = temp["ER3d_u"]
        ER3_a = temp["ER3_a"]
        ER3_u = temp["ER3_u"]

        temp = getMeanVar_changepoint1_discrete(edgelist, order)
        muo_a = temp["muo_a"]
        muo_u = temp["muo_u"]
        varo_a = temp["varo_a"]
        varo_u = temp["varo_u"]
        mu1_a = temp["mu1_a"]
        mu2_a = temp["mu2_a"]
        var1_a = temp["var1_a"]
        var2_a = temp["var2_a"]
        var12_a = temp["var12_a"]
        mu1_u = temp["mu1_u"]
        mu2_u = temp["mu2_u"]
        var1_u = temp["var1_u"]
        var2_u = temp["var2_u"]
        var12_u = temp["var12_u"]

        p_hat = (t-1)/(N-2)
        q_hat = 1-p_hat

        # mu and variance of the weighted edge-count test (average method)
        muw_a = q_hat*mu1_a+p_hat*mu2_a 
        varw_a = q_hat**2*var1_a+p_hat**2*var2_a+2*q_hat*p_hat*var12_a
        muw_u = q_hat*mu1_u+p_hat*mu2_u 
        varw_u = q_hat**2*var1_u+p_hat**2*var2_u+2*q_hat*p_hat*var12_u

        # mu and variance of the difference of two with-in group edge-counts (average method)
        mud_a = mu1_a-mu2_a 
        vard_a = var1_a+var2_a-2*var12_a
        mud_u = mu1_u-mu2_u
        vard_u = var1_u+var2_u-2*var12_u

        # original case
        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            A = N-K+edgenum
            ER3o_a = A**3-3*A**2*muo_a+3*A*(varo_a+muo_a**2)-ER3_a
            ER3o_u = G**3-3*G**2*muo_u+3*G*(varo_u+muo_u**2)-ER3_u
            ro_a = np.nan_to_num((-ER3o_a+3*muo_a*varo_a+muo_a**3)/(varo_a**(3/2)), nan=0, posinf=0, neginf=0) # E(Zo_a^3)
            ro_u = np.nan_to_num((-ER3o_u+3*muo_u*varo_u+muo_u**3)/(varo_u**(3/2)), nan=0, posinf=0, neginf=0) # E(Zo_u^3)
            c1_o_a = np.nan_to_num(rho_one_discrete(N, t, one_a, two_a, three_a), nan=0, posinf=0, neginf=0)
            c1_o_u = np.nan_to_num(rho_one_discrete(N, t, one_u, two_u, three_u), nan=0, posinf=0, neginf=0)
            b_a = scanZ["original"]["Zo_a_max"]
            b_u = scanZ["original"]["Zo_u_max"]
            result_o_a = pval1_discrete_sub2(N, b_a, ro_a, c1_o_a, lower, upper)
            result_o_u = pval1_discrete_sub2(N, b_u, ro_u, c1_o_u, lower, upper)

            # average
            if result_o_a > 0:
                output["skew"]["ori_a"] = ((np.array([result_o_a, 1])).min())
            # union
            if result_o_u > 0:
                output["skew"]["ori_u"] = ((np.array([result_o_u, 1])).min())

        if gu.anyin({"all", "weighted", "wei", "w", "max", "m"}, statistic):
            rw_a = np.nan_to_num((ER3w_a-3*muw_a*varw_a-muw_a**3)/(varw_a**(3/2)), nan=0, posinf=0, neginf=0) # E(Zw_a^3)
            rw_u = np.nan_to_num((ER3w_u-3*muw_u*varw_u-muw_u**3)/(varw_u**(3/2)), nan=0, posinf=0, neginf=0) # E(Zw_u^3)
            c1_w = np.nan_to_num(rho_w(N, t), nan=0, posinf=0, neginf=0)

            # weighted case
            if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
                b_a = scanZ["weighted"]["Zw_a_max"]
                b_u = scanZ["weighted"]["Zw_u_max"]
                result_w_a = pval1_discrete_sub2(N, b_a, rw_a, c1_w, lower, upper)
                result_w_u = pval1_discrete_sub2(N, b_u, rw_u, c1_w, lower, upper)

                # average
                if result_w_a > 0:
                    output["skew"]["wei_a"] = ((np.array([result_w_a, 1])).min())
                # union
                if result_w_u > 0:
                    output["skew"]["wei_u"] = ((np.array([result_w_u, 1])).min())
            
            # max_type case
            if gu.anyin({"all", "max", "m"}, statistic):
                c1_d = np.nan_to_num(rho_d(N, t), nan=0, posinf=0, neginf=0)
                rd_a = np.nan_to_num((ER3d_a-3*mud_a*vard_a-mud_a**3)/(vard_a**(3/2)), nan=0, posinf=0, neginf=0)
                if rd_a[np.int64(N/2-1)] == 0:
                    rd_a[np.int64(N/2-1)] = rd_a[np.int64(N/2)]
                rd_u = np.nan_to_num((ER3d_u-3*mud_u*vard_u-mud_u**3)/(vard_u**(3/2)), nan=0, posinf=0, neginf=0)
                if rd_u[np.int64(N/2-1)] == 0:
                    rd_u[np.int64(N/2-1)] = rd_u[np.int64(N/2)]

                b_a = scanZ["max_type"]["M_a_max"]
                b_u = scanZ["max_type"]["M_u_max"]
                result_d_a = pval1_discrete_sub1(N, b_a, rd_a, c1_d, lower, upper)
                result_w_a = pval1_discrete_sub2(N, b_a, rw_a, c1_w, lower, upper)
                result_d_u = pval1_discrete_sub1(N, b_u, rd_u, c1_d, lower, upper)
                result_w_u = pval1_discrete_sub2(N, b_u, rw_u, c1_w, lower, upper)

                # average
                if result_d_a != 0 and result_w_a != 0:
                    output["skew"]["max_a"] = 1-(1-(np.array([result_d_a, 1])).min())*(1-(np.array([result_w_a, 1])).min())
                # union
                if result_d_u !=0 and result_w_u != 0:
                    output["skew"]["max_u"] = 1-(1-(np.array([result_d_u, 1])).min())*(1-(np.array([result_w_u, 1])).min())

    # generalized test p-value is the same for both skew correction and no skew correction
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        b_a = scanZ["generalized"]["S_a_max"]
        b_u = scanZ["generalized"]["S_u_max"]

        def integrandG_a(x, w):
            x1 = rho_d(N, x)
            x2 = rho_w(N, x)
            return 2*(x1*np.cos(w)**2+x2*np.sin(w)**2)*b_a*Nu(np.sqrt(2*b_a*(x1*np.cos(w)**2+x2*np.sin(w)**2)))/(2*np.pi)

        def integrandG_u(x, w):
            x1 = rho_d(N, x)
            x2 = rho_w(N, x)
            return 2*(x1*np.cos(w)**2+x2*np.sin(w)**2)*b_u*Nu(np.sqrt(2*b_u*(x1*np.cos(w)**2+x2*np.sin(w)**2)))/(2*np.pi)

        pval_gen_a = chi2.pdf(b_a, 2)*integrate.dblquad(integrandG_a, a=0, b=2*np.pi, gfun=lower, hfun=upper)[0]
        pval_gen_u = chi2.pdf(b_u, 2)*integrate.dblquad(integrandG_u, a=0, b=2*np.pi, gfun=lower, hfun=upper)[0]
        output["no_skew"]["gen_a"] = np.array([pval_gen_a, 1]).min()
        output["no_skew"]["gen_u"] = np.array([pval_gen_u, 1]).min()

        if skew_corr == True:
            output["skew"]["gen_a"] = np.array([pval_gen_a, 1]).min()
            output["skew"]["gen_u"] = np.array([pval_gen_u, 1]).min()
    
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
# support function (skewness correction) for approximated p-value in the single changepoint setting
# approximated p-value with extrapolation for max-count statistic
def pval1_discrete_sub1(N, b, r, x, lower, upper):
    theta_b = np.zeros(np.int64(N), dtype=np.float64)
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = np.nan_to_num(((np.sqrt((1+2*r*b)[pos])-1)/r[pos]), nan=0, posinf=0, neginf=0)
    ratio = np.exp((b-theta_b)**2/2+r*theta_b**3/6)/np.sqrt(1+r*theta_b)
    a = x*Nu(np.sqrt(2*b**2*x))*ratio
    
    nn_l = np.ceil(N/2)-(((np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)>0))[0]).size)
    nn_r = np.ceil(N/2)-(((np.where((1+2*r[np.int64(np.ceil(N/2-1)):np.int64(N)]*b)>0))[0]).size)
    if nn_l > .35*N:
        return 0
    if nn_l >= lower:
        neg = (np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)<=0))[0]
        dif = np.append(np.diff(neg), (N/2-nn_l))
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
    if nn_r >= (N-upper):
        neg = (np.where((1+2*r[np.int64(np.ceil(N/2-1)):np.int64(N)]*b)<=0))[0]
        id1 = ((np.append(np.int64(neg+np.ceil(N/2-1)-1), np.int64(np.ceil(N/2-1)-1))).min())
        id2 = id1-np.int64(np.ceil(.03*N))
        id3 = id2-np.int64(np.ceil(.09*N))
        inc = (ratio[id3]-ratio[id2])/(id3-id2)
        ratio[id2:np.int64(N)] = ratio[id2-1]+inc*(np.arange((id2+1), (N+1))-(id2+1))
        ratio[((np.asarray(ratio<0)).nonzero())[0]] = 0
        a[np.int64((N/2-1)):np.int64(N)] = (x*Nu(np.sqrt(2*b**2*x))*ratio)[np.int64((N/2-1)):np.int64(N)]
        
    neg2 = (np.where(a<0))[0]
    a[neg2] = 0
    
    def integrand_discrete1_sub1(s, a):
        return a[np.int64(s)]
        
    return 2*norm.pdf(b)*b*integrate.quad(integrand_discrete1_sub1, a=lower, b=upper, args=(a,), limit=3000)[0]

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
# approximated p-value (skewness correction) with extrapolation for weighted statistic
def pval1_discrete_sub2(N, b, r, x, lower, upper):
    theta_b = np.zeros(np.int64(N), dtype=np.float64)
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = ((np.sqrt((1+2*r*b)[pos])-1)/r[pos])
    ratio = np.exp((b-theta_b)**2/2+r*theta_b**3/6)/np.sqrt(1+r*theta_b)
    a = np.nan_to_num(x*Nu(np.sqrt(2*b**2*x))*ratio, nan=0, posinf=0, neginf=0)
    
    NN = np.int64((np.int64(N)-(pos.size)))
    if NN > .75*N:
        return 0
    if NN >= ((lower-1)+(N-upper)):
        neg = (np.where((1+2*r*b)<=0))[0]
        dif = neg[1:NN]-neg[0:(NN-1)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
        a[np.int64((N/2)):np.int64(N)] = a[np.int64((N/2)-1)::-1]
        a = np.nan_to_num(a, nan=0)
        neg2 = (np.where(a<0))[0]
        a[neg2] = 0

    def integrand_discrete1_sub2(s, a):
        return a[np.int64(s)]

    return norm.pdf(b)*b*integrate.quad(integrand_discrete1_sub2, a=lower, b=upper, args=(a,), limit=3000)[0]

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
# p-value from permutation for a single changepoint
def permpval1_discrete(edgelist, order, scanZ, statistic={"all"}, B=100, n0=None, n1=None):
    N = np.int64(order.size) # total number of nodes
    B = np.float64(B)
    # define default values for n0 and n1, which are functions of N
    if n0 is None:
        n0 = np.int64(np.ceil(.05*N)-1)
    if n1 is None:
        n1 = np.int64(np.floor(.95*N)-1)
    
    if n0 < 1 or n0 >= n1:
        n0 = np.int64(1)
    else:
        n0 = np.int64(np.ceil(n0))
    if n1 > (N-3) or n1 <= n0:
        n1 = np.int64(N-3)
    else:
        n1 = np.int64(np.floor(n1))

    Z_ori_a = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_ori_u = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_wei_a = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_wei_u = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_max_a = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_max_u = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_gen_a = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)
    Z_gen_u = np.zeros((np.int64(B), np.int64(N)), dtype=np.float64)

    order_shuffled = order.copy()
    for b in np.arange(np.int64(B), dtype=np.int64):
        np.random.shuffle(order_shuffled)
        gcpstar = changepoint1_discrete(edgelist, order_shuffled, statistic, n0, n1)
        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            Z_ori_a[b, :] = gcpstar["original"]["Zo_a"]
            Z_ori_u[b, :] = gcpstar["original"]["Zo_u"]
        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            Z_wei_a[b, :] = gcpstar["weighted"]["Zw_a"]
            Z_wei_u[b, :] = gcpstar["weighted"]["Zw_u"]
        if gu.anyin({"all", "max", "m"}, statistic):
            Z_max_a[b, :] = gcpstar["max_type"]["M_a"]
            Z_max_u[b, :] = gcpstar["max_type"]["M_u"]
        if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
            Z_gen_a[b, :] = gcpstar["generalized"]["S_a"]
            Z_gen_u[b, :] = gcpstar["generalized"]["S_u"]

    # container for the results
    output = {}
    p = 1-(np.arange(B, dtype=np.float64)/B)

    # pval: permutation p-value
    # curve: distribution of B max(Z(t))
    # maxZs: B max(Z(t)) after calculation by B permutation
    # Z: B Z(t) (B by N matrix)
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        maxZ_a = (Z_ori_a[:, n0:(n1+1)]).max(axis=1)
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = (Z_ori_u[:, n0:(n1+1)]).max(axis=1)
        maxZs_u = np.sort(maxZ_u)
        output["ori_a"] = {"pval" : ((maxZs_a >= scanZ["original"]["Zo_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Z_ori_a}
        output["ori_u"] = {"pval" : ((maxZs_u >= scanZ["original"]["Zo_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Z_ori_u}
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        maxZ_a = (Z_wei_a[:, n0:(n1+1)]).max(axis=1)
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = (Z_wei_u[:, n0:(n1+1)]).max(axis=1)
        maxZs_u = np.sort(maxZ_u)
        output["wei_a"] = {"pval" : ((maxZs_a >= scanZ["weighted"]["Zw_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Z_wei_a}
        output["wei_u"] = {"pval" : ((maxZs_u >= scanZ["weighted"]["Zw_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Z_wei_u}
    if gu.anyin({"all", "max", "m"}, statistic):
        maxZ_a = (Z_max_a[:, n0:(n1+1)]).max(axis=1)
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = (Z_max_u[:, n0:(n1+1)]).max(axis=1)
        maxZs_u = np.sort(maxZ_u)
        output["max_a"] = {"pval" : ((maxZs_a >= scanZ["max_type"]["M_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Z_max_a}
        output["max_u"] = {"pval" : ((maxZs_u >= scanZ["max_type"]["M_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Z_max_u}
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        maxZ_a = (Z_gen_a[:, n0:(n1+1)]).max(axis=1)
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = (Z_gen_u[:, n0:(n1+1)]).max(axis=1)
        maxZs_u = np.sort(maxZ_u)
        output["gen_a"] = {"pval" : ((maxZs_a >= scanZ["generalized"]["S_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Z_gen_a}
        output["gen_u"] = {"pval" : ((maxZs_u >= scanZ["generalized"]["S_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Z_gen_u}

    return output



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ DISCRETE CHANGE-INTERVAL DETECTION █                                                                              ▐
# ▌ Purpose : defines functions for detecting change-intervals in the the discrete setting                              ▐
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
def gchangeinterval_discrete(E, order, statistic={"all"}, l0=None, l1=None, pval_asym=True, skew_corr=True, pval_perm=0):
    N = np.float64(order.size) # the edge matrix should be NxN (square)
    edgelist = gu.edgematrix_to_edgelist(E) # convert the edge matrix to an edge list
    
    # define default values for l0 and l1, which are functions of N
    if l0 is None: l0 = np.int64(np.ceil(.05*N))
    if l1 is None: l1 = np.int64(np.floor(.95*N))
    if l0 < 1 or l0 >= l1: l0 = np.int64(1)
    if l1 > (N-2) or l1 <= l0: l1 = np.int64(N-2)
    l0 = np.int64(np.ceil(l0))
    l1 = np.int64(np.floor(l1))

    r1 = {}
    r1["scanZ"] = changeinterval1_discrete(edgelist, order, statistic, l0, l1)

    # compute asymptotic p-values
    if pval_asym == True:
        r1["pval_asym"] = pval2_discrete(edgelist, order, r1["scanZ"], statistic, skew_corr, l0, l1)

    # compute permutation p-values
    pval_perm = round(pval_perm)
    if pval_perm > 0:
        r1["pval_perm"] = permpval2_discrete(edgelist, order, r1["scanZ"], statistic, pval_perm, l0, l1)

    # store meta information about the function call
    r1["meta"] = {"type" : "interval", "start" : l0, "end" : l1}

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
# change interval
# edgelist is the edgenum by 2 matrix of edges that represents the similarity graph
# order is the index of the data (the order of the data)
def changeinterval1_discrete(edgelist, order, statistic={"all"}, l0=None, l1=None):
    np.seterr(divide="ignore", invalid="ignore")
    N = np.float64(order.size)
    
    # define default values for l0 and l1, which are functions of N
    if l0 is None: l0 = np.int64(np.ceil(.05*N))
    if l1 is None: l1 = np.int64(np.floor(.95*N))
    if l0 < 1 or l0 >= l1: l0 = np.int64(1)
    if l1 > (N-2) or l1 <= l0: l1 = np.int64(N-2)
    l0 = np.int64(np.ceil(l0))
    l1 = np.int64(np.floor(l1))

    temp = getMeanVar_changepoint1_discrete(edgelist, order) # use the same function in changepoint1_discrete()
    muo_a = temp["muo_a"]
    muo_u = temp["muo_u"]
    varo_a = temp["varo_a"]
    varo_u = temp["varo_u"]
    mu1_a = temp["mu1_a"]
    mu2_a = temp["mu2_a"]
    var1_a = temp["var1_a"]
    var2_a = temp["var2_a"]
    var12_a = temp["var12_a"]
    mu1_u = temp["mu1_u"]
    mu2_u = temp["mu2_u"]
    var1_u = temp["var1_u"]
    var2_u = temp["var2_u"]
    var12_u = temp["var12_u"]

    t = np.arange(1, (N+1), dtype=np.float64)
    p_hat = (t-1)/(N-2)
    q_hat = 1-p_hat
    
    # mu and variance of the weighted edge-count test (average method)
    muw_a = q_hat*mu1_a+p_hat*mu2_a 
    varw_a = q_hat**2*var1_a+p_hat**2*var2_a+2*q_hat*p_hat*var12_a
    
    # mu and variance of the difference of two with-in group edge-counts (average method)
    mud_a = mu1_a-mu2_a 
    vard_a = var1_a+var2_a-2*var12_a
    
    # mu and variance of the weighted edge-count test (union method)
    muw_u = q_hat*mu1_u+p_hat*mu2_u 
    varw_u = q_hat**2*var1_u+p_hat**2*var2_u+2*q_hat*p_hat*var12_u
    
    # mu and variance of the difference of two with-in group edge-counts (union method)
    mud_u = mu1_u-mu2_u 
    vard_u = var1_u+var2_u-2*var12_u

    temp = getR1R2_discrete2(edgelist, order)
    R1_a = temp["R1_a"]
    R2_a = temp["R2_a"]
    R1_u = temp["R1_u"]
    R2_u = temp["R2_u"]
    Ro_a = temp["Ro_a"]
    Ro_u = temp["Ro_u"]
    Rw_a = temp["Rw_a"]
    Rw_u = temp["Rw_u"]
    Rd_a = R1_a-R2_a	
    Rd_u = R1_u-R2_u

    dif = np.zeros((np.int64(N), np.int64(N)), dtype=np.int64)
    for i in np.arange(np.int64(N), dtype=np.int64):
        for j in np.arange(np.int64(N), dtype=np.int64):
            dif[i, j] = j-i
    difv = dif.flatten() # note: as.vector() in R flattens in F order, so the transpose is taken in R
    ids = (np.where(difv>0))[0]
    ids2 = (np.where((difv>=l0)&(difv<=l1)))[0]
    
    scanZ = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        Ro_av = Ro_a.flatten()
        Zv_a = np.zeros(np.int64(N*N), dtype=np.float64)
        Zv_a[ids] = -(Ro_av[ids]-muo_a[(difv[ids]-1)])/np.sqrt(varo_a[(difv[ids]-1)])
        Zo_a_max = (Zv_a[ids2]).max()
        tauhat_a0 = (np.where(Zv_a==Zo_a_max))[0]
        tauhat_a = (np.array(np.unravel_index(tauhat_a0, (np.int64(N), np.int64(N))))).squeeze()
        Ro_uv = Ro_u.flatten()
        Zv_u = np.zeros(np.int64(N*N), dtype=np.float64)
        Zv_u[ids] = -(Ro_uv[ids]-muo_u[(difv[ids]-1)])/np.sqrt(varo_u[(difv[ids]-1)])
        Zo_u_max = (Zv_u[ids2]).max()
        tauhat_u0 = (np.where(Zv_u==Zo_u_max))[0]
        tauhat_u = (np.array(np.unravel_index(tauhat_u0, (np.int64(N), np.int64(N))))).squeeze()
        scanZ["original"] = {"tauhat_a" : tauhat_a, "Zo_a_max" : Zo_a_max, "Zo_a" : Zv_a.reshape((np.int64(N), np.int64(N))), "Ro_a" : Ro_a,
                             "tauhat_u" : tauhat_u, "Zo_u_max" : Zo_u_max, "Zo_u" : Zv_u.reshape((np.int64(N), np.int64(N))), "Ro_u" : Ro_u}

    if gu.anyin({"all", "weighted", "wei", "w", "max", "m", "generalized", "gen", "g"}, statistic):
        Rw_av = Rw_a.flatten()
        Zwv_a = np.zeros(np.int64(N*N), dtype=np.float64)
        Zwv_a[ids] = (Rw_av[ids]-muw_a[(difv[ids]-1)])/np.sqrt(varw_a[(difv[ids]-1)])
        Rw_uv = Rw_u.flatten()
        Zwv_u = np.zeros(np.int64(N*N), dtype=np.float64)
        Zwv_u[ids] = (Rw_uv[ids]-muw_u[(difv[ids]-1)])/np.sqrt(varw_u[(difv[ids]-1)])

        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            Zw_a_max = (Zwv_a[ids2]).max()
            tauhat_a0 = (np.where(Zwv_a==Zw_a_max))[0]
            tauhat_a = (np.array(np.unravel_index(tauhat_a0, (np.int64(N), np.int64(N))))).squeeze()
            Zw_u_max = (Zwv_u[ids2]).max()
            tauhat_u0 = (np.where(Zwv_u==Zw_u_max))[0]
            tauhat_u = (np.array(np.unravel_index(tauhat_u0, (np.int64(N), np.int64(N))))).squeeze()
            scanZ["weighted"] = {"tauhat_a" : tauhat_a, "Zw_a_max" : Zw_a_max, "Zw_a" : Zwv_a.reshape((np.int64(N), np.int64(N))), "Rw_a" : Rw_a,
                                 "tauhat_u" : tauhat_u, "Zw_u_max" : Zw_u_max, "Zw_u" : Zwv_u.reshape((np.int64(N), np.int64(N))), "Rw_u" : Rw_u}

        if gu.anyin({"all", "max", "m", "generalized", "gen", "g"}, statistic):
            Rd_av = Rd_a.flatten()
            Zdv_a = np.zeros(np.int64(N*N), dtype=np.float64)
            Zdv_a[ids] = (Rd_av[ids]-mud_a[(difv[ids]-1)])/np.sqrt(vard_a[(difv[ids]-1)])
            Rd_uv = Rd_u.flatten()
            Zdv_u = np.zeros(np.int64(N*N), dtype=np.float64)
            Zdv_u[ids] = (Rd_uv[ids]-mud_u[(difv[ids]-1)])/np.sqrt(vard_u[(difv[ids]-1)])

            if gu.anyin({"all", "max", "m"}, statistic):
                Zwv_a = np.nan_to_num(Zwv_a, nan=0, posinf=0, neginf=0)
                M_a = np.zeros(np.int64(N*N), dtype=np.float64)
                M_a[ids] = (np.concatenate(((np.abs(Zdv_a[ids])).reshape((-1, 1)), (Zwv_a[ids]).reshape((-1, 1))), axis=1)).max(axis=1)
                M_a_max = (M_a[ids2]).max()
                tauhat_a0 = (np.where(M_a==M_a_max))[0]
                tauhat_a = (np.array(np.unravel_index(tauhat_a0, (np.int64(N), np.int64(N))))).squeeze()
                Zwv_u = np.nan_to_num(Zwv_u, nan=0, posinf=0, neginf=0)
                M_u = np.zeros(np.int64(N*N), np.float64)
                M_u[ids] = (np.concatenate(((np.abs(Zdv_u[ids])).reshape((-1, 1)), (Zwv_u[ids]).reshape((-1, 1))), axis=1)).max(axis=1)
                M_u_max = (M_u[ids2]).max()
                tauhat_u0 = (np.where(M_u==M_u_max))[0]
                tauhat_u = (np.array(np.unravel_index(tauhat_u0, (np.int64(N), np.int64(N))))).squeeze()
                scanZ["max_type"] = {"tauhat_a" : tauhat_a, "M_a_max" : M_a_max, "M_a" : M_a.reshape((np.int64(N), np.int64(N))),
                                     "tauhat_u" : tauhat_u, "M_u_max" : M_u_max, "M_u" : M_u.reshape((np.int64(N), np.int64(N)))}

            if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
                Zwv_a = np.nan_to_num(Zwv_a, nan=0, posinf=0, neginf=0)
                Sv_a = np.zeros(np.int64(N*N), dtype=np.float64)
                Sv_a[ids] = (Zwv_a[ids])**2+(Zdv_a[ids])**2
                S_a_max = (Sv_a[ids2]).max()
                tauhat_a0 = (np.where(Sv_a==S_a_max))[0]
                tauhat_a = (np.array(np.unravel_index(tauhat_a0, (np.int64(N), np.int64(N))))).squeeze()
                Zwv_u = np.nan_to_num(Zwv_u, nan=0, posinf=0, neginf=0)
                Sv_u = np.zeros(np.int64(N*N), dtype=np.float64)
                Sv_u[ids] = (Zwv_u[ids])**2+(Zdv_u[ids])**2
                S_u_max = (Sv_u[ids2]).max()
                tauhat_u0 = (np.where(Sv_u==S_u_max))[0]
                tauhat_u = (np.array(np.unravel_index(tauhat_u0, (np.int64(N), np.int64(N))))).squeeze()
                scanZ["generalized"] = {"tauhat_a" : tauhat_a, "S_a_max" : S_a_max, "S_a" : Sv_a.reshape((np.int64(N), np.int64(N))),
                                        "tauhat_u" : tauhat_u, "S_u_max" : S_u_max, "S_u" : Sv_u.reshape((np.int64(N), np.int64(N)))}

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
# supporting function for changeinterval1_discrete()
# give the test statistics such as R1_a(t1,t2), R2_u(t1,t2), Ro_a(t1,t2), Rw_a(t1,t2)
def getR1R2_discrete2(edgelist, order):
    N = np.float64(order.size)
    K = np.float64((np.unique(order)).size)
    edgenum = np.float64((edgelist.shape)[0])
    mk = (np.unique(order, return_counts=True))[1]
    temp1 = ((mk**2-mk).sum())/2
    temp2 = 0.0
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        temp2 += mk[e1]*mk[e2]
        
    R1_a = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    R2_a = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    R1_u = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    R2_u = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    Ro_a = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    Ro_u = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    Rw_a = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)
    Rw_u = np.zeros((np.int64(N), np.int64(N)), dtype=np.float64)

    for i in np.arange(np.int64((N-1)), dtype=np.int64):
        for j in np.arange((i+1), np.int64(N), dtype=np.int64):
            if j == (i+1):
                V = np.zeros((2, np.int64(K)), dtype=np.float64)
                V[1, ] = mk
                V[0, order[j]] = 1
                V[1, order[j]] -= 1
                So_a1 = ((V[0, ]*V[1, ]*2/mk).sum())
                So_u1 = ((V[0, ]*V[1, ]).sum())
                So_a2 = 0
                So_u2 = 0
                for k in np.arange(np.int64(edgenum), dtype=np.int64):
                    So_a2 += (V[0, edgelist[k, 0]]*V[1, edgelist[k, 1]]+V[0, edgelist[k, 1]]*V[1, edgelist[k, 0]])/mk[edgelist[k, 0]]/mk[edgelist[k, 1]]
                    So_u2 += (V[0, edgelist[k, 0]]*V[1, edgelist[k, 1]]+V[0, edgelist[k, 1]]*V[1, edgelist[k, 0]])
                R1_a[i, j] = 0
                Ro_a[i, j] = So_a1+So_a2
                R2_a[i, j] = N-K+edgenum-R1_a[i, j]-Ro_a[i, j]
                R1_u[i, j] = 0
                Ro_u[i, j] = So_u1+So_u2
                R2_u[i, j] = temp1+temp2-R1_u[i, j]-Ro_u[i, j]
            else:
                V[0, order[j]] += 1
                V[1, order[j]] -= 1
                probmat = (np.concatenate(np.where(edgelist==order[j]), dtype=np.int64)).reshape((-1, 2), order="F")
                probmat[:, 1] = 1-probmat[:, 1]
                R1_a[i, j] = R1_a[i, j-1]+2*(V[0, order[j]]-1)/mk[order[j]]+((V[0, edgelist[probmat[:, 0], probmat[:, 1]]]/mk[edgelist[probmat[:, 0], probmat[:, 1]]]).sum())/mk[order[j]]
                Ro_a[i, j] = Ro_a[i, j-1]+(2-(4*V[0, order[j]]-2)/mk[order[j]])+(((V[1, edgelist[probmat[:, 0], probmat[:, 1]]]-V[0, edgelist[probmat[:, 0], probmat[:, 1]]])/mk[edgelist[probmat[:, 0], probmat[:, 1]]]).sum())/mk[order[j]]
                R1_u[i, j] = R1_u[i, j-1]+(V[0, order[j]]-1)+((V[0, edgelist[probmat[:, 0], probmat[:, 1]]]).sum())
                Ro_u[i, j] = Ro_u[i, j-1]+(mk[order[j]]-2*V[0, order[j]]+1)+((V[1, edgelist[probmat[:, 0], probmat[:, 1]]]-V[0, edgelist[probmat[:, 0], probmat[:, 1]]]).sum())
            R2_a[i, j] = N-K+edgenum-R1_a[i, j]-Ro_a[i, j]
            R2_u[i, j] = temp1+temp2-R1_u[i, j]-Ro_u[i, j]
            Rw_a[i, j] = ((N-j+i-1)/(N-2))*R1_a[i, j]+(j-i-1)/(N-2)*R2_a[i, j]
            Rw_u[i, j] = ((N-j+i-1)/(N-2))*R1_u[i, j]+(j-i-1)/(N-2)*R2_u[i, j]

    return {"R1_a" : R1_a, "R2_a" : R2_a, "R1_u" : R1_u, "R2_u" : R2_u, "Ro_a" : Ro_a, "Ro_u" : Ro_u, "Rw_a" : Rw_a, "Rw_u" : Rw_u}

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
# p-value approximation (analytical p-value) for change interval
def pval2_discrete(edgelist, order, scanZ, statistic={"all"}, skew_corr=True, lower=None, upper=None):
    N = np.float64(order.size)
    
    # define default values for l0 and l1, which are functions of N
    if lower is None: lower = np.int64(np.ceil(.05*N))
    if upper is None: upper = np.int64(np.floor(.95*N))
    if lower < 1 or lower >= upper: lower = np.int64(1)
    if upper > (N-2) or upper <= lower: upper = np.int64(N-2)
    lower = np.int64(np.ceil(lower))
    upper = np.int64(np.floor(upper))

    output = {}
    t = np.arange(1, (N+1), dtype=np.float64)
    K = np.float64((np.unique(order)).size)
    edgenum = np.float64((edgelist.shape)[0])
    mk = (np.unique(order, return_counts=True))[1]
    
    node_deg_a = np.zeros(np.int64(K), dtype=np.float64)
    node_deg_u = np.zeros(np.int64(K), dtype=np.float64)
    quan = quan_u = 0.0
    for i in np.arange(np.int64(edgenum), dtype=np.int64):
        e1 = edgelist[i, 0]
        e2 = edgelist[i, 1]
        node_deg_a[e1] += 1
        node_deg_a[e2] += 1
        node_deg_u[e1] += mk[e2]
        node_deg_u[e2] += mk[e1]
        quan += 1/mk[e1]/mk[e2]
        quan_u += mk[e1]*mk[e2]
    temp3 = ((1/mk).sum())
    temp1 = ((node_deg_a/mk).sum())
    temp8 = ((node_deg_a**2/mk).sum())
    G = ((mk*(mk-1)).sum())/2+quan_u
    node_deg_u += mk-1
    one_a = 2*K-2*temp3+quan
    two_a = 4*(N-2*K+2*edgenum+temp8/4-temp1+temp3)
    three_a = (N-K+edgenum)**2
    one_u = G
    two_u = ((mk*node_deg_u**2).sum())
    three_u = G**2

    # asymptotic p-value: NO SKEW CORRECTION
    # always return the non skew corrected p-values
    output["no_skew"] = {}
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        b_a = scanZ["original"]["Zo_a_max"]
        b_u = scanZ["original"]["Zo_u_max"]

        def integrandO_a(x):
            c1 = rho_one_discrete(N, x, one_a, two_a, three_a)
            return (c1*Nu(b_a*np.sqrt(2*c1)))**2*(N-x)

        def integrandO_u(x):
            c1 = rho_one_discrete(N, x, one_u, two_u, three_u)
            return (c1*Nu(b_u*np.sqrt(2*c1)))**2*(N-x)

        pval_ori_a = b_a**3*norm.pdf(b_a)*integrate.quad(integrandO_a, a=lower, b=upper, limit=3000)[0]
        pval_ori_u = b_u**3*norm.pdf(b_u)*integrate.quad(integrandO_u, a=lower, b=upper, limit=3000)[0]
        output["no_skew"]["ori_a"] = ((np.array([pval_ori_a, 1])).min())
        output["no_skew"]["ori_u"] = ((np.array([pval_ori_u, 1])).min())

    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        b_a = scanZ["weighted"]["Zw_a_max"]
        b_u = scanZ["weighted"]["Zw_u_max"]

        def integrandW_a(x):
            c1 = rho_w(N, x)
            return (c1*Nu(b_a*np.sqrt(2*c1)))**2*(N-x)

        def integrandW_u(x):
            c1 = rho_w(N, x)
            return (c1*Nu(b_u*np.sqrt(2*c1)))**2*(N-x)

        pval_wei_a = b_a**3*norm.pdf(b_a)*integrate.quad(integrandW_a, a=lower, b=upper, limit=3000)[0]
        pval_wei_u = b_u**3*norm.pdf(b_u)*integrate.quad(integrandW_u, a=lower, b=upper, limit=3000)[0]
        output["no_skew"]["wei_a"] = ((np.array([pval_wei_a, 1])).min())
        output["no_skew"]["wei_u"] = ((np.array([pval_wei_u, 1])).min())

    if gu.anyin({"all", "max", "m"}, statistic):
        b_a = scanZ["max_type"]["M_a_max"]
        b_u = scanZ["max_type"]["M_u_max"]

        def integrandD_a(x):
            c1 = rho_d(N, x)
            return (c1*Nu(b_a*np.sqrt(2*c1)))**2*(N-x)

        def integrandD_u(x):
            c1 = rho_d(N, x)
            return (c1*Nu(b_u*np.sqrt(2*c1)))**2*(N-x)

        def integrandW_a(x):
            c1 = rho_w(N, x)
            return (c1*Nu(b_a*np.sqrt(2*c1)))**2*(N-x)

        def integrandW_u(x):
            c1 = rho_w(N, x)
            return (c1*Nu(b_u*np.sqrt(2*c1)))**2*(N-x)

        pval_a1 = 2*b_a**3*norm.pdf(b_a)*integrate.quad(integrandD_a, a=lower, b=upper, limit=3000)[0]
        pval_a2 = b_a**3*norm.pdf(b_a)*integrate.quad(integrandW_a, a=lower, b=upper, limit=3000)[0]
        pval_max_a = 1-(1-((np.array([pval_a1, 1])).min()))*(1-((np.array([pval_a2, 1])).min()))
        pval_u1 = 2*b_u**3*norm.pdf(b_u)*integrate.quad(integrandD_u, a=lower, b=upper, limit=3000)[0]
        pval_u2 = b_u**3*norm.pdf(b_u)*integrate.quad(integrandW_u, a=lower, b=upper, limit=3000)[0]
        pval_max_u = 1-(1-((np.array([pval_u1, 1])).min()))*(1-((np.array([pval_u2, 1])).min()))
        output["no_skew"]["max_a"] = pval_max_a
        output["no_skew"]["max_u"] = pval_max_u

    # asymptotic p-value: SKEW CORRECTION
    if skew_corr == True:
        output["skew"] = {}
        if gu.anyin({"all", "original", "ori", "o", "weighted", "wei", "w", "max", "m"}, statistic):
            temp = skewcorr(edgelist, order) # give statistics for the third moment of E(Rw_a^3)
            ER3w_a = temp["ER3w_a"]
            ER3w_u = temp["ER3w_u"]
            ER3d_a = temp["ER3d_a"]
            ER3d_u = temp["ER3d_u"]
            ER3_a = temp["ER3_a"]
            ER3_u = temp["ER3_u"]
    
            temp = getMeanVar_changepoint1_discrete(edgelist, order)
            muo_a = temp["muo_a"]
            muo_u = temp["muo_u"]
            varo_a = temp["varo_a"]
            varo_u = temp["varo_u"]
            mu1_a = temp["mu1_a"]
            mu2_a = temp["mu2_a"]
            var1_a = temp["var1_a"]
            var2_a = temp["var2_a"]
            var12_a = temp["var12_a"]
            mu1_u = temp["mu1_u"]
            mu2_u = temp["mu2_u"]
            var1_u = temp["var1_u"]
            var2_u = temp["var2_u"]
            var12_u = temp["var12_u"]
    
            p_hat = (t-1)/(N-2)
            q_hat = 1-p_hat
    
            # mu and variance of the weighted edge-count test (average method)
            muw_a = q_hat*mu1_a+p_hat*mu2_a 
            varw_a = q_hat**2*var1_a+p_hat**2*var2_a+2*q_hat*p_hat*var12_a
            muw_u = q_hat*mu1_u+p_hat*mu2_u 
            varw_u = q_hat**2*var1_u+p_hat**2*var2_u+2*q_hat*p_hat*var12_u
            
            # mu and variance of the difference of two with-in group edge-counts (average method)
            mud_a = mu1_a-mu2_a 
            vard_a = var1_a+var2_a-2*var12_a
            mud_u = mu1_u-mu2_u
            vard_u = var1_u+var2_u-2*var12_u
    
            if gu.anyin({"all", "original", "ori", "o"}, statistic):
                A = N-K+edgenum
                ER3o_a = A**3-3*A**2*muo_a+3*A*(varo_a+muo_a**2)-ER3_a
                ER3o_u = G**3-3*G**2*muo_u+3*G*(varo_u+muo_u**2)-ER3_u
                ro_a = np.nan_to_num(((-ER3o_a+3*muo_a*varo_a+muo_a**3)/(varo_a**(3/2))), nan=0, posinf=0, neginf=0) # E(Zo_a^3)
                ro_u = np.nan_to_num(((-ER3o_u+3*muo_u*varo_u+muo_u**3)/(varo_u**(3/2))), nan=0, posinf=0, neginf=0) # E(Zo_u^3)
                c1_o_a = np.nan_to_num((rho_one_discrete(N, t, one_a, two_a, three_a)), nan=0, posinf=0, neginf=0)
                c1_o_u = np.nan_to_num((rho_one_discrete(N, t, one_u, two_u, three_u)), nan=0, posinf=0, neginf=0)
                b_a = scanZ["original"]["Zo_a_max"]
                b_u = scanZ["original"]["Zo_u_max"]
                result_o_a = pval2_discrete_sub2(N, b_a, ro_a, c1_o_a, lower, upper)
                result_o_u = pval2_discrete_sub2(N, b_u, ro_u, c1_o_u, lower, upper)
    
                # average
                if result_o_a > 0:
                    output["skew"]["ori_a"] = ((np.array([result_o_a, 1])).min())
                    
                # union
                if result_o_u > 0:
                    output["skew"]["ori_u"] = ((np.array([result_o_u, 1])).min())
    
            if gu.anyin({"all", "weighted", "wei", "w", "max", "m"}, statistic):
                rw_a = np.nan_to_num((ER3w_a-3*muw_a*varw_a-muw_a**3)/(varw_a**(3/2)), nan=0, posinf=0, neginf=0) # E(Zw_a^3)
                rw_u = np.nan_to_num((ER3w_u-3*muw_u*varw_u-muw_u**3)/(varw_u**(3/2)), nan=0, posinf=0, neginf=0) # E(Zw_u^3)
                c1_w = np.nan_to_num(rho_w(N, t), nan=0, posinf=0, neginf=0)
    
                if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
                    b_a = scanZ["weighted"]["Zw_a_max"]
                    b_u = scanZ["weighted"]["Zw_u_max"]
                    result_w_a = pval2_discrete_sub2(N, b_a, rw_a, c1_w, lower, upper)
                    result_w_u = pval2_discrete_sub2(N, b_u, rw_u, c1_w, lower, upper)
    
                    # average 
                    if result_w_a > 0:
                        output["skew"]["wei_a"] = ((np.array([result_w_a, 1])).min())
    
                    # union
                    if result_w_u > 0:
                        output["skew"]["wei_u"] = ((np.array([result_w_u, 1])).min())
    
                if gu.anyin({"all", "max", "m"}, statistic):
                    c1_d = np.nan_to_num(rho_d(N, t), nan=0, posinf=0, neginf=0)
                    rd_a = np.nan_to_num((ER3d_a-3*mud_a*vard_a-mud_a**3)/(vard_a**(3/2)), nan=0, posinf=0, neginf=0)
                    if rd_a[np.int64(N/2-1)] == 0:
                        rd_a[np.int64(N/2-1)] = rd_a[np.int64(N/2)]
                    rd_u = np.nan_to_num((ER3d_u-3*mud_u*vard_u-mud_u**3)/(vard_u**(3/2)), nan=0, posinf=0, neginf=0)
                    if rd_u[np.int64(N/2-1)] == 0:
                        rd_u[np.int64(N/2-1)] = rd_u[np.int64(N/2)]
                    b_a = scanZ["max_type"]["M_a_max"]
                    b_u = scanZ["max_type"]["M_u_max"]
                    result_d_a = pval2_discrete_sub1(N, b_a, rd_a, c1_d, lower, upper)
                    result_w_a = pval2_discrete_sub2(N, b_a, rw_a, c1_w, lower, upper)
                    result_d_u = pval2_discrete_sub1(N, b_u, rd_u, c1_d, lower, upper)
                    result_w_u = pval2_discrete_sub2(N, b_u, rw_u, c1_w, lower, upper)
    
                    # average
                    if result_d_a != 0 and result_w_a != 0:
                        output["skew"]["max_a"] = 1-(1-(np.array([result_d_a, 1]).min()))*(1-(np.array([result_w_a, 1]).min()))
    
                    # union 
                    if result_d_u != 0 and result_w_u != 0:
                        output["skew"]["max_u"] = 1-(1-(np.array([result_d_u, 1]).min()))*(1-(np.array([result_w_u, 1]).min()))

    # generalized test p-value is the same for both skew correction and no skew correction
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        b_a = scanZ["generalized"]["S_a_max"]
        b_u = scanZ["generalized"]["S_u_max"]

        def integrandG_a(x, w):
            x1 = rho_d(N, x)
            x2 = rho_w(N, x)
            return (N-x)*(2*(x1*np.cos(w)**2+x2*np.sin(w)**2)*b_a*Nu(np.sqrt(2*b_a*(x1*np.cos(w)**2+x2*np.sin(w)**2))))**2/(2*np.pi)

        def integrandG_u(x, w):
            x1 = rho_d(N, x)
            x2 = rho_w(N, x)
            return (N-x)*(2*(x1*np.cos(w)**2+x2*np.sin(w)**2)*b_u*Nu(np.sqrt(2*b_u*(x1*np.cos(w)**2+x2*np.sin(w)**2))))**2/(2*np.pi)

        pval_gen_a = chi2.pdf(b_a, 2)*integrate.dblquad(integrandG_a, a=0, b=2*np.pi, gfun=lower, hfun=upper)[0]
        pval_gen_u = chi2.pdf(b_u, 2)*integrate.dblquad(integrandG_u, a=0, b=2*np.pi, gfun=lower, hfun=upper)[0]
        output["no_skew"]["gen_a"] = np.array([pval_gen_a, 1]).min()
        output["no_skew"]["gen_u"] = np.array([pval_gen_u, 1]).min()

        if skew_corr == True:
            output["skew"]["gen_a"] = np.array([pval_gen_a, 1]).min()
            output["skew"]["gen_u"] = np.array([pval_gen_u, 1]).min()
    
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
# support function (skewness correction) for asymptotic p-value in changeinterval setting
# approximated p-value with extrapolation for max-count statistic
def pval2_discrete_sub1(N, b, r, x, lower, upper):
    np.seterr(divide="ignore", invalid="ignore")
    theta_b = np.zeros(np.int64(N), dtype=np.float64)
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = np.nan_to_num(((np.sqrt((1+2*r*b)[pos])-1)/r[pos]), nan=0, posinf=0, neginf=0)
    ratio = np.exp((b-theta_b)**2/2+r*theta_b**3/6)/np.sqrt(1+r*theta_b)
    a = (b**2*x*Nu(np.sqrt(2*b**2*x)))**2*ratio

    nn_l = np.ceil(N/2)-(((np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)>0))[0]).size)
    nn_r = np.ceil(N/2)-(((np.where((1+2*r[np.int64(np.ceil(N/2-1)):np.int64(N)]*b)>0))[0]).size)
    if nn_l > .35*N:
        return 0
    if nn_l >= lower:
        neg = (np.where((1+2*r[:np.int64(np.ceil(N/2))]*b)<=0))[0]
        dif = np.append(np.diff(neg), (N/2-nn_l))
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
    if nn_r >= (N-upper):
        neg = (np.where((1+2*r[np.int64(np.ceil(N/2-1)):np.int64(N)]*b)<=0))[0]
        id1 = ((np.append(np.int64(neg+np.ceil(N/2-1)-1), np.int64(np.ceil(N/2-1)-1))).min())
        id2 = id1-np.int64(np.ceil(.03*N))
        id3 = id2-np.int64(np.ceil(.09*N))
        inc = (ratio[id3]-ratio[id2])/(id3-id2)
        ratio[id2:np.int64(N)] = ratio[(id2-1)]+inc*((np.arange((id2+1), (N+1)))-(id2+1))
        ratio[((np.asarray(ratio<0)).nonzero())[0]] = 0
        a[np.int64(N/2-1):np.int64(N)] = ((b**2*x*Nu(np.sqrt(2*b**2*x)))**2*ratio)[np.int64(N/2-1):np.int64(N)]

    neg2 = (np.where(a<0))[0]
    a[neg2] = 0
    
    def integrand_discrete2_sub1(s, a, N):
        return a[np.int64(s-1)]*(N-s)

    return 2*norm.pdf(b)/b*integrate.quad(integrand_discrete2_sub1, a=lower, b=upper, args=(a, N), limit=3000)[0]

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
# approximated p-value (skewness correction) with extrapolation for weighted statistic
def pval2_discrete_sub2(N, b, r, x, lower, upper):
    np.seterr(divide="ignore", invalid="ignore")
    theta_b = np.zeros(np.int64(N), dtype=np.float64)
    pos = (np.where((1+2*r*b)>0))[0]
    theta_b[pos] = (np.sqrt((1+2*r*b)[pos])-1)/r[pos]
    ratio = np.exp((b-theta_b)**2/2+r*theta_b**3/6)/np.sqrt(1+r*theta_b) # S(t) in integrand
    a = np.nan_to_num(((b**2*x*Nu(np.sqrt(2*b**2*x)))**2*ratio), nan=0, posinf=0, neginf=0)
    NN = np.int64(np.int64(N)-(pos.size))
    if NN >= ((lower-1)+(N-upper)):
        neg = (np.where((1+2*r*b)<=0))[0]
        dif = neg[1:NN]-neg[0:(NN-1)]
        id1 = dif.argmax()
        id2 = id1+np.int64(np.ceil(.03*N))
        id3 = id2+np.int64(np.ceil(.09*N))
        inc = (a[id3]-a[id2])/(id3-id2)
        a[id2::-1] = a[(id2+1)]-inc*(np.arange(1, (id2+2)))
        a[np.int64(N/2):np.int64(N)] = a[np.int64(N/2-1)::-1]
        a = np.nan_to_num(a, nan=0)
        neg2 = (np.where(a<0))[0]
        a[neg2] = 0

    def integrand_discrete2_sub2(s, a, N):
        return a[np.int64(s-1)]*(N-s)
        
    return norm.pdf(b)/b*integrate.quad(integrand_discrete2_sub2, a=lower, b=upper, args=(a, N), limit=3000)[0]

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
def permpval2_discrete(edgelist, order, scanZ, statistic={"all"}, B=100, l0=None, l1=None):
    N = np.float64(order.size)
    B = np.float64(B)
    # define default values for l0 and l1, which are functions of N
    if l0 is None: l0 = np.int64(np.ceil(.05*N))
    if l1 is None: l1 = np.int64(np.floor(.95*N))
    if l0 < 1 or l0 >= l1: l0 = np.int64(1)
    if l1 > (N-2) or l1 <= l0: l1 = np.int64(N-2)
    l0 = np.int64(np.ceil(l0))
    l1 = np.int64(np.floor(l1))

    Zmax_ori_a = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_ori_u = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_wei_a = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_wei_u = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_max_a = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_max_u = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_gen_a = np.zeros(np.int64(B), dtype=np.float64)
    Zmax_gen_u = np.zeros(np.int64(B), dtype=np.float64)

    order_shuffled = order.copy()
    for b in np.arange(np.int64(B), dtype=np.int64):
        np.random.shuffle(order_shuffled) # permute the data
        gcpstar = changeinterval1_discrete(edgelist, order_shuffled, statistic, l0, l1)
        if gu.anyin({"all", "original", "ori", "o"}, statistic):
            Zmax_ori_a[b] = gcpstar["original"]["Zo_a_max"]
            Zmax_ori_u[b] = gcpstar["original"]["Zo_u_max"]
        if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
            Zmax_wei_a[b] = gcpstar["weighted"]["Zw_a_max"]
            Zmax_wei_u[b] = gcpstar["weighted"]["Zw_u_max"]
        if gu.anyin({"all", "max", "m"}, statistic):
            Zmax_max_a[b] = gcpstar["max_type"]["M_a_max"]
            Zmax_max_u[b] = gcpstar["max_type"]["M_u_max"]
        if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
            Zmax_gen_a[b] = gcpstar["generalized"]["S_a_max"]
            Zmax_gen_u[b] = gcpstar["generalized"]["S_u_max"]

    output = {}
    p = 1-(np.arange(B, dtype=np.float64)/B)
    # pval: permuatation p-value
    # curve: distribution of B max(Z(t))
    # maxZs: B max(Z(t)) after calculation by B permutation
    # Z: B Z(t) (B by N matrix)
    if gu.anyin({"all", "original", "ori", "o"}, statistic):
        maxZ_a = Zmax_ori_a
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = Zmax_ori_u
        maxZs_u = np.sort(maxZ_u)
        output["ori_a"] = {"pval" : ((maxZs_a >= scanZ["original"]["Zo_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Zmax_ori_a}
        output["ori_u"] = {"pval" : ((maxZs_u >= scanZ["original"]["Zo_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Zmax_ori_u}
    if gu.anyin({"all", "weighted", "wei", "w"}, statistic):
        maxZ_a = Zmax_wei_a
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = Zmax_wei_u
        maxZs_u = np.sort(maxZ_u)
        output["wei_a"] = {"pval" : ((maxZs_a >= scanZ["weighted"]["Zw_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Zmax_wei_a}
        output["wei_u"] = {"pval" : ((maxZs_u >= scanZ["weighted"]["Zw_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Zmax_wei_u}
    if gu.anyin({"all", "max", "m"}, statistic):
        maxZ_a = Zmax_max_a
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = Zmax_max_u
        maxZs_u = np.sort(maxZ_u)
        output["max_a"] = {"pval" : ((maxZs_a >= scanZ["max_type"]["M_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Zmax_max_a}
        output["max_u"] = {"pval" : ((maxZs_u >= scanZ["max_type"]["M_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Zmax_max_u}
    if gu.anyin({"all", "generalized", "gen", "g"}, statistic):
        maxZ_a = Zmax_gen_a
        maxZs_a = np.sort(maxZ_a)
        maxZ_u = Zmax_gen_u
        maxZs_u = np.sort(maxZ_u)
        output["gen_a"] = {"pval" : ((maxZs_a >= scanZ["generalized"]["S_a_max"]).sum())/B, "curve" : np.concatenate((maxZs_a.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_a" : maxZs_a, "Z" : Zmax_gen_a}
        output["gen_u"] = {"pval" : ((maxZs_u >= scanZ["generalized"]["S_u_max"]).sum())/B, "curve" : np.concatenate((maxZs_u.reshape((-1, 1)), p.reshape((-1, 1))), axis=1), "maxZs_u" : maxZs_u, "Z" : Zmax_gen_u}

    return output



# ▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜
# ▌ █ PRINTING AND PLOTTING █                                                                                           ▐
# ▌ Purpose : prints and plots the results of gchangepoint_discrete or gchangeinterval_discrete for quick and easy      ▐
# ▌           reading and visualization                                                                                 ▐
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
# ║ DO_PRINT() METADATA                                                                                                 ║
# ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
# ║ Function    : do_print                                                                                              ║
# ║ Purpose     : summarizes and prints the results of gchangepoint_discrete or gchangeinterval_discrete in a succinct  ║
# ║               and readable format                                                                                   ║
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
def do_print(results, printEsts=True, printScans=True, printAsyms=True, printPerms=True, decimal_places=4):
    # print estimated change-points/change-intervals and max scan statistics
    if "scanZ" in results:
        # print information about the original test results
        if "original" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = results["meta"]["type"]
            if ("tauhat_a" not in results["scanZ"]["original"]) and ("tauhat_u" not in results["scanZ"]["original"]):
                print(f"NO ESTIMATED CHANGEPOINTS FOR ORIGINAL TEST")
                return
                
            print(f"\n\nORIGINAL TEST RESULTS")
            print(f"---------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat_a" in results["scanZ"]["original"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [average]: {results['scanZ']['original']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} time (t) [average]: {results['scanZ']['original']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["original"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [union]: {results['scanZ']['original']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} time (t) [union]: {results['scanZ']['original']['tauhat_u']+1}")
            if ("tauhat_a" in results["scanZ"]["original"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [average]: {results['scanZ']['original']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [average]: {results['scanZ']['original']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["original"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [union]: {results['scanZ']['original']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [union]: {results['scanZ']['original']['tauhat_u']+1}")

            # print the maximum scan statistic for the original test
            if ("Zo_a_max" in results["scanZ"]["original"]) and (printScans==True):
                print(f"Maximum original scan statistic [average]: {formatfloat(results['scanZ']['original']['Zo_a_max'], decimal_places)}")
            if ("Zo_u_max" in results["scanZ"]["original"]) and (printScans==True):
                print(f"Maximum original scan statistic [union]: {formatfloat(results['scanZ']['original']['Zo_u_max'], decimal_places)}")

            # print asymptotic p-value for the original test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "ori_a" in results["pval_asym"]["no_skew"]:
                        print(f"Original asymptotic p-value, no skew correction [average]: {formatfloat(results['pval_asym']['no_skew']['ori_a'], decimal_places)}")
                    if "ori_u" in results["pval_asym"]["no_skew"]:
                        print(f"Original asymptotic p-value, no skew correction [union]: {formatfloat(results['pval_asym']['no_skew']['ori_u'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "ori_a" in results["pval_asym"]["skew"]:
                        print(f"Original asymptotic p-value, with skew correction [average]: {formatfloat(results['pval_asym']['skew']['ori_a'], decimal_places)}")
                    if "ori_u" in results["pval_asym"]["skew"]:
                        print(f"Original asymptotic p-value, with skew correction [union]: {formatfloat(results['pval_asym']['skew']['ori_u'], decimal_places)}")

            # print permutation p-values for the original test
            if ("pval_perm" in results) and (printPerms==True):
                if "ori_a" in results["pval_perm"]:
                    print(f"Original permutation p-value [average]: {formatfloat(results['pval_perm']['ori_a']['pval'], decimal_places)}")
                if "ori_u" in results["pval_perm"]:
                    print(f"Original permutation p-value [union]: {formatfloat(results['pval_perm']['ori_u']['pval'], decimal_places)}")
                
                
        # print information about the weighted test results
        if "weighted" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = results["meta"]["type"]
            if ("tauhat_a" not in results["scanZ"]["weighted"]) and ("tauhat_u" not in results["scanZ"]["weighted"]):
                print(f"\n\nNO ESTIMATED CHANGEPOINT FOR WEIGHTED TEST")
                return

            print(f"\n\nWEIGHTED TEST RESULTS")
            print(f"---------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat_a" in results["scanZ"]["weighted"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [average]: {results['scanZ']['weighted']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} time (t) [average]: {results['scanZ']['weighted']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["weighted"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [union]: {results['scanZ']['weighted']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} time (t) [union]: {results['scanZ']['weighted']['tauhat_u']+1}")
            if ("tauhat_a" in results["scanZ"]["weighted"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [average]: {results['scanZ']['weighted']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [average]: {results['scanZ']['weighted']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["weighted"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [union]: {results['scanZ']['weighted']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [union]: {results['scanZ']['weighted']['tauhat_u']+1}")

            # print the maximum scan statistic for the weighted test
            if ("Zw_a_max" in results["scanZ"]["weighted"]) and (printScans==True):
                print(f"Maximum weighted scan statistic [average]: {formatfloat(results['scanZ']['weighted']['Zw_a_max'], decimal_places)}")
            if ("Zw_u_max" in results["scanZ"]["weighted"]) and (printScans==True):
                print(f"Maximum weighted scan statistic [union]: {formatfloat(results['scanZ']['weighted']['Zw_u_max'], decimal_places)}")

            # print asymptotic p-value for the weighted test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "wei_a" in results["pval_asym"]["no_skew"]:
                        print(f"Weighted asymptotic p-value, no skew correction [average]: {formatfloat(results['pval_asym']['no_skew']['wei_a'], decimal_places)}")
                    if "wei_u" in results["pval_asym"]["no_skew"]:
                        print(f"Weighted asymptotic p-value, no skew correction [union]: {formatfloat(results['pval_asym']['no_skew']['wei_u'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "wei_a" in results["pval_asym"]["skew"]:
                        print(f"Weighted asymptotic p-value, with skew correction [average]: {formatfloat(results['pval_asym']['skew']['wei_a'], decimal_places)}")
                    if "wei_u" in results["pval_asym"]["skew"]:
                        print(f"Weighted asymptotic p-value, with skew correction [union]: {formatfloat(results['pval_asym']['skew']['wei_u'], decimal_places)}")

            # print permutation p-values for the weighted test
            if ("pval_perm" in results) and (printPerms==True):
                if "wei_a" in results["pval_perm"]:
                    print(f"Weighted permutation p-value [average]: {formatfloat(results['pval_perm']['wei_a']['pval'], decimal_places)}")
                if "wei_u" in results["pval_perm"]:
                    print(f"Weighted permutation p-value [union]: {formatfloat(results['pval_perm']['wei_u']['pval'], decimal_places)}")

        # print information about the weighted test results
        if "max_type" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = results["meta"]["type"]
            if ("tauhat_a" not in results["scanZ"]["max_type"]) and ("tauhat_u" not in results["scanZ"]["max_type"]):
                print(f"\n\nNO ESTIMATED CHANGEPOINT FOR MAX-TYPE TEST")
                return             

            print(f"\n\nMAX-TYPE TEST RESULTS")
            print(f"---------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat_a" in results["scanZ"]["max_type"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [average]: {results['scanZ']['max_type']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} time (t) [average]: {results['scanZ']['max_type']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["max_type"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [union]: {results['scanZ']['max_type']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} time (t) [union]: {results['scanZ']['max_type']['tauhat_u']+1}")
            if ("tauhat_a" in results["scanZ"]["max_type"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [average]: {results['scanZ']['max_type']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [average]: {results['scanZ']['max_type']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["max_type"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [union]: {results['scanZ']['max_type']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [union]: {results['scanZ']['max_type']['tauhat_u']+1}")

            # print the maximum scan statistic for the max_type test
            if ("M_a_max" in results["scanZ"]["max_type"]) and (printScans==True):
                print(f"Maximum max-type scan statistic [average]: {formatfloat(results['scanZ']['max_type']['M_a_max'], decimal_places)}")
            if ("M_u_max" in results["scanZ"]["max_type"]) and (printScans==True):
                print(f"Maximum max-type scan statistic [union]: {formatfloat(results['scanZ']['max_type']['M_u_max'], decimal_places)}")

            # print asymptotic p-value for the max_type test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "max_a" in results["pval_asym"]["no_skew"]:
                        print(f"Max-type asymptotic p-value, no skew correction [average]: {formatfloat(results['pval_asym']['no_skew']['max_a'], decimal_places)}")
                    if "max_u" in results["pval_asym"]["no_skew"]:
                        print(f"Max-type asymptotic p-value, no skew correction [union]: {formatfloat(results['pval_asym']['no_skew']['max_u'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "max_a" in results["pval_asym"]["skew"]:
                        print(f"Max-type asymptotic p-value, with skew correction [average]: {formatfloat(results['pval_asym']['skew']['max_a'], decimal_places)}")
                    if "max_u" in results["pval_asym"]["skew"]:
                        print(f"Max-type asymptotic p-value, with skew correction [union]: {formatfloat(results['pval_asym']['skew']['max_u'], decimal_places)}")

            # print permutation p-values for the max_type test
            if ("pval_perm" in results) and (printPerms==True):
                if "max_a" in results["pval_perm"]:
                    print(f"Max-type permutation p-value [average]: {formatfloat(results['pval_perm']['max_a']['pval'], decimal_places)}")
                if "max_u" in results["pval_perm"]:
                    print(f"Max-type permutation p-value [union]: {formatfloat(results['pval_perm']['max_u']['pval'], decimal_places)}")

        # print information about the weighted test results
        if "generalized" in results["scanZ"]:
            # check if results is a change-point or change-interval
            # if tauhat is not in the dictonary return with message
            point_or_interval = results["meta"]["type"]
            if ("tauhat_a" not in results["scanZ"]["generalized"]) and ("tauhat_u" not in results["scanZ"]["generalized"]):
                print(f"\n\nNO ESTIMATED CHANGEPOINT FOR GENERALIZED TEST")
                return

            print(f"\n\nGENERALIZED TEST RESULTS")
            print(f"------------------------")
            # print estimated change-points/change-intervals if requested
            if ("tauhat_a" in results["scanZ"]["generalized"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [average]: {results['scanZ']['generalized']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} time (t) [average]: {results['scanZ']['generalized']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["generalized"]) and (printEsts==True) and (point_or_interval=="point"):
                print(f"Estimated change-{point_or_interval} index (i) [union]: {results['scanZ']['generalized']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} time (t) [union]: {results['scanZ']['generalized']['tauhat_u']+1}")
            if ("tauhat_a" in results["scanZ"]["generalized"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [average]: {results['scanZ']['generalized']['tauhat_a']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [average]: {results['scanZ']['generalized']['tauhat_a']+1}")
            if ("tauhat_u" in results["scanZ"]["generalized"]) and (printEsts==True) and (point_or_interval=="interval"):
                print(f"Estimated change-{point_or_interval} indices (i, j) [union]: {results['scanZ']['generalized']['tauhat_u']}")
                print(f"Estimated change-{point_or_interval} times (t1, t2) [union]: {results['scanZ']['generalized']['tauhat_u']+1}")

            # print the maximum scan statistic for the generalized test
            if ("S_a_max" in results["scanZ"]["generalized"]) and (printScans==True):
                print(f"Maximum generalized scan statistic [average]: {formatfloat(results['scanZ']['generalized']['S_a_max'], decimal_places)}")
            if ("S_u_max" in results["scanZ"]["generalized"]) and (printScans==True):
                print(f"Maximum generalized scan statistic [union]: {formatfloat(results['scanZ']['generalized']['S_u_max'], decimal_places)}")

            # print asymptotic p-value for the generalized test
            if ("pval_asym" in results) and (printAsyms==True):
                if ("no_skew" in results["pval_asym"]):
                    if "gen_a" in results["pval_asym"]["no_skew"]:
                        print(f"Generalized asymptotic p-value, no skew correction [average]: {formatfloat(results['pval_asym']['no_skew']['gen_a'], decimal_places)}")
                    if "gen_u" in results["pval_asym"]["no_skew"]:
                        print(f"Generalized asymptotic p-value, no skew correction [union]: {formatfloat(results['pval_asym']['no_skew']['gen_u'], decimal_places)}")
                        
                if ("skew" in results["pval_asym"]):
                    if "gen_a" in results["pval_asym"]["skew"]:
                        print(f"Generalized asymptotic p-value, with skew correction [average]: {formatfloat(results['pval_asym']['skew']['gen_a'], decimal_places)}")
                    if "gen_u" in results["pval_asym"]["skew"]:
                        print(f"Generalized asymptotic p-value, with skew correction [union]: {formatfloat(results['pval_asym']['skew']['gen_u'], decimal_places)}")

            # print permutation p-values for the generalized test
            if ("pval_perm" in results) and (printPerms==True):
                if "gen_a" in results["pval_perm"]:
                    print(f"Generalized permutation p-value [average]: {formatfloat(results['pval_perm']['gen_a']['pval'], decimal_places)}")
                if "gen_u" in results["pval_perm"]:
                    print(f"Generalized permutation p-value [union]: {formatfloat(results['pval_perm']['gen_u']['pval'], decimal_places)}")