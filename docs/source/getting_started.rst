===============
Getting Started
===============

----------------
Dev Installation
----------------
You can install the development version (latest features but less stable) of `gDetect` directly from GitHub using pip:

.. code-block:: bash

    pip install git+https://github.com/atwold/gDetect.git

---------------------------------------------
Basic Usage for the Continuous Online Setting
---------------------------------------------
After installing the `gDetect` package, you can import the :mod:`continuousonline` module directly from :mod:`gdetect`. Examples within the continuous online change-point setting will use the following import convention.

.. code-block:: python

    from gdetect import continuousonline as con

The primary, user-facing function of `gDetect` for the continuous online setting is :func:`gstream() <gdetect.continuousonline.gstream>`. At minimum, :func:`gstream() <gdetect.continuousonline.gstream>` requires the user to specify its first four arguments. That is, the user needs to pass in ``distance_matrix`` a distance matrix calculated from a valid similarity metric, specify ``L`` the testing "window" size (the number of most recent observations that will be used to construct the KNN graph), declare ``N0`` the number of historical observations, and set ``k`` the number of nearest neighbors used when constructing the KNN. ``distance_matrix`` must be a :class:numpy.ndarray of shape (2, 2). The other three must be fixed integers. The following example demonstates a minimal working example by first simulating data, computing the distance matrix, running :func:`gstream() <gdetect.continuousonline.gstream>`, and finally printing the results with :func:`con_print() <gdetect.continuousonline.con_print>`.

.. code-block:: python

    # import the necessary packages and modules for this example
    from gdetect import continuousonline as con
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    from scipy.stats import norm
    
    # set seed for reproducibility
    np.random.seed(50)

    N = 120 # total number of observations
    tau = 78 # index of actual change
    d = 40 # dimension of observations

    # create data matrix
    # distribution 1 spans from observation indices 0 to (tau-1)
    # distribution 2 spans from observations indices tau to (N-1)
    data_cpon = np.concatenate(((norm.rvs(size=(tau*d))).reshape((tau, d)),
                                (norm.rvs(loc=3, size=((N-tau)*d))).reshape(((N-tau), d))),
                               axis=0)

    # create data distance matrix, modify diagonal
    D_cpon = euclidean_distances(data_cpon)
    np.fill_diagonal(D_cpon, np.max(D_cpon)+100)

    # N0 is number of historical observations, here 40 [indices 0 through 39], tests start at time N0+1 = 41 or     index N0+1-1 = N0 = 40
    # so tests start at time 41, index 40
    # tau=78, so a change-point occurs 38 observations (tau-N0) after tests begin
    # L is the number of observations the k-NN graph will be constructed from, it's the number of most recent   observations to be used in the tests
    # effectively, L is a window size for the testing process (best practice is to set L=N0 to use as many hist.    obs as possible)
    N0 = L = 40

    # online continuous change-point tests, k=5 set for k-NN
    res5 = con.gstream(D_cpon, L, N0, 5, skew_corr=False)
    print("**********************************")
    print("* RESULTS OF ONLINE CHANGE-POINT *")
    print("**********************************")
    con.con_print(res5, decimal_places=4)

.. code-block:: none

    **********************************
    * RESULTS OF ONLINE CHANGE-POINT *
    **********************************
    
    
    ORIGINAL TEST RESULTS
    ---------------------
    Stopping indices (n-N0):
    [43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66
     67 68 69 70 71 72]
    
    Scan statistics (ZL|y):
    [ 1.3216  1.6282  1.2691  0.6362  0.9281  1.0429  0.9745  1.2034  0.4778
      0.3705  0.1955  0.1599  0.5877  0.6292  0.2769  0.4606  0.592   1.217
      1.3127  1.4347  1.7642  2.053   1.9307  2.0421  1.4909  1.1301  1.2303
      2.2907  2.3356  2.1815  1.9936  1.7215  1.7899  1.6704  2.0364  2.046
      2.3572  2.0546  1.9819  2.3239  2.4675  2.7673  3.9804  5.0202  6.2854
      7.3069  7.9576  8.7239  9.4683 10.3716 10.8687 11.2441 11.639  11.871
     12.0883 12.228  12.3058 12.2952 12.2147 12.2042 12.0014 11.8483 11.5668
     11.3394 10.9835 10.5673  9.0059  8.2148  7.7423  6.9843  5.6771  5.2899
      4.4651  3.5372  2.4238  1.8311  0.1384 -0.6981  0.138   0.2597]
    
    Threshold (bZ): 4.0823
    
    
    
    WEIGHTED TEST RESULTS
    ---------------------
    Stopping indices (n-N0):
    [43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66
     67 68 69 70 71 72]
    
    Scan statistics (WL|y):
    [ 1.581   1.6945  1.4146  0.8823  1.0143  1.0774  0.786   0.8927  0.1989
      0.0067  0.0902 -0.0909  0.4314  0.4682  0.7728  0.779   1.3051  2.2504
      2.3352  2.6574  2.6014  2.7223  2.5592  2.6386  1.7267  1.2708  1.1939
      2.0939  2.1124  1.8799  1.6566  1.6624  1.7527  1.7812  2.2515  2.0964
      2.3076  1.997   1.9819  2.4145  2.6452  3.0711  3.727   5.3333  6.8381
      8.1337  9.1741 10.2541 11.2797 12.5864 12.4787 12.2689 12.3188 12.2936
     12.3561 12.3474 12.335  12.2952 12.2433 12.3388 12.2714 12.3291 12.2234
     12.4134 12.3301 12.6499 11.1012  9.7878  9.0417  7.9375  6.4015  5.7761
      4.1412  2.9625  2.0007  1.3259 -0.2708 -0.7563 -0.6526 -0.4372]
    
    Threshold (bW): 4.1289
    
    
    
    MAX-TYPE TEST RESULTS
    ---------------------
    Stopping indices (n-N0):
    [43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66
     67 68 69 70 71]
    
    Scan statistics (ML|y):
    [ 1.581   1.6945  1.4146  0.967   1.0143  1.0774  1.2572  1.4926  1.2015
      1.2716  1.2868  1.2766  1.3487  1.212   1.0837  0.9179  1.3051  2.2504
      2.3352  2.6574  2.6014  2.7223  2.5592  2.6386  1.7267  1.2708  1.5511
      2.0939  2.1124  1.8799  2.0125  1.7838  1.8068  1.9025  2.2515  2.0964
      2.3076  1.997   1.9819  2.4145  2.6452  3.0711  3.727   5.3333  6.8381
      8.1337  9.1741 10.2541 11.2797 12.5864 12.4787 12.2689 12.3188 12.2936
     12.3561 12.3474 12.335  12.2952 12.2433 12.3388 12.2714 12.3291 12.2234
     12.4134 12.3301 12.6499 11.1012  9.7878  9.0417  7.9375  6.4015  5.7761
      4.1412  2.9625  2.0007  1.7169  1.5475  1.3398  1.0934  1.1859]
    
    Threshold (bM): 4.3197
    
    
    
    GENERALIZED TEST RESULTS
    ------------------------
    Stopping indices (n-N0):
    [43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66
     67 68 69 70 71]
    
    Scan statistics (SL|y):
    [  2.7928   2.9231   2.0072   1.2128   1.0347   1.1865   1.6292   2.3828
       1.553    2.6975   2.4712   2.6734   1.9287   2.5442   3.0137   1.4493
       2.1919   7.76     7.4843   9.6716   7.9376   8.696    7.3858   7.7019
       3.469    1.8163   2.6109   5.9746   5.592    5.0479   4.9929   4.1898
       4.7186   4.2999   5.0887   4.4137   5.5571   4.4944   4.5329   6.5766
       7.7286  10.1082  16.5762  29.8746  48.7098  67.7793  84.917  105.433
     127.2641 158.4173 155.7187 150.5249 151.7536 151.1319 152.6728 152.4595
     152.1512 151.1708 149.8981 152.2457 150.5879 152.0057 149.4106 154.0925
     152.0321 160.0202 123.2442  95.8339  81.8877  63.5265  41.3874  33.7767
      20.1769  12.513    5.8928   5.8857   6.4654   6.9133   8.5431   7.2223]
    
    Threshold (bS): 23.1582

:func:`gstream() <gdetect.continuousonline.gstream>` returns the stopping indices, scan statistics, and thresholds for every requested method (default is all).