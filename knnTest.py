import numpy as np
import scipy.stats as sc
import pandas as pd
from recommend import load_k

A = np.random.rand(5,3)
# A = np.array([[1,2,3],[4,5,6],[9,8,9]])
print "Matrix A:"
print A

def getCorrMatrix(A):
    """ Get the correlation matrix of A.
    @param A: matrix
    @return: matrix where element [i,j] is corr(A's ith row, A's jth row)
    """

    #TODO: Set corrs[i,i] to nonzero value? Dont' want yourself in your own top k NNs

    corrs = np.zeros([A.shape[0], A.shape[0]])

    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            if i == j:
                corrs[i, i] = -10
            else:
                corrs[i, j] = sc.pearsonr(A[i, :], A[j, :])[0]
                corrs[j, i] = sc.pearsonr(A[i, :], A[j, :])[0]

    return corrs

print
C = getCorrMatrix(A)
print "Corr Matrix C:"
print C

def getKNNDict(C, k):
    """ Get a dictionary containing k nearest neighbors of each row of a matrix, given its correlation matrix

    @param C: Correlation Matrix
    @return: dictionary mappings rows of A to their top k NNs, based on highest correlations
    """

    #TODO: Get bidirectional capability? I.e., corr = -1 is as good as corr = +1

    if k >= C.shape[0]:
        raise Exception("k is too big!")

    knnDict = {}

    for r in range(C.shape[0]):
        knnDict[r] = sorted([s for s in range(C.shape[1])], reverse = True, key = lambda x: C[r, x])[:k]

    return knnDict

print
K = getKNNDict(C, 2)
print K

def getSuggestionFromkNNs(A, k, r, c, weighted = False):
    """

    @param A: Matrix of user-ratings
    @param k: number of NNs to use
    @param r: what user to get a suggestion for
    @param c: Column to get a suggestion on
    @param weighted: True if weighting by correlation as well, False otherwise
    @return:
    """

    #TODO: bidirectional capability?

    C = getCorrMatrix(A)
    K = getKNNDict(C, k)

    knnRatings = [A[s,c] for s in K[r]]
    print
    print "Ratings of k NNs:"
    print knnRatings
    return np.mean(knnRatings)

print
print "Mean rating of c of k NNs:"
print getSuggestionFromkNNs(A, 2, 0, 2)


def corr_ignore_nan(a, b):
    """

    @param a: a list of numbers
    @param b: a list of numbers
    @return: Correlation between a and b, ignoring elements where a or b are nan
    """

    print "a b"
    print a
    print b

    a2 = []
    b2 = []

    for i in range(len(a)):
        if not (np.isnan(a[i]) or np.isnan(b[i])):
            a2.append(a[i])
            b2.append(b[i])

    print a2, b2


    if len(a2) < 1:
        return -100

    corr = sc.pearsonr(a2, b2)[0]
    if not np.isnan(corr):
        return corr
    return -1000

def getRatingMatrixFromDF(df):
    """

    @param df: pandas df from load_k
    @return:
    """

    users = set(df["userID"])
    songs = set(df["itemID"])
    # print len(users)
    # print users
    #
    print "# Songs:", len(songs)
    # print songs

    # user_song_dict = {}
    # for user in users:
    #     user_song_dict[user] = []
    # for r in range(df.shape[0]):
    #     user_song_dict[df.loc[r]["userID"]].append(df.loc[r]["itemID"])
    #
    # print user_song_dict

    user_songs_df = pd.DataFrame(index = users, columns = songs)
    # print user_songs_df
    # print user_songs_df.shape

    for r in range(df.shape[0]):
        row = df.loc[r]
        # user_songs_df[row["userID"]][row["itemID"]] = row["rating"]
        user_songs_df.at[row["userID"], row["itemID"]] = row["rating"]

    r1 = user_songs_df.iloc[0]
    r2 = user_songs_df.iloc[1]
    print "HERE"
    # print r1
    # print r2
    print "CORR:", corr_ignore_nan(list(r1), list(r2))



MAX_RATING = 5
DATA_POINTS_TO_READ = 100000
max_rating_filter=10

DATA = load_k(DATA_POINTS_TO_READ, max_rating_filter=max_rating_filter)

print
print getRatingMatrixFromDF(DATA)






