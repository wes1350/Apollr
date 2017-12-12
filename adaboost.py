from recommend import *
from knn import *

# perform_SVD(DATA_POINTS_TO_READ, './train_triplets.txt', max_rating_filter=20)
# perform_SVDpp(DATA_POINTS_TO_READ, './train_triplets.txt', max_rating_filter=20)
# perform_knn_basic(10000, './artistsc.txt', max_rating_filter=20)
# perform_knn_baseline(10000, './artistsc.txt', max_rating_filter=20)
# perform_knn_with_means(10000, './artistsc.txt', max_rating_filter=20)
# perform_knn_with_zscore(10000, './artistsc.txt', max_rating_filter=20)

models = [
perform_SVD(DATA_POINTS_TO_READ, './train_triplets.txt', max_rating_filter=20),
# perform_SVDpp(DATA_POINTS_TO_READ, './train_triplets.txt', max_rating_filter=20),
perform_knn_basic(10000, './artistsc.txt', max_rating_filter=20),
perform_knn_baseline(10000, './artistsc.txt', max_rating_filter=20),
perform_knn_with_means(10000, './artistsc.txt', max_rating_filter=20),
perform_knn_with_zscore(10000, './artistsc.txt', max_rating_filter=20)
]

class AdaBoost(object):
    def __init__(self, models):
        self.models = models
        self.weights = [1/len(models) for len(models)]
        self.error = [0 for len(models)]
        self.votes = [0 for len(models)]

    def train(self, samples, iterations):
        for i in range(iterations)
