from recommend import *
from surprise import prediction_algorithms as pa

def perform_knn(algo, data_points, filename, max_rating_filter = 0):
    """
        uses knn algorithm to perform collaborative filterting

        :param data_points
        :return:
    """
    df = load_k(data_points, filename, data_points, max_rating_filter=max_rating_filter)
    numRows = df.shape[0]
    numCols = df.shape[1]
    train, test = split_for_eval(df, num_train=int(0.9*numRows))

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)

    algo = algo()

    train_set = data.build_full_trainset()
    algo.train(train_set)

    '''evaluating test set'''

    deviation = 0
    accuracy = 0
    acc_sq_err = 0
    for row in test.iterrows():
        user = row[1][2]
        song_id = row[1][0]
        rating = row[1][1]

        prediction = algo.predict(user, song_id, rating)
        ''' predicting'''
        # print('prediction',prediction[3])
        # print('rating', rating)

        """ prediction takes form: [userid, songid, actual_ration, estimated_rating, dictionary"""
        deviation = deviation + abs(rating - prediction[3])

        if rating == round(prediction[3]):
            accuracy += 1
        else:
            acc_sq_err += (rating -round(prediction[3]))**2

    error = 1.0*deviation/(len(test) - 1)
    accuracy = 1.0*accuracy/(len(test)-1)
    acc_rms = (1.0*acc_sq_err/(len(test)-1))**0.5
    return "ERROR, ACCURACY, ACCURACY_RMS:", error, accuracy, acc_rms

print('########################')
print('KNNBasic performance', perform_knn(pa.knns.KNNBasic, DATA_POINTS_TO_READ, './artistsc.txt', max_rating_filter=20))
print('########################')
print('KNNBaseline performance', perform_knn(pa.knns.KNNBaseline, DATA_POINTS_TO_READ, './artistsc.txt', max_rating_filter=20))
print('########################')
print('KNNWithMeans performance', perform_knn(pa.knns.KNNWithMeans, DATA_POINTS_TO_READ, './artistsc.txt', max_rating_filter=20))
print('########################')
print('KNNWithZScore performance', perform_knn(pa.knns.KNNWithZScore, DATA_POINTS_TO_READ, './artistsc.txt', max_rating_filter=20))
print('########################')
print('CoClustering performance', perform_knn(CoClustering, DATA_POINTS_TO_READ, './artistsc.txt', max_rating_filter=20))
