from surprise import Reader, Dataset, SVD
import pandas as pd
import math


MAX_RATING = 5
DATA_POINTS_TO_READ = 100000


def convert_to_rating(data, max_rating=MAX_RATING, max_rating_filter = 0):
    """
        Converts a list of play counts for a certain user to a list of ratings based on the
        given scale.

        :param data:
        :param max_rating:
        :param max_rating_filter:
        :return:
    """

    probability_bin = 1.0/max_rating
    max_value = max(data)
    if max_value <= max_rating_filter:
        return None
    ratings = [1.0*play_count/max_value for play_count in data]
    return [math.ceil(1.0*percentage/probability_bin) for percentage in ratings]


def load_k(k, filename='../867FPData/train_triplets.txt', max_rating=MAX_RATING, max_rating_filter = 0):
    """
        Reads the first k files from the data set and creates a panda data frame with
        columns itemID, useID and rating by transforming the play count to a range
        (1, max rating) for each user.

        :param k:
        :param filename:
        :return:
    """
    ct = 0
    itemids = []
    userids = []
    ratings = []

    temp_ratings = []
    temp_itemids = []
    temp_userids = []

    with open(filename, mode='r') as f:
        line = next(f).strip().split()
        current_user = line[0]

        # userids.append(line[0])
        # itemids.append(line[1])
        temp_userids.append(line[0])
        temp_itemids.append(line[1])
        temp_ratings.append(float(line[2]))

        for i in range(k-1):
            line = next(f).strip().split()

            # itemids.append(line[1])
            # userids.append(line[0])

            if current_user == line[0]:
                temp_ratings.append(float(line[2]))
                temp_userids.append(line[0])
                temp_itemids.append(line[1])
            else:
                converted_rating = convert_to_rating(temp_ratings, max_rating, max_rating_filter)
                if converted_rating is not None:
                    ct +=1
                    print(ct)
                    ratings.extend(converted_rating)
                    userids.extend(temp_userids)
                    itemids.extend(temp_itemids)
                temp_ratings = [float(line[2])]
                temp_itemids = [line[1]]
                temp_userids = [line[0]]
                current_user = line[0]

    converted_rating = convert_to_rating(temp_ratings, max_rating, max_rating_filter)
    if converted_rating is not None:
        ratings.extend(converted_rating)
        userids.extend(temp_userids)
        itemids.extend(temp_itemids)

    ratings_dict = {'itemID': itemids, 'userID': userids, 'rating': ratings}
    df = pd.DataFrame(ratings_dict)
    return df


def split_for_eval(data_frame, num_train):
    """
        Removes num_test_points rows from the data frame to use as test data.

        :param data_frame:
        :param num_test_points:
        :return:
    """

    assert num_train < len(data_frame)

    train = data_frame.sample(n=num_train)
    test = data_frame.drop(train.index)

    return train, test


def perform_CF(data_points, max_rating_filter = 0):
    """
        Gets data to perform filtering

        :param data_points:
        :return:
    """
    df = load_k(data_points, max_rating_filter=max_rating_filter)
    numRows = df.shape[0]
    print("NUM ROWS:", numRows)
    train, test = split_for_eval(df, num_train=int(0.9*numRows))

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)

    algo = SVD()
    train_set = data.build_full_trainset()
    algo.train(train_set)

    """ Row takes form (index (song_id, rating, user_id) )"""

    deviation = 0
    accuracy = 0
    acc_sq_err = 0
    for row in test.iterrows():

        user = row[1][2]
        song_id = row[1][0]
        rating = row[1][1]

        prediction = algo.predict(user, song_id, rating)
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


print(perform_CF(DATA_POINTS_TO_READ, max_rating_filter=20))
# print(convert_to_rating([i for i in range(1,13)], 5))
