from surprise import Reader, Dataset, SVD
import pandas as pd
import math


MAX_RATING = 5


def convert_to_rating(data, max_rating=MAX_RATING):
    """
        Converts a list of play counts for a certain user to a list of ratings based on the
        given scale.

        :param data:
        :param max_rating:
        :return:
    """

    probability_bin = 1/max_rating
    max_value = max(data)
    ratings = [play_count/max_value for play_count in data]
    return [math.ceil(percentage/probability_bin) for percentage in ratings]


def load_k(k=1000, filename='train_triplets.txt', max_rating=MAX_RATING):
    """
        Reads the first k files from the data set and creates a panda data frame with
        columns itemID, useID and rating by transforming the play count to a range
        (1, max rating) for each user.

        :param k:
        :param filename:
        :return:
    """
    itemids = []
    userids = []
    ratings = []

    temp_ratings = []

    with open(filename, mode='r') as f:
        line = next(f).strip().split()
        current_user = line[0]

        userids.append(line[0])
        itemids.append(line[1])
        temp_ratings.append(float(line[2]))

        for i in range(k-1):
            line = next(f).strip().split()

            itemids.append(line[1])
            userids.append(line[0])

            if current_user == line[0]:
                temp_ratings.append(float(line[2]))
            else:
                ratings.extend(convert_to_rating(temp_ratings, max_rating))
                temp_ratings = [float(line[2])]
                current_user = line[0]

    ratings.extend(convert_to_rating(temp_ratings, max_rating))
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


def perform_CF(data_points=10000):
    """
        Gets data to perform filtering

        :param data_points:
        :return:
    """

    train, test = split_for_eval(load_k(data_points), num_train=int(0.9*data_points))

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)

    algo = SVD()
    train_set = data.build_full_trainset()
    algo.train(train_set)

    """ Row takes form (index (song_id, rating, user_id) )"""

    deviation = 0
    accuracy = 0
    for row in test.iterrows():

        user = row[1][2]
        song_id = row[1][0]
        rating = row[1][1]

        prediction = algo.predict(user, song_id, rating)
        """ prediction takes form: [userid, songid, actual_ration, estimated_rating, dictionary"""
        deviation = deviation + abs(rating - prediction[3])

        if rating == round(prediction[3]):
            accuracy += 1

    error = deviation/(len(test) - 1)
    accuracy = accuracy/(len(test)-1)
    return error, accuracy


print(perform_CF(100000))
