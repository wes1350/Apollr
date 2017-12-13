import numpy as np
import os, random, math
import statistics as st

TRIPLETS_PATH = "artistsc.txt"
PATH = os.path.join("data")
if not os.path.exists(PATH): os.mkdir(PATH)

TRAIN_PATH = os.path.join(PATH, "gaussian_ratings.npz")

PERCENT_TRAIN = 0.6
PERCENT_VAL = 0.2
PERCENT_TEST = 0.2
MAX_RATING = 5


class IdAssigner:

    def __init__(self):
        self.current_id = 1
        self.min_id = 1
        self.seen = {}
        self.temp = {}

    def get_id(self, item):
        if item not in self.seen:
            self.temp[item] = self.current_id
            self.current_id += 1
            
        return self.seen[item] if item in self.seen else self.temp[item]
    
    def accept(self):
        self.min_id = self.current_id
        self.seen.update(self.temp)
        self.temp = {}
    
    def reject(self):
        self.current_id = self.min_id
        self.temp = {}

def max_convert(data, max_rating=MAX_RATING):

    probability_bin = 1.0/max_rating
    max_value = max(data)
    ratings = [1.0*play_count/max_value for play_count in data]
    return [math.ceil(1.0*percentage/probability_bin) for percentage in ratings]

def gaussian_conversion(data):
    mean = st.mean(data)
    dev = st.stdev(data)
    ratings = [dev]
    
    for count in data:
        if dev == 0:
            ratings.append(3.0)
        else:
            ratings.append(min(5., max(1., 3. + (count-mean)/dev )))
    
    return ratings
    
def generate_mat_rep(num_users, min_song_count, filename=TRIPLETS_PATH):

    entries_seen = 0
    train_pairs = []
    val_threshold = PERCENT_TRAIN + PERCENT_VAL
    val_pairs = []
    test_pairs = []

    play_counts = []
    itemids = [0]
    convert_to_rating = gaussian_conversion
    song_assigner = IdAssigner()

    with open(filename, mode='r') as f:
        line = next(f).strip().split('\t')
        current_user = [line[0], random.random()]
        itemids.append(song_assigner.get_id(line[1]))
        play_counts.append(float(line[2]))

        for entry in f:

            if entries_seen == num_users:
                break

            line = entry.strip().split('\t')

            if current_user[0] == line[0]:
                play_counts.append(float(line[2]))
                itemids.append(song_assigner.get_id(line[1]))
            else:
                if max(play_counts) >= min_song_count and len(play_counts) > 1:
                    entries_seen += 1
                    song_assigner.accept()
                    if current_user[1] < PERCENT_TRAIN:
                        train_pairs.append(np.array(itemids))
                        train_pairs.append(np.array(convert_to_rating(play_counts)))
                    elif PERCENT_TRAIN <= current_user[1] < val_threshold:
                        val_pairs.append(np.array(itemids))
                        val_pairs.append(convert_to_rating(play_counts))
                    elif val_threshold <= current_user[1] <= 1.0:
                        test_pairs.append(np.array(itemids))
                        test_pairs.append(convert_to_rating(play_counts))
                else:
                    song_assigner.reject()

                current_user = [line[0], random.random()]
                itemids = [0, song_assigner.get_id(line[1])]
                play_counts = [float(line[2])]
                
                

        train_pairs.append(np.array([song_assigner.current_id]))
        np.savez_compressed(TRAIN_PATH, train=train_pairs, val=val_pairs, test=test_pairs)
        return len(train_pairs)//2, len(val_pairs)//2, len(test_pairs)//2, song_assigner.current_id


if __name__ == "__main__":
    print(generate_mat_rep(29000000, 10))
   #mat = np.load(TRAIN_PATH)
