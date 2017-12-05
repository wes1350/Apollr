import numpy as np
import os, random

TRIPLETS_PATH = "train_triplets.txt"
PATH = os.path.join("data")
if not os.path.exists(PATH): os.mkdir(PATH)

TRAIN_PATH = os.path.join(PATH, "train.npz")

PERCENT_TRAIN = 0.6
PERCENT_VAL = 0.2
PERCENT_TEST = 0.2


class IdAssigner:

    def __init__(self):
        self.current_id = 0
        self.seen = {}

    def get_id(self, item):
        if item not in self.seen:
            self.seen[item] = self.current_id
            self.current_id += 1
        return self.seen[item]


def generate_mat_rep(k, min_song_count, filename=TRIPLETS_PATH):

    entries_seen = 0
    train_pairs = []
    val_threshold = PERCENT_TRAIN + PERCENT_VAL
    val_pairs = []
    test_pairs = []

    play_counts = []
    itemids = []

    song_assigner = IdAssigner()

    with open(filename, mode='r') as f:
        line = next(f).strip().split()
        current_user = [line[0], random.random()]
        itemids.append(song_assigner.get_id(line[1]))
        play_counts.append(float(line[2]))
        entries_seen += 1

        for entry in f:

            if entries_seen == k:
                break

            line = entry.strip().split()

            if current_user[0] == line[0]:
                play_counts.append(float(line[2]))
                itemids.append(song_assigner.get_id(line[1]))
            else:
                if max(play_counts) >= min_song_count:
                    if current_user[1] < PERCENT_TRAIN:
                        train_pairs.append(np.array(itemids))
                        train_pairs.append(np.array(play_counts))
                    elif PERCENT_TRAIN <= current_user[1] < val_threshold:
                        val_pairs.append(np.array(itemids))
                        val_pairs.append(np.array(play_counts))
                    elif val_threshold <= current_user[1] < 1.0:
                        test_pairs.append(np.array(itemids))
                        test_pairs.append(np.array(play_counts))

                current_user = [line[0], random.random()]
                itemids = []
                play_counts = []

            entries_seen += 1

        train_pairs.append(np.array(itemids))
        train_pairs.append(np.array(play_counts))
        train_pairs.append(np.array([song_assigner.current_id]))
        np.savez_compressed(TRAIN_PATH, train=train_pairs, val=val_pairs, test=test_pairs)
        return len(train_pairs)//2


if __name__ == "__main__":
    generate_mat_rep(15000000, 20)
   #mat = np.load(TRAIN_PATH)
