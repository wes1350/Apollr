import numpy as np


class IdAssigner:

    def __init__(self):
        self.current_id = 0
        self.seen = {}

    def get_id(self, item):
        if item not in self.seen:
            self.seen[item] = self.current_id
            self.current_id += 1
        return self.seen[item]


def generate_mat_rep(k, filename='train_triplets.txt'):
    """
        Create a representation of a sparse matrix for the song data

        :param k:
        :param filename:
        :return:
    """

    entries_seen = 0
    id_playcount_pairs = []
    play_counts = []
    itemids = []
    song_assigner = IdAssigner()

    with open(filename, mode='r') as f:
        line = next(f).strip().split()
        current_user = line[0]
        itemids.append(song_assigner.get_id(line[1]))
        play_counts.append(float(line[2]))
        entries_seen += 1

        for entry in f:

            if entries_seen == k:
                break

            line = entry.strip().split()

            if current_user == line[0]:
                play_counts.append(float(line[2]))
                itemids.append(song_assigner.get_id(line[1]))
            else:
                id_playcount_pairs.append(np.array(itemids))
                id_playcount_pairs.append(np.array(play_counts))
                current_user = line[0]
                itemids = []
                play_counts = []

            entries_seen += 1

        id_playcount_pairs.append(np.array(itemids))
        id_playcount_pairs.append(np.array(play_counts))
        id_playcount_pairs.append(np.array([song_assigner.current_id]))
        id_playcount_pairs = np.array(id_playcount_pairs)
        np.save("id_playcount_pairs", id_playcount_pairs)
        return


if __name__ == "__main__":
    generate_mat_rep(50000000)
    mat = np.load("id_playcount_pairs.npy")
