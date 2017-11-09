from surprise import Reader, Dataset

with open('tp1000.txt') as f:
    lines = f.read().splitlines()

reader = Reader(line_format='user song rating', sep='\t')

data = Dataset.load_builtin('tp1000.txt', reader=reader)
