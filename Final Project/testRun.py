# import hdf5_getters
# from MSongsDB-master/PythonSrc import hdf5_getters.py

import sys
sys.path.append("./MSongsDB-master/PythonSrc")
import hdf5_getters


h5 = hdf5_getters.open_h5_file_read("./MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5")
duration = hdf5_getters.get_duration(h5)
title = hdf5_getters.get_title(h5,1)
name = hdf5_getters.get_artist_name(h5)
# print(h5)
h5.close()

print(duration)
print(title)
print(name)