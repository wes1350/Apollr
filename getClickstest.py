import matplotlib.pyplot as plt
import pickle
from time import gmtime, strftime

click_coordinates = []

def on_click(event):
    if event.inaxes is not None:
        click_coordinates.append((event.xdata, event.ydata))
        print((event.xdata, event.ydata))
    else:
        print('Clicked ouside axes bounds but inside plot window')

fig, ax = plt.subplots()
fig.canvas.callbacks.connect('button_press_event', on_click)
plt.show()

with open('coords'+ str(len(click_coordinates)) +'.pickle', 'wb') as handle:
    pickle.dump(click_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('coords.pickle', 'rb') as handle:
#     new_coords = pickle.load(handle)
#
# print(new_coords)