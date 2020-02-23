import os

with open("meta.csv", "w") as file:
    tracks = os.listdir("tracks")
    labels = os.listdir("masks")
    for i, track in enumerate(tracks):
        file.write("..\\dataset\\tracks\\" + track + "," + "..\\dataset\\masks\\" + labels[i] + '\n')
