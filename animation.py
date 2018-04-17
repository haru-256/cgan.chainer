import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import pathlib

fig = plt.figure(figsize=(5, 5))

path = pathlib.Path("result/preview")

ims = []

for epoch in range(1, 101):
    img = cv2.imread(str(path / "image_{}epoch.png".format(epoch)), 0)
    frame = plt.imshow(img, cmap=plt.cm.gray)
    ims.append([frame])
ani = animation.ArtistAnimation(fig, ims, interval=300)
# ani.save('anim.mp4', writer="ffmpeg")
plt.axis("off")
ani.save('anim.gif', writer="imagemagick")
plt.show()
