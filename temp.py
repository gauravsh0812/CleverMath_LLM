from PIL import Image
import os

c=0
for i in os.listdir("data/images"):
    img = Image.open(f"data/images/{i}")
    w,h = img.size
    if w!=480 or h!=320:
        print(i, [w,h])
        c+=1
print(c)