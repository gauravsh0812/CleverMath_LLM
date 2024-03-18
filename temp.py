from PIL import Image

for i in range(100):
    img = Image.open(f"data/images/{i}.png")
    w,h = img.size
    if w!=480 or h!=320:
        print([w,h])