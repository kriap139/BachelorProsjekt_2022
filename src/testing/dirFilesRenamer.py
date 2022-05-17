import os

directory = "/home/kripi/Downloads/lablingggg/lablingggg"

for filename in os.listdir(directory):
    fn, ext = os.path.splitext(filename)
    newName = f"{fn}ali{ext}"

    os.rename(f"{os.path.join(directory, filename)}", f"{os.path.join(directory, newName)}")
