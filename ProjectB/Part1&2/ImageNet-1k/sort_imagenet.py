import os
import shutil
import sys

if sys.argv[2] == 'train':
    class_idx = 2
    base_dir = os.path.join(sys.argv[1], 'train')
elif sys.argv[2] == 'val':
    class_idx = 3
    base_dir = os.path.join(sys.argv[1], 'val')
else:
    raise

for image in os.listdir(base_dir):
    class_name = image.split('_')[class_idx][0:-5]
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        os.mkdir(class_dir)
    shutil.move(os.path.join(base_dir, image), class_dir)
