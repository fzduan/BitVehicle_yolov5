import os

exp9_dir = "runs/train/exp9"
for root, dirs, files in os.walk(exp9_dir):
    for file in files:
        print(os.path.join(root, file))
