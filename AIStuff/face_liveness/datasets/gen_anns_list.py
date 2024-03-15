import os
from sklearn.model_selection import train_test_split

image_folder = "./motion_analysis/video_data"
abs_path = os.path.dirname(os.path.abspath(image_folder))

# cleans .txt files
open(os.path.join(abs_path, 'classInd.txt'), 'w').write("")
open(os.path.join(abs_path, 'trainlist01.txt'), 'w').write("")
open(os.path.join(abs_path, 'testlist01.txt'), 'w').write("")
open(os.path.join(abs_path, 'trainval.txt'), 'w').write("")

# updates labels.txt
labels = os.listdir(os.path.join(abs_path, "video_data"))
for i, label in enumerate(labels):
    with open(os.path.join(abs_path, 'classInd.txt'), 'a') as f:
        f.write(str(i) + " " + label)
        f.write('\n')

# loading mapping...
dict_labels = {}
a = open(os.path.join(abs_path, 'classInd.txt'), 'r').read()
c = a.split('\n')
for i in c[:len(c) - 1]:
    dict_labels.update({i.split(' ')[1]: i.split(' ')[0]})

# generating trainval.txt
for i, label in enumerate(labels):
    vid_names = os.listdir(os.path.join(abs_path, "video_data", label))
    for video_name in vid_names:
        with open(os.path.join(abs_path, 'trainval.txt'), 'a') as f:
            f.write(os.path.join(label, video_name) + " " + dict_labels[label])
            f.write('\n')

X = []
y = []
for data in open(os.path.join(abs_path, 'trainval.txt'), 'r').read().split("\n"):
    data_path = data.split()
    if data_path:
        img_path = data_path[0]
        label = data_path[1]
        X.append(img_path)
        y.append(label)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

real_train_path, real_train_label = [], []
fake_train_path, fake_train_label = [], []
for index, label in enumerate(y_train):
    if label == "0":
        real_train_path.append(X_train[index])
        real_train_label.append(label)
    else:
        fake_train_path.append(X_train[index])
        fake_train_label.append(label)

if len(real_train_path) < len(fake_train_path):
    n = len(fake_train_path) // len(real_train_path)
    d = len(fake_train_path) % len(real_train_path)
    real_train_path = n * real_train_path + real_train_path[:d]
    real_train_label = n * real_train_label + real_train_label[:d]
else:
    n = len(real_train_path) // len(fake_train_path)
    d = len(real_train_path) % len(fake_train_path)
    fake_train_path = n * fake_train_path + fake_train_path[:d]
    fake_train_label = n * fake_train_label + fake_train_label[:d]


X_train = real_train_path + fake_train_path
y_train = real_train_label + fake_train_label

# generating train.txt
for i, j in zip(X_train, y_train):
    with open(os.path.join(abs_path, 'trainlist01.txt'), 'a') as f:
        f.write(i + " " + j)
        f.write('\n')

# generating test.txt
for i, j in zip(X_test, y_test):
    with open(os.path.join(abs_path, 'testlist01.txt'), 'a') as f:
        f.write(i)
        f.write('\n')
os.system("mv ./motion_analysis/classInd.txt ./motion_analysis/annotation/")
os.system("mv ./motion_analysis/testlist01.txt ./motion_analysis/annotation/")
os.system("mv ./motion_analysis/trainlist01.txt ./motion_analysis/annotation/")
os.system("rm ./motion_analysis/trainval.txt")

# # generating train.txt
# for i,label in enumerate(labels):
#     video_names=os.listdir(os.path.join(abs_path,image_folder,'val',label))
#     for video_name in video_names:
#         with open(os.path.join(abs_path,'val.txt'),'a') as f:
#             f.write(os.path.join(abs_path,image_folder,'val',
#             label,video_name)+ " " + dict_labels[label])
#             f.write('\n')
