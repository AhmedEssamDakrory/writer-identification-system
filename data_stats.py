import os

path = "TestCases"
file_list = []
for root, dirs, files in os.walk(path):
    for file in files:
        file_list.append(file)

print("number of images : " , len(file_list))
file_set = set(file_list)
print("number of unique images: ", len(file_set))

s = set()
for folder in os.listdir(path):
    for writer in os.listdir(os.path.join(path, folder)):
        if writer != "test.png":
            s.add(writer)
print("number of unique writers : ", len(s))
