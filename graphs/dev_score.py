fscore = []
with open("../outputs/fscore_dev.txt",'r',encoding = 'utf-8') as f:
    s = f.readline().split()
    while s:
        fscore.append(float(s[-1]))
        s = f.readline().split()

print(fscore[:37])