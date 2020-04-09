batch_loss = []
aggregated_loss = []
with open("../outputs/batch_loss.txt",'r',encoding = 'utf-8') as f:
    s = f.readline().split()
    while s:
        batch_loss.append(float(s[-1]))
        s = f.readline().split()

index = 0
avg = 0.0
for var in batch_loss:
    if index < 104:
        avg += var  
        index += 1
    else:
        aggregated_loss.append(avg/104)
        index = 0
        avg = 0.0

print(aggregated_loss)