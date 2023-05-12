
y = []

for line in open("expsample_DQN_agentLogs.txt","r",encoding='UTF-8'):
    if line[0:9] == "overall s":
        y.append(eval(line[32:39]))
        # print(eval(line[32:39]))
        # print(overall switching reward tensor(-0.1623),)


import matplotlib.pyplot as plt

x = []

for i in range(len(y)):
    x.append(i)

plt.plot(x, y)
plt.savefig("5,2,sw,小decay.jpg")