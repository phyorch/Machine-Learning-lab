fr = open('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab1/train.txt')

data = [row.split() for row in fr.readlines()]

for row in data:
    row = [int(i) for i in row]

a = 1