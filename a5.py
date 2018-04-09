from naiveBayesDensity import *
from test import *
from itertools import *
#auto-mpg
outFile = open("auto-mpg-fixed.txt", "w")
for line in open("auto-mpg.txt"):
    if '?' not in line:
        newLineNums = "\t".join([a for a in "".join([a for a in takewhile(lambda x: not x.isalpha(), line)]).split(" ") if len(a) > 0])
        newLineAttrs = "".join([a for a in dropwhile(lambda x: not x.isalpha(), line)])
        print(newLineNums + newLineAttrs)
        outFile.write(newLineNums + newLineAttrs)
objectsS = Classifier("auto-mpg-fixed.txt","class\tnum\tnum\tnum\tnum\tnum\tnum\tnum\tattr")
print(objectsS.classify(['amc hornet'], [6,199.0,97.00,2774.,15.5,70,1]))
print()
inputFile = open("test.txt", "r")
lists = []
lines = inputFile.readlines()
inputFile.close()
for line in lines:
    field = line.strip().split('\n')
    x = field[0].split(',')
    lists.append(x)


outputFile = open("a5.txt", "w")

for line in lists:
    for i in line:
        outputFile.write(i + "\t")
    outputFile.write("\n")

outputFile.close()

#car
objectsF = Classifier2("a5.txt","attr\tattr\tattr\tattr\tattr\tattr\tclass")
print(objectsF.prior)
print(objectsF.classify2(['vhigh','vhigh','2','2','big','high'], []))
