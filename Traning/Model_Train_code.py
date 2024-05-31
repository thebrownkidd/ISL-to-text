import numpy as np
import pandas as pd
import tensorflow as tf
f = open("C:/Projects/ISL-to-text/0-9/0/asl0.jpgposencoded.txt")
a = f.readline()
def toarray(a):
    a = a[1:len(a)-2]
    val = []
    x=0
    while x<len(a):
        if(len(val) == 21):
            break
        temp = ""
        # print(a[x:])
        for y in range(x,len(a)):
            if a[y] != ",":
                temp += a[y]
            else:
                x = y+1
                break
        if len(temp) != 0:
            val.append(float(temp))
        x+=1
    return val
def one_hot(i):
    ret = [0]*9
    if i <= 5:
        ret[i] = 1
    else:
        ret[i-1] = 1
    return ret
model = tf.keras.Sequential([tf.keras.layers.Dense(units = 21, activation = 'relu',input_shape = [21]),tf.keras.layers.Dense(units = 9,activation = 'relu'),tf.keras.layers.Dense(units = 9, activation = 'softmax')])
model.summary()