import os

train_ind_list = []
val_ind_list = []
with open("test.txt","r") as train_f:
    for line in train_f:
        train_ind_list.append(line)
with open("val.txt","r") as val_f:
    for line in val_f:
        val_ind_list.append(line)

trainval_ind_list = []
trainval_ind_list.extend(train_ind_list)
trainval_ind_list.extend(val_ind_list)
trainval_ind_list = sorted(trainval_ind_list)

with open("valtest.txt","w") as trainval_f:
    for ind in trainval_ind_list:
        trainval_f.write(ind)
