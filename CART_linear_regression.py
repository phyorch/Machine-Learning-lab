
import numpy



def data_split(data, split_position):
    left_subdata = data[0:split_position]
    right_subdata = data[split_position:]

    return left_subdata,right_subdata

def var_cal(data):
    if not data:
        return 0
    y = [elem[1] for elem in data]
    ny = numpy.array(y)
    ny2 = ny * ny
    var = ny2.sum()/len(ny) - (ny.sum()/len(ny)) **2
    return var


def err_cal(left_data, right_data):
    var_left = var_cal(left_data)
    var_right = var_cal(right_data)
    error = var_left + var_right
    return error


def findbest(data,situantion):
    err_cur = []
    for pos in range(len(data)):
        data_left,data_right = data_split(data,pos)
        err_cur.append(err_cal(data_left,data_right))

    err_cur_correct = [[err_cur[i],i] for i in range(len(err_cur))]
    err_cur_correct.sort(key=lambda  x: x[0])
    bestpos = err_cur_correct[situantion][1]
    if bestpos==0 or bestpos==len(err_cur_correct):
        situantion +=1
        findbest(data,situantion)
    x_value = (data[bestpos-1][0] + data[bestpos][0]) / 2
    return bestpos,x_value

'''def Tree_situation(data,TreeNode):
    y_value = [i[1] for i in data]
    x_value = [i[0] for i in data]
    if len(data)>4:
        if y_value.count(y_value[0])==len(y_value):
            return y_value[0]
        elif x_value.count(x_value[0])==len(x_value):
            return sum(y_value)/len(y_value)
        else:
            TreeNode[x_value]['leftchild'] = Tree_creat(data)
    elif len(data)>0:
        return sum(y_value)/len(y_value)'''

def Tree_creat(data, minimum_size = 4):
    situation = 0
    bestpos,x_value = findbest(data,situation)
    left_subdata,right_subdata = data_split(data,bestpos)
    TreeNode = {x_value:{}}

    #return Tree_situation(left_subdata,TreeNode)
    #return Tree_situation(right_subdata,TreeNode)

    y_leftvalue = [i[1] for i in left_subdata]
    y_rightvalue = [i[1] for i in right_subdata]
    x_leftvalue = [i[0] for i in left_subdata]
    x_rightvalue = [i[0] for i in right_subdata]
    if len(left_subdata)>4:
        if y_leftvalue.count(y_leftvalue[0])==len(y_leftvalue):
            return y_leftvalue[0]
        elif x_leftvalue.count(x_leftvalue[0]) == len(x_leftvalue):
            return sum(y_leftvalue) / len(y_leftvalue)
        else:
            TreeNode[x_value]['leftchild'] = Tree_creat(left_subdata)
    elif len(left_subdata)>0:
        return sum(y_leftvalue)/len(y_leftvalue)

    if len(right_subdata)>4:
        if x_rightvalue.count(x_rightvalue[0])==len(x_rightvalue):
            return y_rightvalue[0]
        elif x_rightvalue.count(x_rightvalue[0]) == len(x_rightvalue):
            return sum(y_rightvalue) / len(y_rightvalue)
        else:
            TreeNode[x_value]['rightchild'] = Tree_creat(right_subdata)
    elif len(right_subdata) > 0:
        return sum(y_rightvalue) / len(y_rightvalue)


fr = open('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab1/train.txt')

data = [row.split() for row in fr.readlines()]

colum_len = len(data[0])
for row in range(len(data)):
    for colum in range(colum_len):
        data[row][colum] = float(data[row][colum])
data.sort()

CART_Tree = Tree_creat(data)


