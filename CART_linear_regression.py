
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
    var = (ny2.sum() - ny.sum()) **2
    return var


def err_cal(left_data, right_data):
    var_left = var_cal(left_data)
    var_right = var_cal(right_data)
    error = var_left + var_right
    return error


def findbest(data):
    err_cur = []
    for pos in range(len(data)):
        data_left,data_right = data_split(data,pos)
        err_cur.append(err_cal(data_left,data_right))

    err_cur_correct = [[err_cur[i],i] for i in range(len(err_cur))]
    err_cur_correct.sort(key=lambda  x: x[0])
    bestpos = err_cur_correct[0][1]
    '''bestpos==0 or bestpos==len(err_cur_correct):
        situantion +=1
        findbest(data,situantion)'''
    if bestpos==0 or bestpos==len(data):
        x_value = data[bestpos][0]
    else:
        x_value = (data[bestpos-1][0] + data[bestpos][0]) / 2
    return bestpos,x_value

def testdata_split(data, critical_value):
    pos = 0
    for i in range(len(data)):
        if float(data[i][1])<critical_value:
            pos = i+1
    left_subdata = data[0:pos]
    right_subdata = data[pos:]
    return left_subdata, right_subdata

def varcal_merge(data,treemean):
    y = [data[i][1] for i in range(len(data))]
    treemean_list = [treemean for i in data]
    var_list = [(y[i]-treemean_list[i])**2 for i in range(len(data))]
    var = sum(var_list)
    return var

def varcal_split(data,estimated_value):
    y = [data[i][1] for i in range(len(data))]
    estimated_list = [estimated_value for i in data]
    var_list = [(y[i] - estimated_list[i]) ** 2 for i in range(len(data))]
    var = sum(var_list)
    return var



def Tree_creat(data, minimum_size = 10):
    situation = 0
    bestpos,x_bestlabel = findbest(data)
    y_value = [i[1] for i in data]
    x_value = [i[0] for i in data]
    if bestpos==0 or bestpos==len(data):#分割点在边界，停止分割，返回平均值
        return sum(y_value)/len(data)


    if y_value.count(y_value[0])==len(y_value):#所有y相等，返回y
        return y_value[0]
    elif x_value.count(x_value[0]) == len(x_value):#所有x相等，返回平均值
        return sum(y_value) / len(y_value)
    if len(data) > minimum_size:#仅当大于4时进一步分裂
        left_subdata, right_subdata = data_split(data, bestpos)
        TreeNode = {x_bestlabel: {}}
        TreeNode[x_bestlabel]['leftchild'] = Tree_creat(left_subdata)
        TreeNode[x_bestlabel]['rightchild'] = Tree_creat(right_subdata)
    else:#小于4不分裂，返回平均值
        return sum(y_value)/len(y_value)
    #return Tree_situation(left_subdata,TreeNode)
    #return Tree_situation(right_subdata,TreeNode)

    return TreeNode




def isTree(tree):
    return (type(tree).__name__=='dict')


def postprune(tree,testdata):
    feature_value = list(tree.keys())[-1]
    #if isTree(tree[feature_value]['leftchild']) or isTree(tree[feature_value]['rightchild']):
    #if len(testdata)==0:
    #    (tree[feature_value]['leftchild'] + tree[feature_value]['rightchild']) / 2
    left_testdata,right_testdata = testdata_split(testdata,feature_value)

    if isTree(tree[feature_value]['leftchild']):
        tree[feature_value]['leftchild'] = postprune(tree[feature_value]['leftchild'],left_testdata)
    if isTree(tree[feature_value]['rightchild']):
        tree[feature_value]['rightchild'] = postprune(tree[feature_value]['rightchild'],right_testdata)
    if not isTree(tree[feature_value]['leftchild']) and not isTree(tree[feature_value]['rightchild']):
        treemean = (tree[feature_value]['leftchild'] + tree[feature_value]['rightchild']) / 2
        error_left = varcal_split(left_testdata, tree[feature_value]['leftchild'])
        error_right = varcal_split(right_testdata, tree[feature_value]['rightchild'])
        error_split = error_right + error_left
        error_merge = varcal_merge(data, treemean)
        if error_merge < error_split:
            return treemean
        else:
            return tree
    else:
        return tree


fr = open('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab1/train.txt')
data = [row.split() for row in fr.readlines()]
colum_len = len(data[0])
for row in range(len(data)):
    for colum in range(colum_len):
        data[row][colum] = float(data[row][colum])
data.sort()


fr = open('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab1/test.txt')
testdata = [row.split() for row in fr.readlines()]
colum_len = len(testdata[0])
for row in range(len(testdata)):
    for colum in range(colum_len):
        testdata[row][colum] = float(testdata[row][colum])
testdata.sort()

#data_debug = [[i,2.14*i] for i in range(1,5)] + [[i,1.76*i] for i in range(6,10)] + [[i,1.12*i] for i in range(11,15)]
data_debug_test = [[i,1.05*i] for i in range(1,3)] + [[i,1.2*i] for i in range(4,6)] +  [[i,2*i] for i in range(7,9)] + [[i,2.2*i] for i in range(10,12)] + [[i,2.3*i] for i in range(13,15)] + [[i,2.35*i] for i in range(16,18)]
CART_Tree = Tree_creat(data)
a = 1
PRUNE_CART_Tree = postprune(CART_Tree,data_debug_test)
b = 1
print(CART_Tree)
print(PRUNE_CART_Tree)