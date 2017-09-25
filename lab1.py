from math import log2
import operator

#根据某一dataset来计算香农熵
def Shannon_entropy(data_set):
    shannon_entropy = 0
    data_class = [temp[-1] for temp in data_set]
    #data_class_set = set(data_class)
    #data_class_list = [i for i in data_class_set]
    data_class_list = list(set(data_class))
    proba = [0 for i in data_class_list]
    for temp in data_class:
        for example in range(len(data_class_list)):
            if (temp == data_class_list[example]):
                proba[example] = proba[example] +1

    probability = [i/len(data_class) for i in proba]

    for i in probability:
        shannon_entropy -= i * log2(i)

    return shannon_entropy


#采用某一feature进行划分,在find_best_feature中会提供该feature的feature list
def feature_split(init_data_set,feature_list,feature_index):
    feature_set = set(feature_list)
    feature_label = list(feature_set)
    data_set_splited = {}#创建一个空字典用于存储划分的多个子dataset,每一个子dataset使用二维列表储存
    for i in feature_label:
        data_set_splited[i] = []

    for i in feature_label:
        for j in range(len(init_data_set)):
            if (feature_list[j]==i):
                reduced_line_data = init_data_set[j][:feature_index]
                reduced_line_data.extend(init_data_set[j][feature_index + 1:])
                data_set_splited[i].append(reduced_line_data)

    return data_set_splited


#当采用某一features划分，根据香农熵来计算某个feature的information gain
def information_gain(init_data_set,set_splited):
    inform_gain = Shannon_entropy(init_data_set)
    for i in set_splited:
        entropy = Shannon_entropy(set_splited[i])
        inform_gain -= (entropy * len(set_splited[i]))/len(init_data_set)

    return inform_gain


#该函数找到当前最理想的划分feature，首先对每一个feature进行划分，然后对划分好的feature计算信息增益，最后找到信息增益最大的feature
def find_best_feature(data_set):
    class_list = [term[-1] for term in data_set]
    #initial_Shannon = Shannon_entropy(class_list)
    feature_num = len(data_set[0])-1

    max_info_gain = 0


    for term in range(feature_num):
        feature_list = [temp[term] for temp in data_set]
        feature_axis = term
        data_set_split = feature_split(data_set,feature_list,feature_axis)
        info_gain = information_gain(data_set,data_set_split)

        if (info_gain > max_info_gain):
            max_info_gain = info_gain
            max_feature_index = term

    return max_feature_index


def information_gain_rate(info_gain,feature_list):
    feature_enum = list(set(feature_list))
    feature_prob = [0 for i in feature_enum]
    for i in feature_list:
        for j in range(len(feature_enum)):
            if (i == feature_enum[j]):
                feature_prob[j] += 1
    feature_prob = [i/len(feature_list) for i in feature_prob]
    H = 0
    for i in feature_prob:
        H -= i * log2(i)

    return H


def find_best_feature_C45(data_set):
    class_list = [term[-1] for term in data_set]
    #initial_Shannon = Shannon_entropy(class_list)
    feature_num = len(data_set[0])-1
    max_info_gain_rate = 0
    for term in range(feature_num):
        feature_list = [temp[term] for temp in data_set]
        feature_axis = term
        data_set_split = feature_split(data_set,feature_list,feature_axis)
        info_gain = information_gain(data_set,data_set_split)
        info_gain_rate = information_gain_rate(info_gain,feature_list)
        if (info_gain_rate > max_info_gain_rate):
            max_info_gain_rate = info_gain_rate
            max_feature_index = term

    return max_feature_index


def class_vote(class_list):
    classCount = {}
    for vote in class_list:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def Tree_creat(data_set,labels,algoritm_type):
    #首先判定两种不能产生子树的情况
    #第一种情况：当前data_set中所有的class结果相同，这说明已经不需要再进行分类了，当前class即为输出
    #第二种情况：当前data_set宽度为1，这意味着所有的features已经被分配完毕，只剩下结果一项，因此根据投票原则，得到输出
    #综上，创建class_list来进行1判断，对data_set进行2判断
    class_list = [last_term[-1] for last_term in data_set]
    if class_list.count(class_list[0]) == len(class_list):#第一种情况
        return class_list[0]
    if len(data_set[0])==1:#第二种情况
        return class_vote(class_list)
    #数据正常，则要从当前dataset中选出一个最优的分类方案
    #feature_best 返回的是整数索引，为了增强可读性，可赋予各个features labels
    if (algoritm_type == 'C4.5'):
        feature_best = find_best_feature_C45(data_set)
        feature_best_label = labels[feature_best]
    if (algoritm_type == 'ID3'):
        feature_best = find_best_feature(data_set)
        feature_best_label = labels[feature_best]

    Tree = {feature_best_label:{}}
    #将当前feature对应的label从labels中删除
    del labels[feature_best]
    #统计当前feature的各个取值
    feature_best_value = [data[feature_best] for data in data_set ]
    feature_best_valueset = set(feature_best_value)
    #对于当前feature的每一个值都递归构造新的子树
    #每一个子树的dataset要经过划分
    #根据上面的信息对已知的最优feature进行划分
    data_set_split = feature_split(data_set, feature_best_value, feature_best)

    for value in feature_best_valueset:
        sub_labels = labels[:]

        sub_data_set = data_set_split[value]
        Tree[feature_best_label][value] = Tree_creat(sub_data_set,sub_labels,algoritm_type)

    return Tree

fr = open('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab1/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
#lensesLabels = ['first','second']
algoritm_type = 'C4.5'
lensesTree = Tree_creat(lenses,lensesLabels,algoritm_type)