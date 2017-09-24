#该函数对当前feature的某一取值进行dataset的划分
def feature_split()


#根据某一feature的列向量来计算香农熵
def Shannon_entropy(features):
    features_set = set(features)





#该函数找到当前最理想的划分feature，首先对每一个feature进行划分，然后对划分好的feature计算信息增益，最后找到信息增益最大的feature
def find_best_feature(data_set):
    class_list = [term[-1] for term in data_set]
    #initial_Shannon = Shannon_entropy(class_list)
    feature_num = len(data_set[0])-1

    max_info_gain = 0

    for term in range(feature_num)
        feature_list = [temp[term] for temp in data_set]
        info_gain = information_gain(feature_list)
        if (info_gain > max_info_gain):
            max_info_gain = info_gain
            max_feature_index = term

    return max_feature_index



class_vote(class_list)

information_gain(features)

def Tree_creat(data_set,labels):

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
    for value in feature_best_valueset
        sub_labels = labels[:]
        Tree[feature_best_label][feature_best_value] = Tree_creat

    return Tree

