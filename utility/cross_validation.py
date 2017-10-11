# encoding = utf-8
import os
import numpy as np


def split_data_set(file_name, split_size, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fr = open(file_name, 'r')  # open fileName to read
    onefile = fr.readlines()
    num_line = len(onefile)
    arr = np.arange(num_line)  # get a seq and set len=numLine
    np.random.shuffle(arr)  # generate a random seq from arr
    list_all = arr.tolist()
    each_size = (num_line + 1) / split_size  # size of each split sets
    split_all = []
    each_split = []
    # count_num 统计每次遍历的当前个数
    count_num = 0
    # count_split 统计切分次数
    count_split = 0

    # 遍历整个数字序列
    for i in range(len(list_all)):
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == int(each_size):
            count_split += 1
            array_ = np.array(each_split)
            np.savetxt(out_dir + "/split_" + str(count_split) + '.txt', array_, fmt="%s", delimiter='\t')  # 输出每一份数据
            split_all.append(each_split)  # 将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all


def generate_train_test(split_data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    list_file = os.listdir(split_data_dir)
    cross_now = 0
    for eachfile1 in list_file:
        # 记录当前的交叉次数
        cross_now += 1
        # 对其余九份欠抽样构成训练集
        for eachfile2 in list_file:
            if eachfile2 != eachfile1:
                with open(out_dir + "/train_" + str(cross_now) + ".datasets", 'a') as fw_train:
                    with open(split_data_dir + '/' + eachfile2, 'r') as one_file:
                        read_lines = one_file.readlines()
                        for one_line in read_lines:
                            fw_train.writelines(one_line)

        # 将训练集和测试集文件单独保存起来
        with open(out_dir + "/test_" + str(cross_now) + ".datasets", 'a') as fw_test:
            with open(split_data_dir + '/' + eachfile1, 'r') as one_file:
                read_lines = one_file.readlines()
                for one_line in read_lines:
                    fw_test.writelines(one_line)


def main():
    data_set_name = '/home/elics-lee/academicSpace/dataSet/FilmTrust/ratings.txt'
    out_put_dir = '/home/elics-lee/academicSpace/dataSet/FilmTrust/cv_5'
    cv_size = 5
    split_data_set(data_set_name, cv_size, out_put_dir)

    # test generate_train_test function
    train_test_dir = '/home/elics-lee/academicSpace/dataSet/FilmTrust/test_train'
    generate_train_test(out_put_dir, train_test_dir)


if __name__ == '__main__':
    main()