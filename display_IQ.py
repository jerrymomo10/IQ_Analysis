# -*- coding: UTF-8 -*-

import numpy as np
from math import acos, pi, fabs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# IQ data path
data_path = "/home/jerry/文档/RFID_data/save_data/all_data/dong14443b/"
file_name = "new_save_ISO_14443_Type_B_4.txt"
data_path = data_path+file_name


# read IQ data
def read_data(file_path):
    data_first = list()
    data_second = list()
    with open(file_path,'r') as f:
        for i in range(13):
            f.readline()
        for line in f.readlines():
            i_data, q_data = line.strip().split('\t')
            data_first.append(float(i_data))
            data_second.append(float(q_data))
    f.close()
    return data_first, data_second


# calculate two vector distance
def eucldist_vectorized(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return np.sqrt(np.sum((np.array(coords1) - np.array(coords2))**2))


# calculate two vector angle
def angle_vec2(coords1,coords2):
    dot = np.sum(np.array(coords1)*np.array(coords2))
    len_v1 = eucldist_vectorized(coords1, 0.0)
    len_v2 = eucldist_vectorized(coords2, 0.0)
    cosa = dot/(len_v1*len_v2)
    a = acos(cosa)
    return a*180/pi


# calculate power 100
def cal_power(i_data_list,q_data_list):
    power_list = list()
    for i in range(len(i_data_list)):
        power_list.append(fabs(i_data_list[i]*q_data_list[i]))
    return power_list


# begin calculate
i, q = read_data(data_path)
power = cal_power(i, q)
data_len = len(i)
begin = 17700  # 2:26080 2000  3:
points = 1000
end = begin+points
x_data = range(data_len)[begin:end]

#temp = np.fft.fft(i)


# kmeans to find center
kmeans_data = list()
for m in range(data_len):
    kmeans_data.append([i[m], q[m]])
clf = KMeans(n_clusters=2)
s = clf.fit(kmeans_data[begin:end])
center1, center2 = clf.cluster_centers_
center_x = [center1[0], center2[0]]
center_y = [center1[1], center2[1]]


#print 自测
print "自测中心点 1："+str(center1)
print "自测中心点 2："+str(center2)
am = 0.5*eucldist_vectorized(center1, center2)
print "自测幅度："+str(am)
alpha = angle_vec2(center1, center2)
print "自测相位："+str(alpha)
print


# print ni
ni_center1 = [-1.328, -0.2609]
ni_center2 = [-0.9808, -0.1311]
ni_center_x = [ni_center1[0],ni_center2[0]]
ni_center_y = [ni_center1[1],ni_center2[1]]
print "NI中心点 1："+str(ni_center1)
print "NI中心点 2："+str(ni_center2)
ni_am = 0.5*eucldist_vectorized(ni_center1, ni_center2)
print "NI幅度："+str(ni_am)
ni_alpha = angle_vec2(ni_center1, ni_center2)
print "NI相位:"+str(ni_alpha)
print


#print error
print "幅度误差："+str(fabs(am-ni_am)/ni_am*100)+"%"
print "相位误差："+str(fabs(alpha-ni_alpha))+"度"


# plot func
ax1 = plt.figure(1)
ax2 = plt.figure(2)
ax3 = plt.figure(3)
ax3 = plt.figure(4)

plt.figure(1)
plt.plot(range(data_len), i)
plt.plot(range(data_len), q)

plt.figure(2)
plt.plot(x_data, i[begin:end])
plt.plot(x_data, q[begin:end])

plt.figure(3)
plt.scatter(i[begin:end], q[begin:end], s=10, alpha=0.4, marker='o')
plt.scatter(center_x, center_y, marker='o', label='Ours', s=30)
plt.scatter(ni_center_x, ni_center_y, marker='x', s=30, label='NI')
plt.scatter(0, 0, label='Zero')
plt.legend(loc='upper right')

plt.figure(4)
plt.plot(range(data_len), power)
plt.show()
