# -*- coding: UTF-8 -*-
import numpy as np
from math import acos, pi, fabs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# IQ data path
data_path = "/home/jerry/文档/RFID_data/save_data/all_data/dong15693/1out_of_4_fan_dishaung25/"
file_name = "new_save_ISO_15693_3.txt"
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
            data_first.append(float(i_data)*1000)
            data_second.append(float(q_data)*1000)
    f.close()
    return data_first, data_second


# calculate two vector distance
def eucldist_vectorized(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return np.sqrt(np.sum((np.array(coords1) - np.array(coords2))**2))


# calculate two vector angle
def angle_vec2(coords1,coords2):
    dot = np.array(coords1).dot(np.array(coords2))
    len_v1 = np.sqrt(np.sum(np.array(coords1)**2))
    len_v2 = np.sqrt(np.sum(np.array(coords2)**2))
    cosa = dot/(len_v1*len_v2)
    a = acos(cosa)
    return a*180/pi


# calculate power 100
def cal_am(i_data_list,q_data_list):
    power_list = list()
    for i in range(len(i_data_list)):
        power_list.append(eucldist_vectorized(i_data_list[i],q_data_list[i]))
    return power_list


# calculdate tag begin
def cal_begin_tag(i_data_list, q_data_list):
    data_length = len(i_data_list)
    begin = 500
    carrier_points = 500
    kmeans_points = 500
    steps = 500
    last_distance = 100
    result = list()
    for i in range(begin, data_length-1-kmeans_points, steps):
        i_cariier_avg = np.average(i_data_list[i-carrier_points:i])
        q_cariier_avg = np.average(q_data_list[i-carrier_points:i])
        carrier_point = [i_cariier_avg, q_cariier_avg]
        kmeans_data = list()
        for j in range(kmeans_points):
            kmeans_data.append([i_data_list[i+j], q_data_list[i+j]])
        clf = KMeans(n_clusters=2)
        clf.fit(kmeans_data)
        center1, center2 = clf.cluster_centers_
        current_distance = eucldist_vectorized(center2, center1)
        if current_distance > 15:
            result.append(i)
    final_result = [result[0]]
    for x in range(1, len(result)-2):
        y = x+1
        if (result[y]-result[x])>=3*steps:
            final_result.append(result[y])
    return final_result


# calculate tag begin 2
def cal_tag_begin(signal_am):
    data_length = len(signal_am)
    begin = 1000
    steps = 100
    cal_points = 1000
    result_list = list()
    last_avg_am = np.average(signal_am[begin-cal_points:begin])
    last_wave = 0
    for i in range(begin,data_length-cal_points,steps):
        current_avg_am = np.average(signal_am[i:i+cal_points])
        current_wave = np.abs(current_avg_am-last_avg_am)
        if current_wave > 1000*last_wave or current_wave < last_wave/1000:
            result_list.append(i)
        last_avg_am = current_avg_am
        last_wave = current_wave
    return result_list


# calculate 3avg
def cal_3_avg(signal_am):
    data_length = len(signal_am)
    result = list()
    for i in range(30,data_length-30,30):
        result.append(np.average(signal_am[i-30:i]))
    return result

# begin calculate
i, q = read_data(data_path)
data_am = cal_am(i, q)
#data_am_avg = cal_3_avg(data_am)
data_len = len(i)

'''
tag_begin = cal_begin_tag(i, q)
tag_margin = list()
tag_margin.append(0)
for k in range(1, len(tag_begin)-1):
    tag_margin.append(tag_begin[k]-tag_begin[k-1])
margin_sort = np.argsort(np.array(tag_margin))
print "tag begin: "+str(len(tag_begin))+" data: "+str(tag_begin)
print
print "tag margin:"+str(len(tag_margin))+" data: "+str(tag_margin)
print
print "margin_sort_index:"+str(len(margin_sort))+" data: "+str(margin_sort)
print
print tag_begin[margin_sort[0]],tag_begin[margin_sort[1]]
#exit()
'''

tag_begins = cal_begin_tag(i,q)
print "tag begins: "+str(tag_begins)


'''
15693:

4_fan_didan24
1. 
2. 26080

4_fan_dishuang25
1.24448
3.24448
'''
begin = 24448  # 2:26080 2000  3:
points = 1000
end = begin+points
carrier_points = 1000
x_data = range(data_len)[begin:end]


i_cariier_avg = np.average(i[begin-carrier_points:begin])
q_cariier_avg = np.average(q[begin-carrier_points:begin])
carrier_avg = [i_cariier_avg, q_cariier_avg]

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
am = eucldist_vectorized(center1, center2)
print "自测幅度："+str(am)+" mv"
center1_cariier_distance = eucldist_vectorized(center1, carrier_avg)
center2_cariier_distance = eucldist_vectorized(center2, carrier_avg)
low_scatter_point = center1
high_scatter_point = center2
# low scatter is the Closest to carrier point
if center1_cariier_distance>center2_cariier_distance:
    low_scatter_point = center2
    high_scatter_point = center1
# o_low and low_high angle
alpha = angle_vec2(low_scatter_point, high_scatter_point-low_scatter_point)
print "自测相位："+str(alpha)+"度"
print


# print ni
ni_center1 = np.array([-25.27, 16.57])
ni_center2 = np.array([-39.17, 31.45])
ni_center_x = [ni_center1[0], ni_center2[0]]
ni_center_y = [ni_center1[1], ni_center2[1]]
print "NI中心点 1："+str(ni_center1)
print "NI中心点 2："+str(ni_center2)
ni_am = eucldist_vectorized(ni_center1, ni_center2)
print "NI幅度："+str(ni_am)+" mv"
ni_center1_cariier_distance = eucldist_vectorized(ni_center1, carrier_avg)
ni_center2_cariier_distance = eucldist_vectorized(ni_center2, carrier_avg)
ni_low_scatter_point = ni_center1
ni_high_scatter_point = ni_center2
if ni_center1_cariier_distance>ni_center2_cariier_distance:
    ni_low_scatter_point = ni_center2
    ni_high_scatter_point = ni_center1
# o_low and low_high angle
ni_alpha = angle_vec2(ni_low_scatter_point, ni_high_scatter_point-ni_low_scatter_point)
print "NI相位:"+str(ni_alpha)+"度"
print


#print error
print "幅度误差："+str(fabs(am-ni_am)/ni_am*100)+"%"
print "相位误差："+str(fabs(alpha-ni_alpha))+"度"

###
#fft = np.fft.fft(i[begin:end])  #调用fft变换算法计算频域波形
#print "FFT LEN: "+str(len(fft))
###


# plot func
ax1 = plt.figure(1)
ax2 = plt.figure(2)
ax3 = plt.figure(3)
#ax4 = plt.figure(4)

plt.figure(1)
plt.plot(range(data_len), i)
plt.plot(range(data_len), q)

plt.figure(2)
plt.plot(x_data, i[begin:end])
plt.plot(x_data, q[begin:end])

plt.figure(3)
plt.scatter(i[begin:end], q[begin:end], marker='o',s=10, alpha=0.4)
plt.scatter(center_x, center_y, marker='*', s=50, label='Ours')
plt.scatter(ni_center_x, ni_center_y, marker='x', s=30, label='Ni')
plt.scatter(i_cariier_avg, q_cariier_avg, marker='+', s=50, label='Carrier')
plt.scatter(0, 0, label='Zero')
plt.legend(loc='upper right')

#plt.figure(4)
#plt.plot(range(len(data_am)), data_am)
plt.show()
