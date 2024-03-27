import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
frame = cv2.imread('./photos/blkm1.png')
frame = cv2.resize(frame, (320, 240))
# 2、透视变换 (查看俯视图)
matSrc = np.float32([[0, 149], [320, 149], [281, 72], [43, 72]])
matDst = np.float32([[0, 240], [320, 240], [320, 0], [0, 0]])
matAffine = cv2.getPerspectiveTransform(matSrc, matDst)  # mat 1 src 2 dst
dst = cv2.warpPerspective(frame, matAffine, (320, 240))
cv2.imshow('trans',dst)

# 画蓝色的矩形框，采集数据的时候采集的就是当前蓝色矩形框中的数据
# src 4->dst 4 (左下角 右下角 右上角 左上角 )
pts = np.array([[0, 149], [320, 149], [281, 72], [43, 72]], np.int32)
# 顶点个数：4，矩阵变成4*1*2维
# OpenCV中需要将多边形的顶点坐标变成顶点数×1×2维的矩阵
# 这里 reshape 的第一个参数为-1, 表示“任意”，意思是这一维的值是根据后面的维度的计算出来的
pts = pts.reshape((-1, 1, 2))
cv2.polylines(frame, [pts], True, (255, 0, 255), 3)  # True表示闭合，（255,0,0）是蓝色，3是线宽
# 3、转化图像二值化(看二值化之后的黑白图像）
cv2.imshow('frame', frame)
dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)   #转灰度
dst_retval, dst_binaryzation = cv2.threshold(dst_gray, 68, 255, cv2.THRESH_BINARY)   #二值化
#dst_binaryzation = cv2.erode(dst, None, iterations=1)  # 腐蚀化操作
#dst = cv2.bitwise_not(dst)

# 当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
# 当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列(这里的一列是为了方便理解说的，实际上，在控制台的输出中，仍然是以一行的形式输出的)
histogram = np.sum(dst_binaryzation[dst_binaryzation.shape[0] // 2:, :], axis=0)  # 我们这里是一维数组
#midpoint = np.int(histogram.shape[0] / 2)  # 直方图垂直尺寸

left_sum = np.sum(histogram[:20], axis=0)  # 计算左边像素点之和
right_sum = np.sum(histogram[300:], axis=0)  # 计算右边像素点之和

# print("left_sum =%d "%left_sum)
# print("right_sum = %d"%right_sum)


rightpoint = 320
center_r = 159
# print (histogram)
# print(histogram[::-1])
# plt.plot(histogram)
# plt.plot(histogram[::-1])
# plt.show()

# 4、获取黑线左右边线并绘制线条
# 获取左右线的位置
leftx_base = np.argmin(histogram[:center_r], axis=0)
rightx_base = np.argmin(histogram[::-1][:center_r], axis=0)  # 反转直方图取最右侧的值
print(np.min(histogram[:center_r],axis=0))
print(np.min(histogram[::-1][:center_r],axis=0))
llne=0
rlne=0
if(leftx_base>40):
    print("left lane exists")
elif(np.min(histogram[:center_r],axis=0)>255*100):
    print("left lane dosnt exist")
    llne=1
    leftx_base = np.argmin(histogram[:rightpoint], axis=0)

print(leftx_base, rightx_base)
if(rightx_base>40):
    #print(leftx_base, rightx_base)
    print("right lane exists")
elif(np.min(histogram[::-1][:center_r],axis=0)>255*100):
    print("right lane dosnt exist")
    rlne=1
    rightx_base = np.argmin(histogram[::-1][:rightpoint], axis=0)
else:
    print("right lane exists")
if(llne==1 and rlne==1):
    print("no lane exists or lane lies in the graph")
    if(np.min(histogram[::-1][:center_r],axis=0)>255*130 or np.min(histogram[:center_r],axis=0)>255*130):
        print("lane lies")
rightx_base = 319 - rightx_base
print(leftx_base, rightx_base)
dst_binaryzation = cv2.cvtColor(dst_binaryzation, cv2.COLOR_GRAY2RGB)
cv2.line(dst_binaryzation, (159, 0), (159, 240), (255, 0, 255), 2)  # 图像中线 紫色 # fix liusen 200321  149->159
lane_center = int((leftx_base + rightx_base) / 2)  # 左右线中间位置
# print("lane_center")
# print(lane_center)
cv2.line(dst_binaryzation, (leftx_base, 0), (leftx_base, 240), (0, 255, 0), 2)  # 赛道左侧位置
cv2.line(dst_binaryzation, (rightx_base, 0), (rightx_base, 240), (0, 255, 0), 2)  # 赛道右侧位置
cv2.line(dst_binaryzation, (lane_center, 0), (lane_center, 240), (255, 0, 0), 2)  # 实际中线位置

left_sum_value = int(np.sum(histogram[:center_r], axis=0)) / 159
right_sum_value = int(np.sum(histogram[center_r:], axis=0)) / 159
# print("left_sum_value = %d", left_sum_value)
# print("right_sum_value = %d", right_sum_value)
# 5、计算偏差
Bias = 159 - lane_center
cv2.putText(dst_binaryzation, "Bias: " + str(int(Bias)), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(dst_binaryzation, "rightx_base: " + str(int(rightx_base)), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(dst_binaryzation, "leftx_base: " + str(int(leftx_base)), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
# print(Bias)
cv2.imshow("dst", dst)
cv2.imshow('dst_', dst_binaryzation)
cv2.waitKey()