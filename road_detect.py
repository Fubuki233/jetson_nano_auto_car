import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img0 = cv.imread('./v2-59502527ef55abb6bd02d8b91754768b_720w.jpg')
gray = cv.cvtColor(img0,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
# cv.imshow('yuantu',img0)
# cv.waitKey(0)
#img0 = cv.resize(img0, (600, 600))  # 改为600*600大小的
#plt.imshow(img0)  # 这里做出说明，用plt显示图片是为了顺利找出roi，因为plt显示图片，鼠标指哪里，就会自动显示改点坐标。
#plt.show()
ret, img1 = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)  # 二值化
edged = cv.Canny(img1, 50, 150)
contours, h = cv.findContours(edged.copy(), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)   #只储存车道的顶点数据 只检测外轮廓

largest_contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]   #找到面积最大的车道
sec_largest_contours = sorted(contours, key=cv.contourArea, reverse=True)[2:3]   #找到面积次大的车道
#这样的方法不合理 如果图像出现比车道面积还大的块就会找错车道
#pts = largest_contours.reshape(4, 2)
driving_reference=img0.copy()
numlg=0 #最大车道的坐标数量
numse=0 #次大车道的坐标数量
lg_x=[] #最大车道的x坐标 以下类同
lg_y=[]
se_x=[]
se_y=[]
for cnt in largest_contours:    #获取最大车道的边缘坐标

    for i in cnt:
       lg_x.append(i[0][0])
       lg_y.append(i[0][1])
       print(lg_x[numlg],',',lg_y[numlg])
       #print(numlg)
       numlg+=1
       print(numlg)
for cnt in sec_largest_contours:   #次大车道的边缘坐标

    for i in cnt:
       se_x.append(i[0][0])
       se_y.append(i[0][1])
       print(se_x[numse],',',se_y[numse])
       #print(numlg)
       numse+=1
       print(numse)
for cnt in range(int(min(numlg,numse))):    #绘制路线
    cv.circle(driving_reference, (int((lg_x[cnt]+se_x[cnt])/2),lg_y[cnt]), 1, (0, 0, 213), 3) #方法是取x平均值 以最大车道的Y坐标为路线的Y坐标 Y坐标取值可能有问题
    #cv.circle(driving_reference, (604,387), 3, (0, 0, 213), 3)
#print(largest_contours)
#pts = largest_contours.reshape(4, 2)#将轮廓的点重新整理成一个4x2的矩阵，即四个点，每个点两个坐标。
#print(pts)
contour=img0.copy()
cv.drawContours(contour,largest_contours,-1,(0,0,255),3)

#cv.namedWindow('Image', 0)
cv.imshow("orgin", img0)
cv.imshow("Image", img1)
cv.imshow("contours", contour)
cv.imshow("driving_reference", driving_reference)
cv.waitKey(0)
