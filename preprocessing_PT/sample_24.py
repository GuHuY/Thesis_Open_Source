# Check the result of wavedet

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


interval_list = [1010, 1010, 530, 1420, 1010,1010]
# interval_list = [1000, 1000, 1500, 200, 1000, 1000]
points = []
for idx in range(len(interval_list)-1):
    points.append([interval_list[idx], interval_list[idx+1]])

points = np.array(points)

#创建画布
fig = plt.figure(figsize=(8, 8))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)

ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
#设置x、y轴上刻度显示方向
ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("right")

x = np.linspace(0,1800,1000)
#生成sigmiod形式的y数据

#设置x、y坐标轴的范围
plt.xlim(0,2000)
plt.ylim(0, 2000)
#绘制图形
plt.plot(x,x,c='black',label='y=x')
plt.plot(points[:,0], points[:,1])
plt.plot(points[:,0], points[:,1],'o')
for i in range(len(points)-1):
    plt.text(points[i,0], points[i,1]+30, str(i+1))
plt.text(points[-1,0], points[-1,1]-70, str(len(points)))
plt.xlabel('Time(ms)')
plt.ylabel('Time(ms)')
plt.legend(loc='upper right')
plt.show()
pass