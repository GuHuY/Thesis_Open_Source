# Check the result of wavedet

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


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
y = np.linspace(0,1800,1000)
#生成sigmiod形式的y数据

#设置x、y坐标轴的范围
plt.xlim(0,2000)
plt.ylim(-500, 2000)
#绘制图形
plt.plot(x,x,c='black',label='y=x')
plt.plot(500, 1500, 'x', c='coral', label='RRn(500, 1500)')
plt.plot([1500]*1000, y, c='coral', label='x=1500')
plt.plot(1250, 250, '+', c='b', label='RRn(1250, 250)')
plt.plot([250]*1000, y, c='b', label='x=250')
plt.xlabel('Time(ms)')
plt.ylabel('Time(ms)')
plt.legend(loc='upper right')
plt.show()
pass