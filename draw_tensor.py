import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker


# tensor=np.loadtxt('./tensor.txt')
# tensor=np.array(list([0.0273, 0.0029, 0.0384, 0.0122, 0.0137, 0.0199, 0.0051, 0.0080, 0.0279,
#         0.0231, 0.0306, 0.0037, 0.0145, 0.0263, 0.0157, 0.0006, 0.0256, 0.0266,
#         0.0102, 0.0216, 0.0041, 0.0055, 0.0222, 0.0209, 0.0065, 0.0058, 0.0068,
#         0.0234, 0.0303, 0.0202, 0.0059, 0.0053, 0.0049, 0.0343, 0.0064, 0.0173,
#         0.0056, 0.0044, 0.0039, 0.0046, 0.0319, 0.0204, 0.0226, 0.0034, 0.0376,
#         0.0217, 0.0127, 0.0159, 0.0255, 0.0069, 0.0054, 0.0309, 0.0037, 0.0106,
#         0.0057, 0.0420, 0.0032, 0.0128, 0.0148, 0.0258, 0.0076, 0.0038, 0.0176,
#         0.0250]))

tensor = np.array([0.3970, 0.3280, 0.0140, 0.0696, 0.1914])

# plt.figure(figsize=(8, 7.5)) 

# sns.kdeplot(tensor, color="red", fill=True, alpha=0.5,)  # alpha 控制透明度，即饱和度
# sns.histplot(data=tensor, bins=4, kde=True,color="red")
sns.histplot(data=tensor, bins=4)

# 设置坐标轴刻度但不显示刻度标签
plt.tick_params(
    axis='both',          # 应用到x轴和y轴
    which='both',         # 应用到主刻度和副刻度
    bottom=False,         # 不显示底部刻度
    top=False,            # 不显示顶部刻度
    left=False,           # 不显示左侧刻度
    right=False,          # 不显示右侧刻度
    labelbottom=True,    # 不显示底部刻度标签
    labeltop=False,       # 不显示顶部刻度标签
    labelleft=True,      # 不显示左侧刻度标签
    labelright=False      # 不显示右侧刻度标签
)

ax = plt.gca()

# 设置x轴的主刻度格式器为科学计数法
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# 开启科学计数法
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


# offset_text = ax.xaxis.get_offset_text()
# offset_text.set_position((0.5, 1))
# offset_text.set_fontsize(22)

# 仅显示最多N个刻度
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
# ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.ylabel('')

plt.savefig('./tensor122.png')
# plt.savefig('./tensor122.png', transparent=True)