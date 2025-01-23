import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker


tensor = np.array([0.3970, 0.3280, 0.0140, 0.0696, 0.1914])

categories = ['1', '2', '3', '4', '5']

plt.bar(categories, tensor, color='red', alpha=0.5)

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


plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

plt.ylabel('')

# plt.savefig('./tensor122.png')
plt.savefig('./tensor122.png', transparent=True)