# Test Log Reg
# Dalsi zkusti - https://www.kaggle.com/prashant111/logistic-regression-classifier-tutorial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.interactive(False)

url="https://raw.githubusercontent.com/pcsanwald/kaggle-titanic/master/train.csv"
train = pd.read_csv(url)
train.info()
train.head()

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show(block=True)
plt.interactive(False)

sns.set_style('whitegrid')
sns.countplot(x='survived',data=train)
plt.show(block=True)
plt.interactive(False)



