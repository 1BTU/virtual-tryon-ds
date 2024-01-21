import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('player_data.csv')
for i in range(50):
    x=df.loc[i]
    name=x.loc[['球员']]
    y=x.loc[['犯规','篮板','助攻','抢断','盖帽','失误']]
    labels=np.array(['犯规','篮板','助攻','抢断','盖帽','失误'])
    data=np.array(y)
    plt.rcParams['font.sans-serif']=['SimHei']
    angles=np.linspace(0, 2*np.pi,len(labels),endpoint=False)
    labels=np.concatenate((labels,[labels[0]]))
    data=np.concatenate((data,[data[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    plt.polar(angles, data,'bo-',linewidth=1)
    plt.thetagrids(angles*180/np.pi,labels)
    plt.fill(angles, data,facecolor='b',alpha=0.25)
    plt.title(str(name))
    plt.show()
