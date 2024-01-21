import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
# Load the CSV files
df1 = pd.read_csv('./never_parse_ds_studentmodel_35.csv')
df2 = pd.read_csv('./sdanet.csv')

# Extract channel counts from the weight_shape column
df1['Channel Count'] = df1['weight_shape'].str.extract(r'\((\d+),').astype('float')
df2['Channel Count'] = df2['weight_shape'].str.extract(r'\((\d+),').astype('float')

# Group by network names (first part of the 'name' column) and sum the channel counts
df1['Network'] = df1['name'].apply(lambda x: x.split('.')[0])
df2['Network'] = df2['name'].apply(lambda x: x.split('.')[0])

grouped_df1 = df1.groupby('Network')['Channel Count'].sum()
grouped_df2 = df2.groupby('Network')['Channel Count'].sum()

# Plotting
plt.figure(figsize=(15, 8))
grouped_df1.plot(kind='bar', color='blue', alpha=0.7, label='教师网络')
grouped_df2.plot(kind='bar', color='green', alpha=0.7, label='学生网络')
plt.ylabel('通道总数', fontsize=14)
plt.title('网络部分的总通道数', fontsize=16)
plt.legend()
plt.xticks(rotation=45)
plt.show()