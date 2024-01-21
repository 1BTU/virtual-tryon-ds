import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into pandas dataframes
df1 = pd.read_csv('./never_parse_ds_studentmodel_35.csv')
df2 = pd.read_csv('./sdanet.csv')

# Extract the channel count from the weight_shape column
df1['Channel Count'] = df1['weight_shape'].str.extract(r'\((\d+),').astype(float)
df2['Channel Count'] = df2['weight_shape'].str.extract(r'\((\d+),').astype(float)

# Group by the first part of the name and sum the channel counts
df1['Network'] = df1['name'].str.split('.').str[0]
df2['Network'] = df2['name'].str.split('.').str[0]

grouped_df1 = df1.groupby('Network')['Channel Count'].sum()
grouped_df2 = df2.groupby('Network')['Channel Count'].sum()

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(grouped_df1.index, grouped_df1.values, alpha=0.6, label='Model 1')
plt.bar(grouped_df2.index, grouped_df2.values, alpha=0.6, label='Model 2')
plt.xlabel('Network')
plt.ylabel('Total Channel Count')
plt.title('Channel Count by Network in Two Models')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('input_channel.png')
plt.show()

