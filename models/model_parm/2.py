import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
df1 = pd.read_csv('./never_parse_ds_studentmodel_35.csv')
df2 = pd.read_csv('./sdanet.csv')

# Function to extract the output channels from the weight_shape column
def extract_output_channels(weight_shape):
    if pd.isna(weight_shape):
        return 0
    return int(weight_shape.split(',')[0].strip('()'))

# Apply the function to both dataframes
df1['Output Channels'] = df1['weight_shape'].apply(extract_output_channels)
df2['Output Channels'] = df2['weight_shape'].apply(extract_output_channels)

# Group by the 'name' column and sum the output channels
grouped1 = df1.groupby(df1['name'].str.split('.').str[0]).sum()['Output Channels']
grouped2 = df2.groupby(df2['name'].str.split('.').str[0]).sum()['Output Channels']

# Create a subplot for each dataframe
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the data
sns.barplot(x=grouped1.index, y=grouped1.values, ax=axs[0])
sns.barplot(x=grouped2.index, y=grouped2.values, ax=axs[1])

# Set plot titles
axs[0].set_title('Output Channels by Network Section (Dataset 1)')
axs[1].set_title('Output Channels by Network Section (Dataset 2)')

# Set y-axis labels
axs[0].set_ylabel('Total Output Channels')
axs[1].set_ylabel('Total Output Channels')

# Rotate x-axis labels for readability
for ax in axs:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# Display the plot
plt.tight_layout()
plt.show()
