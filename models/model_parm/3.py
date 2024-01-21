import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('./never_parse_ds_studentmodel_35.csv')
df2 = pd.read_csv('./sdanet.csv')

# Function to extract the output channels from the weight_shape column
def extract_output_channels(row):
    if pd.isna(row['weight_shape']):
        return 0
    return int(row['weight_shape'].split(',')[0].strip('()'))

# Apply the function to both dataframes
df1['output_channels'] = df1.apply(extract_output_channels, axis=1)
df2['output_channels'] = df2.apply(extract_output_channels, axis=1)

# Plotting comparison
plt.figure(figsize=(15, 6))

# Plot for Student Network
plt.plot(df1.index, df1['output_channels'], label='Student Network', marker='o')

# Plot for Teacher Network
plt.plot(df2.index, df2['output_channels'], label='Teacher Network', marker='x')

# Adding titles and labels
plt.title('Comparison of Output Channels - Student vs Teacher Network')
plt.xlabel('Layer Index')
plt.ylabel('Number of Output Channels')
plt.legend()

# Show the plot
plt.show()