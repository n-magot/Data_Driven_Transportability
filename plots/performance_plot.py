import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# File path
file_path_50 = r'C:\...s\PythonProject\experiments_m_bias_Ne50_alpha0.99.csv'
file_path_100 = r'C:\...\PythonProject\experiments_m_bias_Ne100_alpha0.99.csv'
file_path_300 = r'C:\...\PythonProject\experiments_m_bias_Ne300_alpha0.99.csv'

def data_preds(path):
    # Initialize lists for columns
    first_column = []
    second_column = []
    third_column = []
    forth_column = []

    # Read the CSV file
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        # Skip the header
        next(csv_reader)

        # Extract columns
        for row in csv_reader:
            first_column.append(float(row[0]))
            second_column.append(float(row[1]))
            third_column.append(float(row[2]))
            forth_column.append(float(row[3]))

    return first_column, second_column, third_column, forth_column

# For Ne = 50
F_1, F_2, F_3, F_11  = data_preds(file_path_50)

data1 = {'Ne': [50]*len(F_1), 'Algorithm': ['FindsABS']*len(F_1), 'Binary cross-entropy': F_1}
data2 = {'Ne': [50]*len(F_2), 'Algorithm': ['P(Y|do(X), Z, W,  De )']*len(F_2), 'Binary cross-entropy': F_2}
data3 = {'Ne': [50]*len(F_3), 'Algorithm': ['P(Y|do(X), Z, W,  Do* )']*len(F_3), 'Binary cross-entropy': F_3}
data11 = {'Ne': [50]*len(F_11), 'Algorithm': ['P(Y|do(X), Z, W,  De, Do* )']*len(F_11), 'Binary cross-entropy': F_11}

# Create DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df11 = pd.DataFrame(data11)

# For Ne = 100
F_4, F_5, F_6, F_44  = data_preds(file_path_100)
print(len(F_4))
data4 = {'Ne': [100]*len(F_4), 'Algorithm': ['FindsABS']*len(F_4), 'Binary cross-entropy': F_4}
data5 = {'Ne': [100]*len(F_5), 'Algorithm': ['P(Y|do(X), Z, W,  De )']*len(F_5), 'Binary cross-entropy': F_5}
data6 = {'Ne': [100]*len(F_6), 'Algorithm': ['P(Y|do(X), Z, W,  Do* )']*len(F_6), 'Binary cross-entropy': F_6}
data44 = {'Ne': [100]*len(F_44), 'Algorithm': ['P(Y|do(X), Z, W,  De, Do* )']*len(F_44), 'Binary cross-entropy': F_44}

# Create DataFrame
df4 = pd.DataFrame(data4)
df5 = pd.DataFrame(data5)
df6 = pd.DataFrame(data6)
df44 = pd.DataFrame(data44)

# For Ne = 300
F_7, F_8, F_9, F_77  = data_preds(file_path_300)

data7 = {'Ne': [300]*len(F_7), 'Algorithm': ['FindsABS']*len(F_7), 'Binary cross-entropy': F_7}
data8 = {'Ne': [300]*len(F_8), 'Algorithm': ['P(Y|do(X), Z, W,  De )']*len(F_8), 'Binary cross-entropy': F_8}
data9 = {'Ne': [300]*len(F_9), 'Algorithm': ['P(Y|do(X), Z, W,  Do* )']*len(F_9), 'Binary cross-entropy': F_9}
data77 = {'Ne': [300]*len(F_77), 'Algorithm': ['P(Y|do(X), Z, W,  De, Do* )']*len(F_77), 'Binary cross-entropy': F_77}

# Create DataFrame
df7 = pd.DataFrame(data7)
df8 = pd.DataFrame(data8)
df9 = pd.DataFrame(data9)
df77 = pd.DataFrame(data77)


data = pd.concat([df1, df2, df3, df11, df4, df5, df6, df44, df7, df8, df9, df77], ignore_index=True)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# Function to ensure FindsABS is first
alg_order = ['P(Y|do(X), Z, W,  De, Do* )', 'P(Y|do(X), Z, W,  De )', 'P(Y|do(X), Z, W,  Do* )', 'FindsABS']

data['Algorithm'] = pd.Categorical(data['Algorithm'], categories=alg_order, ordered=True)
data = data.sort_values(['Ne', 'Algorithm'])  # Ensuring order

# Compute summary statistics
summary_stats = data.groupby(['Ne', 'Algorithm']).agg(
    mean_mse=('Binary cross-entropy', 'mean'),
    std_mse=('Binary cross-entropy', 'std'),
    count=('Binary cross-entropy', 'count')
).reset_index()
summary_stats['sem_mse'] = summary_stats['std_mse'] / np.sqrt(summary_stats['count'])
summary_stats['ci95'] = 1.96 * summary_stats['sem_mse']  # 95% CI

# Define a high-contrast color palette
palette = sns.color_palette('tab10', n_colors=len(alg_order))
custom_colors = {alg: palette[i] for i, alg in enumerate(alg_order)}

# Define marker styles
marker_styles = {'FindsABS': 'o', 'P(Y|do(X), Z, W,  De )': 's', 'P(Y|do(X), Z, W,  Do* )': 'D', 'P(Y|do(X), Z, W,  De, Do* )': '^'}

# Plot error bars
fig, ax = plt.subplots(figsize=(8, 6))
for i, algorithm in enumerate(alg_order):
    subset = summary_stats[summary_stats['Algorithm'] == algorithm]
    ax.errorbar(subset['Ne'], subset['mean_mse'], yerr=subset['ci95'],
                fmt=marker_styles[algorithm] + '-', capsize=5, label=algorithm, color=custom_colors[algorithm],
                linewidth=2.5 if algorithm == 'P(Y|do(X), Z, W,  De, Do* )' else 2.5, markersize=6 if algorithm == 'FindsABS' else 6)

ax.set_xlabel('Ne', fontsize=30)
ax.set_ylabel('Binary cross-entropy', fontsize=30)
ax.legend(title='', fontsize=22, title_fontsize=22, loc='best')
ax.tick_params(axis='both', labelsize=22)

# Set the y-axis ticks
ax.set_yticks(np.arange(0.4, 0.8, 0.1))  # Adjust the range and step size accordingly

plt.show()
