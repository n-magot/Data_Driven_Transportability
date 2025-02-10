from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve, auc
import csv
import plotly.graph_objects as go

# File path
file_path_50 = r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\random_complete\AUC_random_complete_results_N50.csv'
file_path_100 = r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\random_complete\AUC_random_complete_results_N100.csv'
file_path_300 = r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\random_complete\AUC_random_complete_results_N300.csv'

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
first_column, second_column, third_column, forth_column = data_preds(file_path_50)
y_true_50 =  [0] * 100 + [0] * 100 + [0] * 100 + [1] * 100
y_score_50 = first_column + second_column + third_column + forth_column

# For Ne = 100
first_column, second_column, third_column, forth_column = data_preds(file_path_100)
y_true_100 =  [0] * 100 + [0] * 100 + [0] * 100 + [1] * 100
y_score_100 = first_column + second_column + third_column + forth_column

# For Ne = 300
first_column, second_column, third_column, forth_column = data_preds(file_path_300)
y_true_300 =  [0] * 100 + [0] * 100 + [0] * 100 + [1] * 100
y_score_300 = first_column + second_column + third_column + forth_column

auc_50 = roc_auc_score(y_true_50, y_score_50)
auc_100 = roc_auc_score(y_true_100, y_score_100)
auc_300 = roc_auc_score(y_true_300, y_score_300)

"""ROC with confidence intervals"""
# For Ne = 50
first_column, second_column, third_column, forth_column = data_preds(file_path_100)
true_labels = [0] * 100 + [0] * 100 + [0] * 100 + [1] * 100
pred_prob = first_column + second_column + third_column + forth_column


y_test = np.array(true_labels)
y_pred_prob = np.array(pred_prob)
# Step 3: Compute the initial ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Step 2: Generate bootstrapped confidence intervals for the ROC curve
n_bootstraps = 1000
rng = np.random.RandomState(42)

tprs_bootstrap = []
aucs_bootstrap = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(n_bootstraps):
    # Sample with replacement
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:  # Ensure both classes are present
        continue

    y_test_boot = y_test[indices]
    y_pred_prob_boot = y_pred_prob[indices]

    # Compute ROC and AUC
    fpr_boot, tpr_boot, _ = roc_curve(y_test_boot, y_pred_prob_boot)
    tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
    tprs_bootstrap.append(tpr_interp)
    aucs_bootstrap.append(auc(fpr_boot, tpr_boot))

tprs_bootstrap = np.array(tprs_bootstrap)
mean_tpr = tprs_bootstrap.mean(axis=0)
mean_auc = np.mean(aucs_bootstrap)

# Confidence intervals for AUC using quantiles
alpha = 0.95
lower_auc = np.percentile(aucs_bootstrap, (1 - alpha) / 2 * 100)
upper_auc = np.percentile(aucs_bootstrap, (1 + alpha) / 2 * 100)

# Confidence intervals for TPR using quantiles
lower_tpr = np.percentile(tprs_bootstrap, (1 - alpha) / 2 * 100, axis=0)
upper_tpr = np.percentile(tprs_bootstrap, (1 + alpha) / 2 * 100, axis=0)


# Step 4: Plot the ROC curve with Plotly
fig = go.Figure()

# Add mean ROC curve
fig.add_trace(
    go.Scatter(
        x=mean_fpr,
        y=mean_tpr,
        mode='lines',
        name="Mean ROC",
        line=dict(color='SteelBlue', width=2),
    )
)

# Add individual ROC curve
fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f"ROC (AUC = {roc_auc:.3f})",
        line=dict(color='navy', width=2),
    )
)

# Add confidence interval
fig.add_trace(
    go.Scatter(
        x=np.concatenate([mean_fpr, mean_fpr[::-1]]),
        y=np.concatenate([upper_tpr, lower_tpr[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f"95% CI =[{lower_auc:.3f}, {upper_auc:.3f}]",
    )
)

# Add baseline
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Baseline',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    )
)

# Layout adjustments with custom font settings and centered title
fig.update_layout(
    title="Ne=100",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.45, y=0.1),
    template='plotly_white',
    width=800,
    height=600,
    title_font=dict(size=34, color='black', family='Times New Roman'),  # Title font settings
    xaxis_title_font=dict(size=32, color='black', family='Times New Roman'),  # X-axis title font settings
    yaxis_title_font=dict(size=32, color='black', family='Times New Roman'),  # Y-axis title font settings
    legend_font=dict(size=32, color='black', family='Times New Roman'),  # Legend font settings
    font=dict(size=32, color='black', family='Times New Roman'),  # General font settings for tick labels and other text
    title_x=0.5  # Centers the title
)

# Show plot
fig.show()
