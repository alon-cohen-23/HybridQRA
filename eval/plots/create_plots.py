import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dic = {}

metrics = [
    'answer_correctness',
    'answer_relevancy',
]

cohere_df = pd.read_csv('/Users/aloncohen/Documents/HybridQRA/data/testsest/command-r-plus-08-2024_answers_results.csv')
command_r_scores = cohere_df[metrics].mean().to_dict()
dic['command-r-plus-08-2024'] = command_r_scores

azure_openai_df = pd.read_csv('/Users/aloncohen/Documents/HybridQRA/data/testsest/gpt-4o-sim_answers_results.csv')
gpt_scores = azure_openai_df[metrics].mean().to_dict()
dic['gpt-4o-sim'] = gpt_scores

models = list(dic.keys())

# Prepare data for plotting
scores = {
    model: [dic[model][metric] for metric in metrics]
    for model in models
}

x = np.arange(len(metrics))  # Positions for the metrics
width = 0.25  # Bar width

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Plot bars for each model
bars1 = ax.bar(x - width / 2, scores['command-r-plus-08-2024'], width, label='command-r-plus-08-2024', color='teal', edgecolor='black')
bars2 = ax.bar(x + width / 2, scores['gpt-4o-sim'], width, label='gpt-4o-sim', color='lightgreen', edgecolor='black')

# Add labels to bars
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,  # Center text horizontally
                bar.get_height() + 0.01,           # Slightly above the bar
                f'{bar.get_height():.2f}',         # Format score to 2 decimal places
                ha='center', va='bottom', fontsize=10)

# Add labels, title, and legend
ax.set_title('Model Comparison on Answer Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.set_ylabel('Scores', fontsize=14, fontweight='bold', labelpad=15)
ax.set_xlabel('Metrics', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylim(0, 1.1)  # Ensure scores are visible
ax.legend(fontsize=12, loc='upper left')

# Style adjustments
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Save and display the plot
plt.savefig('model_comparison_bar_plot.png', dpi=300, bbox_inches='tight')
plt.show()


metrics = [
    'context_precision',
    'context_recall',
    'context_relevancy'
]

df = pd.read_csv('/Users/aloncohen/Documents/HybridQRA/data/testsest/command-r-plus-08-2024_answers_results.csv')

scores = df[metrics].mean().to_list()

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 8))

# Adjust the width of the bars
bar_width = 0.5  # Make the bars narrower
bars = ax.bar(metrics, scores, color='skyblue', edgecolor='black', width=bar_width)

# Add labels to the bars
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,  # Center the label horizontally
            score + 0.02,                      # Slightly above the bar
            f'{score:.2f}', va='bottom', ha='center', fontsize=12)

# Title and labels
ax.set_title('ESPN HybridQRA Results', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Metrics', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylim(0, 1.1)  # Extend slightly beyond 1 for visibility

# Style adjustments
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Save the plot
plt.savefig('espn_rag_results_vertical.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()


