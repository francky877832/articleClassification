import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('unified_brute_dataset.csv')

# 1. Class Distribution (Bar plot of the categories)
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Class Distribution (Article Categories)')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Class Proportions with a Pie Chart
class_proportions = df['label'].value_counts(normalize=True)

# Plot the class distribution in a pie chart
plt.figure(figsize=(8, 8))
class_proportions.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2', len(class_proportions)), startangle=90)
plt.title('Class Distribution (Proportions)')
plt.ylabel('')  # Remove the 'y' label
plt.tight_layout()
plt.show()

# 3. Text Feature Extraction for Correlation
# Extract numeric features from the text columns
# For example: word count, average word length in the summary and title

df['summary_word_count'] = df['summary'].apply(lambda x: len(str(x).split()))
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

df['summary_avg_word_length'] = df['summary'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)
df['title_avg_word_length'] = df['title'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# Now, let's calculate the correlation matrix for these new numeric columns
numeric_columns = df[['summary_word_count', 'title_word_count', 'summary_avg_word_length', 'title_avg_word_length']]

# Compute correlation matrix
corr_matrix = numeric_columns.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - Text Features (Summary & Title)')
plt.tight_layout()
plt.show()
