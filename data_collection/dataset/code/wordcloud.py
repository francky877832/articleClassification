import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('unified_processed_final_dataset.csv')

# Combine all the text from the 'summary_processed' column
all_text = " ".join(df['summary_processed'].dropna())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for All Text')
plt.tight_layout()
plt.show()
