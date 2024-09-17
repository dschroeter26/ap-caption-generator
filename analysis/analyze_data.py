import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Load the processed data
captions_df = pd.read_csv('./../data/captions_processed.csv')

# Word frequencies
all_words = ' '.join(captions_df['caption']).split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(20)

# Plot most common words
plt.figure(figsize=(10, 6))
plt.bar(*zip(*most_common_words))
plt.xticks(rotation=45)
plt.title('Top 20 Most Common Words in Captions')
plt.show()

# Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
