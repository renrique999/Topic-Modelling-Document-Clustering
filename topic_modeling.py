# ==========================================
# Topic Modeling using LDA
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

# ==========================================
# Sample News Dataset
# ==========================================

texts = [

    "Artificial intelligence and machine learning are transforming industries",

    "Deep learning improves natural language processing systems",

    "The government passed a new economic policy",

    "Elections and politics affect the economy",

    "Football and cricket are popular sports",

    "Sports tournaments attract many fans"
]

# ==========================================
# Count Vectorization
# ==========================================

vectorizer = CountVectorizer(
    stop_words='english'
)

X = vectorizer.fit_transform(texts)

print("\nVectorization Completed!")

# ==========================================
# LDA Topic Modeling
# ==========================================

lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

print("\nTopics Generated!\n")

# ==========================================
# Display Topics
# ==========================================

words = vectorizer.get_feature_names_out()

for index, topic in enumerate(lda.components_):

    print(f"Topic {index + 1}:")

    print([
        words[i]
        for i in topic.argsort()[-5:]
    ])

    print()

# ==========================================
# K-Means Clustering
# ==========================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

kmeans.fit(X)

labels = kmeans.labels_

print("Cluster Labels:")

print(labels)

# ==========================================
# PCA Visualization
# ==========================================

pca = PCA(n_components=2)

reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(8,5))

plt.scatter(
    reduced[:,0],
    reduced[:,1]
)

plt.title("Document Clustering using PCA")

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.show()