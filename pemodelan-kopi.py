import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("D:\KULIAH\SEM 5\ML\dataset\dataset_uts_machine_learning(1).csv")
df.head()
df.info()
df.columns
# Tambahkan kolom Label 
df["Label"] = df["Nama sampel kopi"].apply(lambda x: "Arabika" if "Arabika" in str(x) else "Robusta")
df.sample(3)
### melakukan one hot encoding untuk robusta = 0, arabika = 1
df['Label'] = df["Label"].apply(lambda x: 1 if x == "Arabika" else 0)
df.isnull().sum()
df.duplicated().sum()

plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='Label', palette='viridis')

# tambahkan nilai di atas tiap batang
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2,
            p.get_height(),
            int(p.get_height()),
            ha='center', va='bottom', fontsize=12)

plt.title('perbandingan kopi robusta dan arabika')
plt.xlabel('Status (0 = robusta, 1 = arabika)')
plt.ylabel('Jumlah kopi')
plt.show()

# corelation matrix
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriks Korelasi antar Fitur')
plt.show()

X = df.drop(columns=['Nama sampel kopi', 'Label', 'No Urut'])
y = df['Label']
X_pca = df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
pca = PCA(n_components=2)  # Ubah ke 3 kalau ingin lihat 3D
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
explained_variance_ratio = pca.explained_variance_ratio_
# if 'cluster' in df.columns:
#     pca_df['Cluster'] = df['cluster']
### # B. K-Means Clustering (Menggunakan K=3 untuk klastering alamiah)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels
pca_df['Cluster'] = df['Cluster'].astype(str) # Tambahkan label klaster ke df PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=80)
plt.title('Score Plot PCA - Hasil K-Means Clustering (K=3)')
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]*100:.2f}%)')
plt.legend(title='Klaster K-Means')
plt.grid(True)
plt.savefig('pca_score_plot_kmeans.png')
plt.show()
# K-Means Clustering
numerik = df.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerik)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
df['Cluster'] = cluster_labels
print("\nHasil Cluster Tiap Baris:")
print(df[['Cluster']].head())
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("\nRata-rata Tiap Cluster:")
print(cluster_summary)
plt.figure(figsize=(6,5))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Visualisasi Klaster K-Means")
plt.xlabel(numerik.columns[0])
plt.ylabel(numerik.columns[1])
plt.show()
# linear regresion
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Akurasi: ", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
labels = model.classes_  # ambil nama kelas

# --- Visualisasi 1: dengan seaborn ---
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Visualisasi Pohon Keputusan (Decision Tree)")
plt.show()
