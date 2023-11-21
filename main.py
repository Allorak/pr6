import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import silhouette_score
import plotly.io as pio

pio.renderers.default = "iframe"

"""
Найти данные для кластеризации. Данные в группе не должны
повторяться. Если признаки в данных имеют очень сильно разные
масштабы, то необходимо данные предварительно нормализовать
"""


def map_column(dataframe: pd.DataFrame, column_name: str):
    if is_numeric_dtype(dataframe[column_name]):
        return

    values = dataframe[column_name].unique()
    indexed_values = {values[index]: index for index in range(len(values))}

    dataframe[column_name].replace(indexed_values, inplace=True)


df = pd.read_csv(Path('vgsales.csv'))
print(df.info())

irrelevant_columns = ['Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Name', 'Global_Sales', 'Year']
relevant_columns = ['Platform', 'Genre', 'Publisher']

df.drop(irrelevant_columns, axis=1, inplace=True)
df.dropna(inplace=True)
for column_name in relevant_columns:
    map_column(df, column_name)
print(df.head())

"""
Провести кластеризацию данных с помощью алгоритма k-means.
Использовать «правило локтя» и коэффициент силуэта для поиска
оптимального количества кластеров.
"""

random_seed = 7213

k_means_models = []
costs = []
silhouettes = []

for i in range(2, 15):
    print(f'Текущий параметр k: {i}')
    k_mean_model = KMeans(n_clusters=i, random_state=random_seed, init='k-means++', n_init=10)
    k_mean_model.fit(df)
    k_means_models.append(k_mean_model)
    costs.append(k_mean_model.inertia_)
    silhouettes.append(silhouette_score(df, k_mean_model.labels_))

figure, plots = plt.subplots(1, 2)
figure.set_size_inches(12, 12)

plots[0].grid()
plots[0].set_title('Правило локтя')
plots[0].plot(np.arange(2, 15), costs, marker='o')

plots[1].grid()
plots[1].set_title('Коэффициент силуэта')
plots[1].plot(np.arange(2, 15), silhouettes, marker='o')
plt.show()

best_k_index = silhouettes.index(max(silhouettes))
best_k = best_k_index + 2
best_model = k_means_models[best_k_index]
print(f'Лучший показатель k: {best_k}')
print(f'Положения центроидов: {best_model.cluster_centers_}')

df['Cluster'] = best_model.labels_

fig = go.Figure(data=[go.Scatter3d(x=df[relevant_columns[0]], y=df[relevant_columns[1]], z=df[relevant_columns[2]],
                                   mode='markers', marker_color=df['Cluster'], marker_size=4)])
fig.show()
input()
"""
Провести кластеризацию данных с помощью алгоритма иерархической
кластеризации
"""

agglomerative_model = AgglomerativeClustering(best_k, compute_distances=True)
agglomerative_model.fit(df)
df['Cluster'] = agglomerative_model.labels_

fig = go.Figure(data=[go.Scatter3d(x=df[relevant_columns[0]], y=df[relevant_columns[1]], z=df[relevant_columns[2]],
                                   mode='markers', marker_color=df['Cluster'], marker_size=4)])
fig.show()
input()
"""
Провести кластеризацию данных с помощью алгоритма DBSCAN.
"""

dbscan_model = DBSCAN(eps=11, min_samples=5)
dbscan_model.fit(df)
df['Cluster'] = dbscan_model.labels_

fig = go.Figure(data=[go.Scatter3d(x=df[relevant_columns[0]], y=df[relevant_columns[1]], z=df[relevant_columns[2]],
                                   mode='markers', marker={"color": df["Cluster"],
                                                           "colorscale": 'thermal',
                                                           "showscale": True})])
fig.show()
