import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_cardio = pd.read_csv('cardio_train.csv', sep=',', index_col=0)

df_cardio.info()
df_cardio.columns

df_cardio.describe()
10798.000000 / 365 #mais novo
23713.000000 / 365 # mais velho

df_cardio.isna().sum()

#ANALISE EXPLORATORIA

#dados numericos
from plotly._subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Box(x=df_cardio['age']/365, name='IDADE'), row=1, col=1)
fig.add_trace(go.Box(x=df_cardio['weight'], name='PESO'), row=2, col=1)
fig.add_trace(go.Box(x=df_cardio['ap_hi'], name='Pressao sanguinea sistolica'), row=3, col=1)
fig.add_trace(go.Box(x=df_cardio['ap_lo']/365, name='Pressao sanguinea diastolica'), row=4, col=1)

fig.update_layout(height=700)
fig.show()

#dados categoricos
from plotly._subplots import make_subplots
fig = make_subplots(rows=2, cols=3)

fig.add_trace(go.Bar(y=df_cardio['gender'].value_counts(), x=['FEMININO', 'MASCULINO'], name='GENERO'), row=1, col=1)
fig.add_trace(go.Bar(y=df_cardio['cholesterol'].value_counts(), x=['NORMAL', 'ACIMA DO NORMAL', 'MUITO ACIMA DO NORMAL'], name='COLESTEROL'), row=1, col=2)
fig.add_trace(go.Bar(y=df_cardio['gluc'].value_counts(), x=['NORMAL', 'ACIMA DO NORMAL', 'MUITO ACIMA DO NORMAL'], name='GLICOSE'), row=1, col=3)
fig.add_trace(go.Bar(y=df_cardio['smoke'].value_counts(), x=['NAO FUMANTE', 'FUMANTE'], name='FUMANTE'), row=2, col=1)
fig.add_trace(go.Bar(y=df_cardio['alco'].value_counts(), x=['NAO ALCOOLATRA', 'ALCOOLATRA'], name='ALCOOLATRA'), row=2, col=2)
fig.add_trace(go.Bar(y=df_cardio['active'].value_counts(), x=['NAO ATIVO', 'ATIVO'], name='ATIVO'), row=2, col=3)


fig.update_layout(template='plotly_dark', height=700)
fig.show()

df_cardio['cardio'].value_counts()
df_cardio.groupby(['active', 'cardio']).count()['id']

#MARCHINE LEARNING

Y = df_cardio['cardio']
X = df_cardio.loc[:, df_cardio.columns != 'cardio']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#treinando o modelo 
from sklearn.ensemble import RandomForestClassifier
ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4)
ml_model.fit(x_train, y_train)

ml_model.predict(x_test.iloc[0].to_frame().transpose())

#avaliação do modelo 
from sklearn.metrics import classification_report, confusion_matrix
predictions = ml_model.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#FEATURE IMPORTANCE
from sklearn.inspection import permutation_importance
result = permutation_importance(ml_model, x_test, y_test, n_repeats=10, n_jobs=2)

sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_test.columns[sorted_idx])
ax.set_title('Permution Importanves (test set)')
fig.tight_layout()
plt.show()

#modelo shap
import shap
explanier = shap.TreeExplainer(ml_model)
shap_values = explanier.shap_values(X)
