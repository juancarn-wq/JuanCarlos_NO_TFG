
"""
@author: Juan Carlos Navarro
"""

import numpy as np
import os 
from scipy.stats import pearsonr
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



rutall = r"Ruta a la caerpeta con los archivos"
rutapl = r"Ruta a la carpeta con los archivos"

archivosLL = [f for f in os.listdir(rutall) if f.endswith('.npz')]
archivosPL = [f for f in os.listdir(rutapl) if f.endswith('.npz')]


consistenciasLL = []
nLL = []
pLL = []
tLL = []
for archivo in archivosLL:
    filename_laliga = os.path.join(rutall, archivo)
    
    datal = np.load(filename_laliga, allow_pickle=True)

    tensorL = datal['Tensor']
    nameL = str(datal['Player_Name'][0])
    positionL = str(datal['Position'])
    pLL.append(positionL)
    teamL = str(datal['Team'][0])
    tLL.append(teamL)
    nLL.append(len(tensorL))
    n = np.shape(tensorL)[0]
    r = 2
    combinaciones = itertools.combinations(range(n), r) 
    correlaciones = []
    for i, j in combinaciones:
        vectori = tensorL[i].flatten()
        vectorj = tensorL[j].flatten()
        correlacion, _ = pearsonr(vectori, vectorj)
        correlaciones.append(correlacion)
    consistenciasLL.append(np.mean(correlaciones))
print(np.mean(consistenciasLL))

print(len(consistenciasLL))
print(len(nLL))


datos = pd.DataFrame({'Games Played': nLL, 'Consistency Coef.': consistenciasLL}) #para trabajar con datos es lo mejor, pasarlo a un objeto tipo tabular de python que son los DataFrame, super utiles en explotacion y mineria de datos

print(datos.head())

#%%
sns.scatterplot(x='Games Played', y='Consistency Coef.', data=datos)

plt.xlabel('n')
plt.ylabel('C')
plt.text(
    x=datos['Games Played'].max() * 0.7,
    y=datos['Consistency Coef.'].max() * 0.95,
    s='LL',
    fontsize=12,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')  
)

plt.show()
#%%


consistenciasPL = []
nPL = []
pPL = []
tPL = []
for archivoP in archivosPL:
    filename_premier = os.path.join(rutapl, archivoP)
    datap = np.load(filename_premier, allow_pickle=True)

    tensorP = datap['Tensor']
    nameP = str(datap['Player_Name'][0])
    positionP = str(datap['Position'])
    pPL.append(positionP)
    teamP = str(datap['Team'][0])
    tPL.append(teamP)
    nPL.append(len(tensorP))
    n = np.shape(tensorP)[0]
    r = 2
    combinaciones = itertools.combinations(range(n), r) 
    correlacionesP = []
    for i, j in combinaciones:
        vectork = tensorP[i].flatten()
        vectorl = tensorP[j].flatten()
        correlacion, _ = pearsonr(vectork, vectorl)
        correlacionesP.append(correlacion)
    consistenciasPL.append(np.mean(correlacionesP))
print(np.mean(consistenciasPL))


print(len(consistenciasPL))
print(len(nPL))

consistencia_min = min(min(consistenciasLL), min(consistenciasPL))
consistencia_max = max(max(consistenciasLL), max(consistenciasPL))

datosP = pd.DataFrame({'Games Played': nPL, 'Consistency Coef.': consistenciasPL}) #para trabajar con datos es lo mejor, pasarlo a un objeto tipo tabular de python que son los DataFrame, super utiles en explotacion y mineria de datos

print(datosP.head()) 
#%%
plt.figure()
sns.scatterplot(x='Games Played', y='Consistency Coef.', data=datosP)

plt.xlabel('n')
plt.ylabel('C')
plt.text(
    x=datos['Games Played'].max() * 0.65,  
    y=datos['Consistency Coef.'].max() * 0.95,  
    s='PL',
    fontsize=12,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')  # fondo blanco, contorno negro
)

plt.show()

#%% Cajas y Bigotes Equipo - La Liga (ordenado) 1
plt.figure()
df_LL = pd.DataFrame({'Equipo': tLL, 'Consistencia': consistenciasLL})
orden_LL = df_LL.groupby('Equipo')['Consistencia'].median().sort_values(ascending=False).index
sns.boxplot(x='Equipo', y='Consistencia', data=df_LL, order=orden_LL)
plt.xticks(rotation=90)
plt.xlabel('Equipo')
plt.ylabel('C')

plt.text(
    x=0.7,  
    y=2 * df_LL['Consistencia'].max(),  
    s='LL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)

plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% Cajas y Bigotes Equipo - Premier League (ordenado) 2
plt.figure()
df_PL = pd.DataFrame({'Equipo': tPL, 'Consistencia': consistenciasPL})
orden_PL = df_PL.groupby('Equipo')['Consistencia'].median().sort_values(ascending=False).index


sns.boxplot(x='Equipo', y='Consistencia', data=df_PL, order=orden_PL)
plt.xticks(rotation=90)
plt.xlabel('Equipo')
plt.ylabel('C')
plt.text(
    x=0.75,  
    y=2 * df_LL['Consistencia'].max(),  
    s='PL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% Cajas y Bigotes Posición - La Liga (ordenado) 3
plt.figure()
df_LL_pos = pd.DataFrame({'Posición': pLL, 'Consistencia': consistenciasLL})
orden_LL_pos = df_LL_pos.groupby('Posición')['Consistencia'].median().sort_values(ascending=False).index


sns.boxplot(x='Posición', y='Consistencia', data=df_LL_pos, order=orden_LL_pos)
plt.xticks(rotation=90)
plt.xlabel('Posición')
plt.ylabel('C')
plt.text(
    x=0.6,  
    y=2 * df_LL['Consistencia'].max(),  
    s='LL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% Cajas y Bigotes Posición - Premier League (ordenado) 4
plt.figure()
df_PL_pos = pd.DataFrame({'Posición': pPL, 'Consistencia': consistenciasPL})
orden_PL_pos = df_PL_pos.groupby('Posición')['Consistencia'].median().sort_values(ascending=False).index


sns.boxplot(x='Posición', y='Consistencia', data=df_PL_pos, order=orden_PL_pos)
plt.xticks(rotation=90)
plt.xlabel('Posición')
plt.ylabel('C')
plt.text(
    x=0.6,  
    y=2 * df_LL['Consistencia'].max(),  
    s='PL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()
#%% IQR LL Equipos 5
plt.figure()
df_LL = pd.DataFrame({'Equipo': tLL, 'Consistencia': consistenciasLL})

iqr_por_equipo = df_LL.groupby('Equipo')['Consistencia'].quantile([0.75, 0.25]).unstack()
iqr_por_equipo['IQR'] = iqr_por_equipo[0.75] - iqr_por_equipo[0.25]

orden_LL = iqr_por_equipo.sort_values('IQR', ascending=False).index


sns.boxplot(x='Equipo', y='Consistencia', data=df_LL, order=orden_LL)
plt.xticks(rotation=90)
plt.xlabel('Equipo')
plt.ylabel('C')

plt.text(
    x=0.6,
    y=0.8,
    s='LL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes
)

plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% IQR PL Equipos 6
plt.figure()
df_PL = pd.DataFrame({'Equipo': tPL, 'Consistencia': consistenciasPL})


iqr_por_equipo = df_PL.groupby('Equipo')['Consistencia'].quantile([0.75, 0.25]).unstack()
iqr_por_equipo['IQR'] = iqr_por_equipo[0.75] - iqr_por_equipo[0.25]


orden_PL = iqr_por_equipo.sort_values('IQR', ascending=False).index


sns.boxplot(x='Equipo', y='Consistencia', data=df_PL, order=orden_PL)
plt.xticks(rotation=90)
plt.xlabel('Equipo')
plt.ylabel('C')


plt.text(
    x=0.85,
    y=0.8,
    s='PL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes
)

plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()
#%% IQR LL Posiciones 7
plt.figure()

df_LL_pos = pd.DataFrame({'Posición': pLL, 'Consistencia': consistenciasLL})


iqr_por_pos = df_LL_pos.groupby('Posición')['Consistencia'].quantile([0.75, 0.25]).unstack()
iqr_por_pos['IQR'] = iqr_por_pos[0.75] - iqr_por_pos[0.25]


orden_LL_pos = iqr_por_pos.sort_values('IQR', ascending=False).index


sns.boxplot(x='Posición', y='Consistencia', data=df_LL_pos, order=orden_LL_pos)


plt.xticks(rotation=90)
plt.xlabel('Posición')
plt.ylabel('C')


plt.text(
    x=0.7,
    y=0.7,
    s='LL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()
#%% IQR PL Posiciones 8
plt.figure()
df_PL_pos = pd.DataFrame({'Posición': pPL, 'Consistencia': consistenciasPL})


iqr_por_pos = df_PL_pos.groupby('Posición')['Consistencia'].quantile([0.75, 0.25]).unstack()
iqr_por_pos['IQR'] = iqr_por_pos[0.75] - iqr_por_pos[0.25]


orden_PL_pos = iqr_por_pos.sort_values('IQR', ascending=False).index


sns.boxplot(x='Posición', y='Consistencia', data=df_PL_pos, order=orden_PL_pos)


plt.xticks(rotation=90)
plt.xlabel('Posición')
plt.ylabel('C')


plt.text(
    x=0.6,
    y=0.8,
    s='PL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes
)

plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()
#%% Cajas y Bigotes Equipo - La Liga (ordenado media) 9
plt.figure()
df_LL = pd.DataFrame({'Equipo': tLL, 'Consistencia': consistenciasLL})
orden_LL = df_LL.groupby('Equipo')['Consistencia'].mean().sort_values(ascending=False).index
sns.boxplot(x='Equipo', y='Consistencia', data=df_LL, order=orden_LL)
plt.xticks(rotation=90)
plt.xlabel('Equipo')
plt.ylabel('C')

plt.text(
    x=0.6,  
    y=2 * df_LL['Consistencia'].max(),  
    s='LL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)

plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% Cajas y Bigotes Equipo - Premier League (ordenado media) 10
plt.figure()
df_PL = pd.DataFrame({'Equipo': tPL, 'Consistencia': consistenciasPL})
orden_PL = df_PL.groupby('Equipo')['Consistencia'].mean().sort_values(ascending=False).index


sns.boxplot(x='Equipo', y='Consistencia', data=df_PL, order=orden_PL)
plt.xticks(rotation=90)
plt.xlabel('Equipo')
plt.ylabel('C')
plt.text(
    x=0.75,  
    y=2* df_LL['Consistencia'].max(),  
    s='PL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  # usa el sistema de coordenadas del gráfico
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% Cajas y Bigotes Posición - La Liga (ordenado media) 11
plt.figure()
df_LL_pos = pd.DataFrame({'Posición': pLL, 'Consistencia': consistenciasLL})
orden_LL_pos = df_LL_pos.groupby('Posición')['Consistencia'].mean().sort_values(ascending=False).index


sns.boxplot(x='Posición', y='Consistencia', data=df_LL_pos, order=orden_LL_pos)
plt.xticks(rotation=90)
plt.xlabel('Posición')
plt.ylabel('C')
plt.text(
    x=0.6,  
    y=2* df_LL['Consistencia'].max(),  
    s='LL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%% Cajas y Bigotes Posición - Premier League (ordenado media) 12
plt.figure()
df_PL_pos = pd.DataFrame({'Posición': pPL, 'Consistencia': consistenciasPL})
orden_PL_pos = df_PL_pos.groupby('Posición')['Consistencia'].mean().sort_values(ascending=False).index


sns.boxplot(x='Posición', y='Consistencia', data=df_PL_pos, order=orden_PL_pos)
plt.xticks(rotation=90)
plt.xlabel('Posición')
plt.ylabel('C')
plt.text(
    x=0.6,  
    y=2 * df_LL['Consistencia'].max(),  
    s='PL',
    ha='center',
    va='bottom',
    fontsize=20,
    weight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    transform=plt.gca().transAxes  
)
plt.tight_layout()
plt.ylim(consistencia_min, consistencia_max)
plt.show()

#%%
def agrupar_posicion(pos):
    pos = pos.lower()
    if 'keeper' in pos:
        return 'Portero'
    elif 'defend' in pos or 'back' in pos:
        return 'Defensa'
    elif 'mid' in pos:
        return 'Centrocampista'
    elif 'forward' in pos or 'striker' in pos or 'wing' in pos:
        return 'Delantero'
    else:
        return 'Otro'

posiciones_LL = [agrupar_posicion(p) for p in pLL]
posiciones_PL = [agrupar_posicion(p) for p in pPL]

df_LL = pd.DataFrame({
    'Posición': posiciones_LL,
    'Consistency Coef.': consistenciasLL
})

df_PL = pd.DataFrame({
    'Posición': posiciones_PL,
    'Consistency Coef.': consistenciasPL
})

df_LL = df_LL[df_LL['Posición'].isin(['Portero', 'Defensa', 'Centrocampista', 'Delantero'])]
df_PL = df_PL[df_PL['Posición'].isin(['Portero', 'Defensa', 'Centrocampista', 'Delantero'])]

plt.figure()

df_LL['Liga'] = 'LL'
df_PL['Liga'] = 'PL'
df_comb = pd.concat([df_LL, df_PL], ignore_index=True)

orden_posiciones = ['Portero', 'Defensa', 'Centrocampista', 'Delantero']
df_comb['Posición'] = pd.Categorical(df_comb['Posición'], categories=orden_posiciones, ordered=True)

sns.boxplot(
    x='Posición', 
    y='Consistency Coef.', 
    hue='Liga', 
    data=df_comb, 
    palette={'LL': 'skyblue', 'PL': 'lightgreen'}
)

plt.xlabel('Posición')
plt.ylabel('C')
plt.ylim(consistencia_min, consistencia_max)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#%%
plt.figure()

sns.ecdfplot(consistenciasLL, color='blue', label='LL')
sns.ecdfplot(consistenciasPL, color='green', label='PL')

plt.xlabel('C')
plt.ylabel('P(x≤C)')

plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
#%% Función de densidad de probabilidad - La Liga y Premier League en un solo gráfico
plt.figure()

sns.kdeplot(consistenciasLL, fill=True, color='blue', label='LL')
sns.kdeplot(consistenciasPL, fill=True, color='green', label='PL')

plt.xlabel('C')
plt.ylabel('P(x=C)')

plt.legend(loc='upper right')

plt.tight_layout()

plt.show()
