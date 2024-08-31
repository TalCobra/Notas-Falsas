# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:55:01 2024

@author: Matheus Henrique Dizaro Moyamoto
"""
# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
from sklearn.metrics import roc_curve, auc #Curva ROC
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score #Matriz de confusão

#%%
dt_fakebill = pd.read_csv('fake_bills.csv', delimiter=';')

dt_fakebill.info()
dt_fakebill.dropna(inplace=True)

dt_fakebill.loc[dt_fakebill['is_genuine']=='True', 'is_genuine'] = 1
dt_fakebill.loc[dt_fakebill['is_genuine']=='False', 'is_genuine'] = 0
dt_fakebill['is_genuine'] = dt_fakebill['is_genuine'].astype('int64')

dt_fakebill.describe()
#%%
lista_colunas = list(dt_fakebill.drop(columns=['is_genuine']).columns)
formula_modelo_fake = ' + '.join(lista_colunas)
formula_modelo_fake = "is_genuine ~ " + formula_modelo_fake

modelo_fake = sm.Logit.from_formula(formula_modelo_fake, dt_fakebill).fit()

modelo_fake.summary()
#%% Stepwise

step_modelo_fake = stepwise(modelo_fake, pvalue_limit=0.05)
#%% Matriz de confusão

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
    plt.savefig(f'confusao_{cutoff}.png')
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores
#%%

dt_fakebill['phat'] = step_modelo_fake.predict()

matriz_confusao(observado=dt_fakebill['is_genuine'],
                predicts=dt_fakebill['phat'],
                cutoff=0.5)

matriz_confusao(observado=dt_fakebill['is_genuine'],
                predicts=dt_fakebill['phat'],
                cutoff=0.8)
#%% Função para construção  de um grafico com diferentes valores de cutoff
def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado
#%%
dados_plotagem = espec_sens(observado = dt_fakebill['is_genuine'],
                            predicts = dt_fakebill['phat'])

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
         color='limegreen', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.show()
plt.savefig('Grafico_espec_sens.png')
#%% Curva ROC

fpr, tpr, thresholds =roc_curve(dt_fakebill['is_genuine'],
                                dt_fakebill['phat'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()
plt.savefig('Grafico_ROC.png')

#%% Realizando uma predição

modelo_fake.predict(pd.DataFrame({'diagonal':[169.88], 
                                     'height_left':[102.50],
                                     'height_right':[104.33],
                                     'margin_low':[4],
                                     'margin_up':[3.22],
                                     'length':[114.10]}))