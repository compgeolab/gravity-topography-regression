import pygmt
import xarray as xr
import ensaio
import boule as bl
import harmonica as hm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy.stats
import bordado as bd
import verde as vd
import pandas as pd

def robust_regression(h,d):
    if len(h) < 10:
        print('Vetor de topografia com tamanho menor que 20')
        return np.nan, np.nan, np.nan

    # entra pra lista qualquer um que cumprir o critério
    has_ocean = np.any(h < 0)
    has_continent = np.any( h >= 0)

    if has_ocean and has_continent: # se 'any' retornar True pra tudo em ambos os casos
        # criando matriz A, com os três parÂmetros
        h_ocean = np.where(h < 0, h, 0)
        h_continent = np.where(h >= 0, h, 0)
        intercept = np.ones(len(h))
        A = np.column_stack((h_ocean, h_continent, intercept))
        case = 'mixed'

    elif has_ocean: # se 'any' retornar False para has_continent
        h_ocean = np.where(h < 0, h, 0)
        intercept = np.ones(len(h))
        A = np.column_stack((h_ocean, intercept))
        case = 'ocean'

    elif has_continent: # se'any' retornar False para has_ocean
        h_continent = np.where( h >= 0, h, 0) 
        intercept = np.ones(len(h))
        A = np.column_stack((h_continent, intercept))
        case = 'continent'

    else:
        return np.nan, np.nan, np.nan

    try:
    
        # resolvendo o sistema Ap = d, rodando regressão sem robustez
        p_non_robust = np.linalg.lstsq(A, d, rcond=None)[0] # [0] gets only the parameters
        residuals = d - (A @ p_non_robust) # dados observados - dados preditos 
    
        W = 1 / (np.abs(residuals**2) + 1e-5) # vetor pesos
    
        # se eu transformar (d * W) em coluna, o resultado será o mesmo, mas o formato sairá como coluna
        # coluna é [ [x], [y], [z] ], e o resultadom em linha é [x, y, z]
        # como d também é um vetor, não preciso transformar em coluna, facilitando a visualização do resultado
        A_weighted = A * W[:, np.newaxis]
        A_T_times_A_weighted = A.T @ A_weighted
        A_T_times_d_weighted = A.T @ (W * d)
        # posso uar np.linalg.solve pois aqui trabalho com matriz quadradas 3x3 
        p_robust = np.linalg.solve(A_T_times_A_weighted, A_T_times_d_weighted)


        if case == 'mixed':
            return p_robust[0], p_robust[1], p_robust[2]

        if case == 'ocean':
            return p_robust[0], np.nan, p_robust[1] # a matriz nesse caso só tem duas colunas

        if case == 'continent':
            return np.nan, p_robust[0], p_robust[1] # a matriz nesse caso só tem duas colunas
        

    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

def plot_window_regression(i_wished_window, h_window, d_window, a_ocean_window, a_continent_window, intercept_window):
    h_limit = np.abs(h_window).max()

    plt.figure(figsize=(10,6))
    ax = plt.gca() # necessário para posicionar a caixa de texto
    plt.xlim(-h_limit, h_limit)

    plt.scatter(h_window, d_window, color='black', alpha=0.2, s=10)

    # plot para oceano, caso a_o exista
    if not np.isnan(a_ocean_window):
        if np.any(h_window < 0): # se retornar True é porque ao menos 1 ponto corresponde a condição
            h_ocean = np.sort(np.append(h_window[h_window < 0], 0)) # para retas se encontrarem em 0, adiciono 0 a h_ocean, e organizo pro plot não bugar
            d_pred_ocean = a_ocean_window * h_ocean + intercept_window # cálculo: d = a_o * h + b, d = Ap
            plt.plot(h_ocean, d_pred_ocean, color='blue', lw=2.5, 
                     label=f'Ocean: {a_ocean_window:.5f} mGal/m')

    # plot_continente
    if not np.isnan(a_continent_window):
        if np.any(h_window >= 0):
            h_continent = np.sort(np.append(h_window[h_window >= 0], 0))
            d_pred_continent = a_continent_window * h_continent + intercept_window # cálculo: d = a_o * h + b, d = Ap
            plt.plot(h_continent, d_pred_continent, color='red', lw=2.5, 
                     label=f'Continent: {a_continent_window:.5f} mGal/m')

    plt.title(f'{i_wished_window}th window: Robust Regression')
    plt.xlabel('Topography (m)')
    plt.ylabel('Bouguer anomaly (mGal)')
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5) # linha no nível do mar 

    legenda = plt.legend(loc='upper right')

    # criação da caixinha para o intercepto b abaixo da legenda
    text_b = f'Intercept (b): {intercept_window:.5f} mGal'
    plt.text(0.98, 0.78, text_b, transform=ax.transAxes, 
             horizontalalignment='right', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.show()