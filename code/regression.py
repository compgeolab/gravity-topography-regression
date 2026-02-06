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
    if len(h) < 5:
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

        # calculando R^2
        d_pred = A @ p_robust # nao faria sentido usar A_weighted pois a análise de R² vai ser sobre dados reais de topografia contidos em A, e não em dados manipulados.
        s_residuals = np.sum((d - d_pred)**2)
        s_mean = np.sum((d - np.mean(d))**2) 
        r2 = 1 - (s_residuals / s_mean) if s_mean != 0 else np.nan


        if case == 'mixed':
            return p_robust[0], p_robust[1], p_robust[2], r2

        if case == 'ocean':
            return p_robust[0], np.nan, p_robust[1], r2 # a matriz nesse caso só tem duas colunas

        if case == 'continent':
            return np.nan, p_robust[0], p_robust[1], r2 # a matriz nesse caso só tem duas colunas
        

    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan


def windows_regression(data, h, d, lon, lat, window_size=3.0, overlap=0.5):
    
    window_coordinates, indices = bd.rolling_window_spherical(
        coordinates=(lon, lat), window_size=window_size, overlap=overlap
    )

    resultados = []
    empty_windows = 0

    for i, idx_window in enumerate(indices):
        h_window = h[idx_window]
        d_window = d[idx_window]

        if len(h_window) < 10:
            empty_windows += 1
            continue

        p_window = robust_regression(h_window, d_window)
        
        n_ocean = np.sum(h_window < 0) # como > retorna True ou False, a soma vai ser o número de 'Trues' em vez da soma dos elementos
        n_cont = np.sum(h_window >= 0)

        resultados.append({
            'original-idx': i,
            'longitude': window_coordinates[0][i],
            'latitude': window_coordinates[1][i],
            'a_o': p_window[0],
            'a_c': p_window[1],
            'b': p_window[2],
            'r2' : p_window[3],
            'total_points': len(h_window),
            'ocean_points': n_ocean,
            'continent_points': n_cont
        })

    print(f" {empty_windows} janelas foram ignoradas por falta de dados.")
    
    return pd.DataFrame(resultados), indices



def plot_window_regression(i_wished_window, h_window, d_window, a_ocean_window, a_continent_window, intercept_window, r2_window):
    h_limit = np.abs(h_window).max()

    plt.figure(figsize=(10,6))
    ax = plt.gca() # necessário para posicionar a caixa de texto
    plt.xlim(-h_limit, h_limit)

    plt.scatter(h_window, d_window, color='black', alpha=0.2, s=20)

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
    plt.text(0.98, 0.85, text_b, transform=ax.transAxes, 
             horizontalalignment='right', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))

    text_r2 = f'R²: {r2_window:.5f}'
    plt.text(0.98, 0.79, text_r2, transform=ax.transAxes, 
             horizontalalignment='right', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.show()


import pygmt

def plot_parameters_map(data, parameter, v_range=[-0.2, 0.2], step=0.01, cmap='polar', reverse=True):
    
    # 1. Limpeza e definição da região
    # Remove NaNs da coluna específica para não dar erro no plot
    df_plot = data.dropna(subset=[parameter])
    
    region = [
        df_plot.longitude.min(), df_plot.longitude.max(),
        df_plot.latitude.min(), df_plot.latitude.max()
    ]
    
    fig = pygmt.Figure()

    # 2. Configuração da escala de cores (CPT)
    # Adicionamos reverse=reverse para controlar a polaridade (ex: Vermelho para oceano/positivo)
    pygmt.makecpt(
        cmap=cmap, 
        series=[v_range[0], v_range[1], step], 
        continuous=True, 
        background=True,
        reverse=reverse
    )

    # 3. Base do mapa (Projeção Equidistante Q)
    fig.basemap(
        region=region, 
        projection='Q15c', 
        frame=["af", f"WSne+t'Mapa de {parameter}'"]
    )

    # 4. Elementos geográficos e dados
    fig.coast(land='lightgray', water='white', shorelines='0.5p,black')

    fig.plot(
        x=df_plot.longitude,
        y=df_plot.latitude, 
        fill=df_plot[parameter], 
        cmap=True,
        style='s0.25c', # Quadrados de 0.25cm
        pen='0.1p,black'
    )

    # 5. Barra de cores
    fig.colorbar(frame=f'af+l"Valor de {parameter}"')
    
    return fig