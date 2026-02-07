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

def robust_regression(h, d, min_points, trava=True):
    if len(h) < min_points:
        return np.nan, np.nan, np.nan, np.nan

    # trava de segurança: exige pelo menos 5 pontos para considerar a categoria estatisticamente válida
    n_ocean = np.sum(h < 0) # retorna True para quem corresponde
    n_continent = np.sum(h >= 0)
    
    if trava:
        has_ocean = n_ocean >= 4
        has_continent = n_continent >= 4
    else:
        has_ocean = n_ocean > 0
        has_continent = n_continent > 0

    if has_ocean and has_continent: # se retornar True pra tudo em ambos os casos
        # criando matriz A, com os três parÂmetros
        h_ocean = np.where(h < 0, h, 0)
        h_continent = np.where(h >= 0, h, 0)
        intercept = np.ones(len(h))
        A = np.column_stack((h_ocean, h_continent, intercept))
        case = 'mixed'

    elif has_ocean: # se retornar False para has_continent
        h_ocean = np.where(h < 0, h, 0)
        intercept = np.ones(len(h))
        A = np.column_stack((h_ocean, intercept))
        case = 'ocean'

    elif has_continent: # se retornar False para has_ocean
        h_continent = np.where(h >= 0, h, 0) 
        intercept = np.ones(len(h))
        A = np.column_stack((h_continent, intercept))
        case = 'continent'

    else:
        return np.nan, np.nan, np.nan, np.nan

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

    except:
        return np.nan, np.nan, np.nan, np.nan

    return np.nan, np.nan, np.nan, np.nan


def windows_regression(data, h, d, lon, lat, window_size, overlap, min_points, trava=True):
    
    window_coordinates, indices = bd.rolling_window_spherical(
        coordinates=(lon, lat), window_size = window_size, overlap = overlap
    )

    resultados = []
    empty_windows = 0

    for i, idx_window in enumerate(indices):
        h_window = h[idx_window]
        d_window = d[idx_window]

        if len(h_window) < 10:
            empty_windows += 1
            continue

        p_window = robust_regression(h_window, d_window, min_points, trava=trava)
        
        # Se a regressão retornou None ou menos dados que o esperado, ignoramos a janela
        if p_window is None or len(p_window) < 4:
            continue 

        n_ocean = np.sum(h_window < 0)
        n_cont = np.sum(h_window >= 0)

        resultados.append({
            'longitude': window_coordinates[0][i],
            'latitude': window_coordinates[1][i],
            'a_o': p_window[0],
            'a_c': p_window[1],
            'b': p_window[2],
            'r2' : p_window[3], # Agora garantido que existe
            'original-idx': i,
            'total_points': len(h_window),
            'ocean_points': n_ocean,
            'continent_points': n_cont
        })

    print(f" {empty_windows} janelas foram ignoradas por falta de dados.")
    
    return pd.DataFrame(resultados), indices


def plot_window_regression(i_wished_window, h_window, d_window, a_ocean_window, a_continent_window, intercept_window, r2_window):
    plt.figure(figsize=(10,6))
    ax = plt.gca()

    h_min, h_max = h_window.min(), h_window.max()
    plt.xlim(h_min, h_max)

    plt.scatter(h_window, d_window, color='black', alpha=0.2, s=20, label='Data')

    has_ocean = not np.isnan(a_ocean_window) and np.any(h_window < 0)
    has_continent = not np.isnan(a_continent_window) and np.any(h_window >= 0)

    if has_ocean:
        h_ocean_points = h_window[h_window < 0]
        h_plot = np.sort(h_ocean_points) 
        d_pred = a_ocean_window * h_plot + intercept_window
        plt.plot(h_plot, d_pred, color='blue', lw=2.5, 
                 label=f'Ocean: {a_ocean_window:.5f} mGal/m')

    if has_continent:
        h_cont_points = h_window[h_window >= 0]
        h_plot = np.sort(h_cont_points)
        d_pred = a_continent_window * h_plot + intercept_window
        plt.plot(h_plot, d_pred, color='red', lw=2.5, 
                 label=f'Continent: {a_continent_window:.5f} mGal/m')

    if h_min <= 0 <= h_max:
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

    plt.title(f' {i_wished_window}th window: Robust Regression')
    plt.xlabel('Topography (m)')
    plt.ylabel('Bouguer anomaly (mGal)')
    
    plt.legend(loc='upper right')

    text_stats = f'Intercept (b): {intercept_window:.4f} mGal\nR²: {r2_window:.4f}'
    plt.text(0.98, 0.81, text_stats, transform=ax.transAxes, 
             horizontalalignment='right', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    plt.grid(True, alpha=0.2, linestyle=':')
    plt.show()


'''
def plot_parameters_map(data, parameter, v_range, step, cmap, reverse, projection):
    # 1. Limpeza e verificação de dados existentes
    df_plot = data.dropna(subset=[parameter])
    
    if df_plot.empty:
        print(f"AVISO: A coluna '{parameter}' está vazia. Verifique a regressão.")
        return None

    # 2. Definição da Região
    # Como o CRUST1.0 é um modelo global, o mais seguro é usar o atalho "d" 
    # Isso define automaticamente o limite [-180, 180, -90, 90] sem erros de float
    region = "d" 
    
    fig = pygmt.Figure()

    # 3. Configuração da escala de cores (CPT)
    pygmt.makecpt(
        cmap=cmap, 
        series=[v_range[0], v_range[1], step], 
        continuous=True, 
        background=True,
        reverse=reverse
    )

    # 4. Base do mapa (Projeção Mollweide centralizada no Brasil/Atlântico)
    fig.basemap(
        region=region, 
        projection=f'{projection}/15c', # Centralizado em -45 para ver a margem brasileira
        frame=["af", f"+t'Mapa de {parameter}'"]
    )

    # 5. Elementos geográficos

    # 6. Plotagem dos pontos (Ajuste do tamanho 's')
    # Usamos quadrados de 0.25c para preencher bem a grade de 1 grau
    fig.plot(
        x=df_plot.longitude,
        y=df_plot.latitude, 
        fill=df_plot[parameter], 
        cmap=True,
        style='s0.25c', 
        pen='0.1p,black'
    )

    #fig.coast(shorelines='0.5p,black')


    fig.colorbar(frame=f'af+l"Valor de {parameter}"')
    
    return fig
'''

def plot_parameters_map(data, parameter, v_range, step, cmap, reverse, projection):
    # 1. Limpeza
    df_plot = data.dropna(subset=[parameter])
    
    if df_plot.empty:
        print(f"AVISO: A coluna '{parameter}' está vazia.")
        return None

    # 2. Definição da Região AUTOMÁTICA (Para dar o zoom na sua área)
    # Pegamos os limites dos dados e adicionamos uma pequena margem de 1 grau
    region = [
        df_plot.longitude.min() - 1, df_plot.longitude.max() + 1,
        df_plot.latitude.min() - 1, df_plot.latitude.max() + 1
    ]
    
    fig = pygmt.Figure()

    # 3. Escala de cores
    pygmt.makecpt(
        cmap=cmap, 
        series=[v_range[0], v_range[1], step], 
        continuous=True, 
        reverse=reverse
    )

    # 4. Base do mapa com Zoom (15c de largura)
    fig.basemap(
        region=region, 
        projection=f"{projection}15c", 
        frame=["af", f"+t'Mapa de {parameter}'"]
    )

    # 5. Elementos geográficos (Essencial para se localizar no zoom)
    fig.coast(shorelines='0.5p,black', borders='1/0.2p,gray')

    # 6. Plotagem dos pontos (Quadrados preenchendo a grade)
    fig.plot(
        x=df_plot.longitude,
        y=df_plot.latitude, 
        fill=df_plot[parameter], 
        cmap=True,
        style='s0.25c', # Tamanho do quadrado ajustado para o zoom
        pen='0.1p'
    )

    fig.colorbar(frame=f'af+l"{parameter}"')
    
    return fig