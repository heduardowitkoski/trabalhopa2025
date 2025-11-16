import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

# --- 1. Carregar e Limpar os Dados ---

print("--- Passo 1: Carregando e Limpando Arquivos CSV ---")

try:
    df_indices = pd.read_csv('indices.csv')
    df_coanc = pd.read_csv('coancestralidade.csv')
    
    df_indices = df_indices.rename(columns={
        'Animal': 'AnimalID', 
        'Indice de seleção': 'Index'
    })
    df_coanc = df_coanc.rename(columns={
        'Coef': 'Coancestrality',
        'Animal_1': 'Par1_Nome', # Formato 'Pai_Mae'
        'Animal_2': 'Par2_Nome'  # Formato 'Pai_Mae'
    })

    # Remove animais com índice NaN (se existirem no arquivo de índices)
    df_indices = df_indices.dropna(subset=['Index'])
    
    print("Arquivos de Índices e Coancestralidade carregados e renomeados.")

except FileNotFoundError:
    print("\n!!! ERRO: Arquivos CSV ('indices.csv' ou 'coancestralidade.csv') não encontrados!")
    exit()
except Exception as e:
    print(f"\n!!! ERRO ao carregar ou renomear colunas: {e}")
    exit()


# --- 2. Identificar Fêmeas e Machos (Automaticamente) ---

print("\n" + "="*40 + "\n")
print("--- Passo 2: Identificando Fêmeas e Machos (Automático) ---")
print("Lendo nomes 'Pai_Mae' do arquivo de coancestralidade...")

try:
    machos_set = set()
    femeas_set = set()
    
    todos_nomes_pares = pd.concat([df_coanc['Par1_Nome'], df_coanc['Par2_Nome']]).unique()
    
    print(f"Analisando {len(todos_nomes_pares)} nomes de pares únicos...")
    
    for nome_par in todos_nomes_pares:
        if pd.isna(nome_par):
            continue
        try:
            macho_id_str, femea_id_str = str(nome_par).split('_')
            machos_set.add(int(macho_id_str))
            femeas_set.add(int(femea_id_str))
        except (ValueError, TypeError):
            pass # Ignora nomes mal formatados

    femeas_nomes_lista = list(femeas_set)
    machos_nomes_lista = list(machos_set)

    if not femeas_nomes_lista or not machos_nomes_lista:
        print("\n" + "!"*40)
        print("ERRO: Não foi possível extrair IDs de machos ou fêmeas.")
        print("Verifique o formato 'Pai_Mae' no arquivo de coancestralidade.")
        print("!"*40)
        exit()

    print("Extração de IDs concluída.")

    # Dicionário para acesso rápido aos índices dos pais
    indices_pais = pd.Series(df_indices.Index.values, index=df_indices.AnimalID).to_dict()

    # --- MUDANÇA PRINCIPAL AQUI ---
    # NÃO filtramos mais. Usamos TODOS os animais encontrados no coancestralidade.csv
    femeas_nomes = sorted(femeas_nomes_lista)
    machos_nomes = sorted(machos_nomes_lista)

    num_femeas = len(femeas_nomes)
    num_machos = len(machos_nomes)

    if num_femeas == 0 or num_machos == 0:
         print("ERRO: Nenhuma fêmea ou macho encontrado no arquivo de coancestralidade.")
         exit()

    print(f"Total de fêmeas (F) válidas para o estudo: {num_femeas}")
    print(f"Total de machos (M) válidos para o estudo: {num_machos}")

    # Apenas uma verificação para informar o usuário
    all_animals_in_index_file = set(indices_pais.keys())
    femeas_com_indice = sum(1 for f in femeas_nomes if f in all_animals_in_index_file)
    machos_com_indice = sum(1 for m in machos_nomes if m in all_animals_in_index_file)
    print(f"  > {femeas_com_indice} fêmeas e {machos_com_indice} machos TÊM um índice no arquivo.")
    print("  > Os demais usarão índice 0.0 por padrão.")

    # Mapeamento de ID (int) -> ÍNDICE da Matriz (int, de 0 a N-1)
    map_femea_idx = {nome: i for i, nome in enumerate(femeas_nomes)}
    map_macho_idx = {nome: i for i, nome in enumerate(machos_nomes)}
    print("Mapeamentos (ID -> Posição na Matriz) criados.")

except Exception as e:
    print(f"\n!!! ERRO Inesperado no Passo 2: {e}")
    exit()


# --- 3. Construir Matriz de Índices da Prole (Passo 5 do trabalho) ---

print("\n" + "="*40 + "\n")
print("--- Passo 3: Construindo Matriz de Índices (Mapa de Valor) ---")

matriz_indices = np.zeros((num_femeas, num_machos))

for f_nome, f_idx in map_femea_idx.items():
    for m_nome, m_idx in map_macho_idx.items():
        
        # --- MUDANÇA PRINCIPAL AQUI ---
        # Usamos .get(animal_id, 0.0)
        # Se o animal_id (f_nome ou m_nome) não estiver no dicionário 'indices_pais',
        # ele retorna o valor padrão 0.0
        idx_femea = indices_pais.get(f_nome, 0.0)
        idx_macho = indices_pais.get(m_nome, 0.0)
        
        # Cálculo do índice da prole (média aritmética)
        matriz_indices[f_idx, m_idx] = (idx_femea + idx_macho) / 2.0

print(f"Matriz de Índices (Valor) construída. Dimensões: {matriz_indices.shape}")


# --- 4. Construir Matriz de Coancestralidade (Passos 3 e 4 do trabalho) ---

print("\n" + "="*40 + "\n")
print("--- Passo 4: Construindo Matriz de Coancestralidade (Mapa de Drama) ---")

num_pares_total = num_femeas * num_machos
print(f"Dimensão da Matriz N x N, onde N (pares totais) = {num_pares_total}")

map_par_idx = {}
map_idx_par = [None] * num_pares_total 
map_nome_par_idx = {}

par_idx_counter = 0
for f_nome, f_idx in map_femea_idx.items():
    for m_nome, m_idx in map_macho_idx.items():
        map_par_idx[(f_idx, m_idx)] = par_idx_counter
        map_idx_par[par_idx_counter] = (f_idx, m_idx)
        nome_par_str = f"{m_nome}_{f_nome}" 
        map_nome_par_idx[nome_par_str] = par_idx_counter
        par_idx_counter += 1

print("Mapeamentos de pares (Pai_Mae -> Posição na Matriz) criados.")

matriz_coanc = np.zeros((num_pares_total, num_pares_total), dtype=np.float32)

print("Preenchendo a Matriz de Coancestralidade (isso pode levar um tempo)...")
start_time_coanc = time.time()
pares_encontrados = 0
pares_ignorados = 0

for _, row in df_coanc.iterrows():
    nome_par1 = row['Par1_Nome']
    nome_par2 = row['Par2_Nome']
    coanc_valor = row['Coancestrality']
    
    idx1 = map_nome_par_idx.get(nome_par1)
    idx2 = map_nome_par_idx.get(nome_par2)
    
    # Esta lógica está correta, pois .get() retorna None se o par não
    # foi construído (o que não deve acontecer agora)
    if idx1 is not None and idx2 is not None:
        matriz_coanc[idx1, idx2] = coanc_valor
        matriz_coanc[idx2, idx1] = coanc_valor
        pares_encontrados += 1
    elif nome_par1 and nome_par2:
        pares_ignorados += 1

end_time_coanc = time.time()

print(f"Matriz de Coancestralidade preenchida em {end_time_coanc - start_time_coanc:.2f}s")
print(f"Total de {pares_encontrados} relações de coancestralidade carregadas.")
if pares_ignorados > 0:
    print(f"({pares_ignorados} relações foram ignoradas - isso não deveria acontecer agora).")

if pares_encontrados == 0 and len(df_coanc) > 0:
    print("\n!!! AVISO SÉRIO: Nenhuma relação de coancestralidade foi carregada!")
    
print("\n" + "="*40 + "\n")
print("🎉 PREPARAÇÃO CONCLUÍDA! 🎉")
print("Temos as matrizes 'matriz_indices' e 'matriz_coanc' prontas.")
print("Estamos prontos para o Passo 5: Implementar o Simulated Annealing.")
print("\n" + "="*40 + "\n")


# --- (FIM DA PREPARAÇÃO) ---


# --- Passo 7.1: Estatísticas da Base de Dados ---
# (Obrigatório pelo seu trabalho)
print("\n--- Passo 7.1: Análise da Base de Dados ---")
print(f"Total de Fêmeas (F): {num_femeas}")
print(f"Total de Machos (M): {num_machos}")
print(f"Total de Pares Possíveis (N): {num_pares_total}")

print("\nEstatísticas - Índice de Seleção da Prole (matriz F x M):")
print(f"  Mínimo: {matriz_indices.min():.4f}")
print(f"  Máximo: {matriz_indices.max():.4f}")
print(f"  Média:  {matriz_indices.mean():.4f}")
print(f"  Mediana:{np.median(matriz_indices):.4f}")

# Para coancestralidade, só consideramos os valores > 0
coanc_valores_unicos = matriz_coanc[np.triu_indices(num_pares_total, k=1)]
coanc_valores_positivos = coanc_valores_unicos[coanc_valores_unicos > 0]

if len(coanc_valores_positivos) > 0:
    print("\nEstatísticas - Coancestralidade da Prole (matriz N x N, valores > 0):")
    print(f"  Mínimo: {coanc_valores_positivos.min():.4f}")
    print(f"  Máximo: {coanc_valores_positivos.max():.4f}")
    print(f"  Média:  {coanc_valores_positivos.mean():.4f}")
    print(f"  Mediana:{np.median(coanc_valores_positivos):.4f}")
else:
    print("\nNenhum valor de coancestralidade > 0 encontrado.")


# --- Passo 6: Modelo do Simulated Annealing ---

print("\n" + "="*40 + "\n")
print("--- Passo 6: Definindo Funções do Simulated Annealing ---")

# Peso para garantir a prioridade da coancestralidade
# (Mesmo que o índice seja 0, mantemos a fórmula)
PESO_COANC = 1_000_000_000.0

def criar_solucao_inicial(num_femeas, num_machos, max_uso_por_macho):
    """Gera uma solução aleatória válida que respeita a restrição max_uso."""
    solucao = [-1] * num_femeas
    uso_machos = np.zeros(num_machos, dtype=int)
    
    # Converte max_uso para um array
    if isinstance(max_uso_por_macho, (int, float)):
        max_uso_array = np.full(num_machos, max_uso_por_macho)
    else:
        # Se você quiser definir limites diferentes por macho
        max_uso_array = np.array(max_uso_por_macho)

    # Verifica se o problema é viável
    if num_femeas > np.sum(max_uso_array):
        raise ValueError(f"Problema inviável: {num_femeas} fêmeas, mas capacidade total dos machos é {np.sum(max_uso_array)} (com max_uso={max_uso_por_macho}).")

    for f_idx in range(num_femeas):
        # Encontra machos com "vagas"
        machos_disponiveis = [m_idx for m_idx in range(num_machos) if uso_machos[m_idx] < max_uso_array[m_idx]]
        
        if not machos_disponiveis:
             # Isso só acontece se a verificação de viabilidade falhar
            raise Exception("Erro na lógica: ficou sem machos disponíveis.")
            
        macho_escolhido = random.choice(machos_disponiveis)
        solucao[f_idx] = macho_escolhido
        uso_machos[macho_escolhido] += 1
        
    return solucao, uso_machos

def gerar_vizinho(solucao_atual, uso_machos_atual, max_uso_por_macho):
    """Gera um estado vizinho válido (Swap ou Reassign)."""
    
    vizinho = solucao_atual[:]
    uso_machos = uso_machos_atual.copy()
    
    if isinstance(max_uso_por_macho, (int, float)):
        max_uso_array = np.full(len(uso_machos), max_uso_por_macho)
    else:
        max_uso_array = np.array(max_uso_por_macho)

    num_femeas = len(vizinho)
    num_machos = len(uso_machos)
    
    # 50% chance de Swap (troca)
    if random.random() < 0.5:
        f1, f2 = random.sample(range(num_femeas), 2)
        vizinho[f1], vizinho[f2] = vizinho[f2], vizinho[f1]
        # 'uso_machos' não muda
    
    # 50% chance de Reassign (reatribuição)
    else:
        f_idx = random.randint(0, num_femeas - 1)
        macho_antigo = vizinho[f_idx]
        
        # Lista de machos com vagas (pode incluir o macho antigo se ele tiver >1 uso)
        machos_disponiveis = [m_idx for m_idx in range(num_machos) 
                             if uso_machos[m_idx] < max_uso_array[m_idx]]
        
        # Filtra para garantir que seja um *novo* macho
        machos_novos_disponiveis = [m for m in machos_disponiveis if m != macho_antigo]

        if machos_novos_disponiveis:
            macho_novo = random.choice(machos_novos_disponiveis)
            
            vizinho[f_idx] = macho_novo
            uso_machos[macho_antigo] -= 1
            uso_machos[macho_novo] += 1
        else:
            # Falhou em reatribuir (ex: só o macho antigo tem vaga, ou nenhum tem)
            # Apenas faz um Swap para não perder a iteração
            f1, f2 = random.sample(range(num_femeas), 2)
            vizinho[f1], vizinho[f2] = vizinho[f2], vizinho[f1]
            
    return vizinho, uso_machos

def calcular_energia_e_objetivos(solucao, matriz_indices, matriz_coanc, map_par_idx):
    """Calcula a energia (custo) da solução e os objetivos separados."""
    
    total_coanc = 0.0
    total_index = 0.0
    num_femeas = len(solucao)
    
    # Converte a solução (fêmea -> macho) para índices de pares (0 a N-1)
    indices_pares_solucao = np.zeros(num_femeas, dtype=int)
    for f_idx in range(num_femeas):
        m_idx = solucao[f_idx]
        total_index += matriz_indices[f_idx, m_idx]
        indices_pares_solucao[f_idx] = map_par_idx[(f_idx, m_idx)]

    # Calcula a coancestralidade total
    # Itera sobre todos os pares únicos de fêmeas (i, j)
    for i in range(num_femeas):
        par_i_idx = indices_pares_solucao[i]
        for j in range(i + 1, num_femeas):
            par_j_idx = indices_pares_solucao[j]
            total_coanc += matriz_coanc[par_i_idx, par_j_idx]

    # Energia combinada (queremos MINIMIZAR)
    energia = (PESO_COANC * total_coanc) - total_index
    
    return energia, total_coanc, total_index

def rodar_simulated_annealing(
    num_femeas, num_machos, max_uso, 
    matriz_indices, matriz_coanc, map_par_idx,
    temp_inicial, temp_final, alfa, iter_por_temp
):
    
    # 1. Solução inicial
    try:
        solucao_atual, uso_atual = criar_solucao_inicial(num_femeas, num_machos, max_uso)
    except ValueError as e:
        print(f"\nERRO: {e}")
        return None, 0, 0, [] # Retorna falha
        
    energia_atual, coanc_atual, idx_atual = calcular_energia_e_objetivos(
        solucao_atual, matriz_indices, matriz_coanc, map_par_idx
    )
    
    melhor_solucao = solucao_atual[:]
    melhor_coanc = coanc_atual
    melhor_idx = idx_atual
    
    temp = temp_inicial
    historia = [] # (tempo, coancestralidade, índice)
    start_time = time.time()
    
    print(f"  Iniciando SA (T={temp_inicial}, max_uso={max_uso})... ", end="", flush=True)

    while temp > temp_final:
        for _ in range(iter_por_temp):
            
            # 2. Gerar vizinho
            vizinho, uso_vizinho = gerar_vizinho(solucao_atual, uso_atual, max_uso)
            
            # 3. Calcular energia
            energia_vizinho, coanc_vizinho, idx_vizinho = calcular_energia_e_objetivos(
                vizinho, matriz_indices, matriz_coanc, map_par_idx
            )
            
            # 4. Decidir se aceita
            delta_energia = energia_vizinho - energia_atual
            
            if delta_energia < 0 or random.random() < math.exp(-delta_energia / temp):
                # Aceita (melhorou ou por probabilidade)
                solucao_atual, uso_atual = vizinho[:], uso_vizinho.copy()
                energia_atual, coanc_atual, idx_atual = energia_vizinho, coanc_vizinho, idx_vizinho
                
                # Atualiza a *melhor solução global*
                # (Compara coancestralidade primeiro, depois índice)
                if (coanc_atual < melhor_coanc) or \
                   (coanc_atual == melhor_coanc and idx_atual > melhor_idx):
                    melhor_solucao, melhor_coanc, melhor_idx = solucao_atual[:], coanc_atual, idx_atual
        
        # Guardar histórico para plot
        historia.append((time.time() - start_time, melhor_coanc, melhor_idx))
        
        # Resfriar
        temp *= alfa
        
    print(f"Concluído em {time.time() - start_time:.2f}s. Melhor Coanc: {melhor_coanc:.4f}")
    return melhor_solucao, melhor_coanc, melhor_idx, historia

print("Funções do SA definidas.")


# --- Passo 7.2 a 7.5: Execução dos Testes e Apresentação ---

print("\n" + "="*40 + "\n")
print("--- Passo 7: Executando Experimentos ---")

def plotar_convergencia(historias_runs, titulo):
    """Plota as curvas de convergência para múltiplas execuções."""
    
    if not historias_runs: # Se não houver dados (ex: problema inviável)
        print("Nenhum dado para plotar.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Coancestralidade Total (Minimizar)', color='tab:blue')
    
    # Ax secundário para o Índice
    ax2 = ax1.twinx()
    ax2.set_ylabel('Índice de Seleção Total (Maximizar)', color='tab:red')

    # Se o índice é sempre 0, ajusta o limite do eixo Y
    all_indices = [h[2] for historia in historias_runs for h in historia]
    if all(i == 0 for i in all_indices):
        ax2.set_ylim(-1, 1) # Centraliza em 0

    for i, historia in enumerate(historias_runs):
        if not historia: continue
        tempos = [h[0] for h in historia]
        coancs = [h[1] for h in historia]
        indices = [h[2] for h in historia]
        
        # Plotar com transparência para ver a média
        ax1.plot(tempos, coancs, color='tab:blue', alpha=0.3)
        ax2.plot(tempos, indices, color='tab:red', alpha=0.3)

    # Adiciona legendas de forma limpa
    ax1.plot([], [], color='tab:blue', label='Coancestralidade (Média Runs)')
    ax2.plot([], [], color='tab:red', label='Índice (Média Runs)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()

# --- Hiperparâmetros do SA ---
# (Você pode precisar ajustar isso)
HIPERPARAMETROS = {
    'temp_inicial': 1000.0,
    'temp_final': 0.1,
    'alfa': 0.99,         # Resfriamento (0.99 = lento, 0.9 = rápido)
    'iter_por_temp': 100  # Iterações em cada temperatura
}

# --- Testes do Trabalho ---
max_uso_testes = [12, 18, 20]
num_runs = 5 # Número de repetições por teste (para estatística)


# Loop principal de execução
for max_uso in max_uso_testes:
    print(f"\n--- Iniciando Testes para max_uso = {max_uso} ---")
    
    resultados_finais = []
    historias_runs = []
    
    for r in range(num_runs):
        print(f"  Run {r+1}/{num_runs}...")
        
        resultado = rodar_simulated_annealing(
            num_femeas, num_machos, max_uso, 
            matriz_indices, matriz_coanc, map_par_idx,
            **HIPERPARAMETROS
        )
        
        sol, coanc, idx, hist = resultado
        
        if sol is not None: # Se a execução foi bem-sucedida
            resultados_finais.append({'coanc': coanc, 'index': idx})
            historias_runs.append(hist)
    
    if not resultados_finais:
        print(f"Nenhum resultado válido para max_uso={max_uso}. Pulando...")
        continue

    # --- (Passo 7.3 e 7.4) Apresentar estatísticas dos resultados
    coancs_finais = [r['coanc'] for r in resultados_finais]
    indices_finais = [r['index'] for r in resultados_finais]
    
    print("\nEstatísticas das Soluções Finais (das {num_runs} runs):")
    print(f"  Coancestralidade (Min, Média, Mediana, Max):")
    print(f"  {np.min(coancs_finais):.4f}, {np.mean(coancs_finais):.4f}, {np.median(coancs_finais):.4f}, {np.max(coancs_finais):.4f}")
    
    print(f"  Índice de Seleção (Min, Média, Mediana, Max):")
    print(f"  {np.min(indices_finais):.4f}, {np.mean(indices_finais):.4f}, {np.median(indices_finais):.4f}, {np.max(indices_finais):.4f}")

    # --- (Passo 7.3) Plotar curvas de convergência
    plotar_convergencia(historias_runs, f'Convergência do SA (max_uso = {max_uso}, {num_runs} Runs)')


print("\n" + "="*40 + "\n")
print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
print("Analise os gráficos e os valores de Média/Mediana no console.")
print("="*40 + "\n")
