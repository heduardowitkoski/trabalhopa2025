import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÃO INICIAL E REPRODUTIBILIDADE
# =============================================================================

# Define a semente (seed) para garantir que os números aleatórios sejam os mesmos
# em toda execução.
random.seed(42)
np.random.seed(42)

# =============================================================================
# TRABALHO PRÁTICO DE IMPLEMENTAÇÃO - SISTEMAS DE MELHORAMENTO ANIMAL
# Otimização de Acasalamento usando Simulated Annealing
#
# OBJETIVO: Encontrar a melhor combinação de acasalamentos (Fêmea x Macho)
# que minimize a coancestralidade (endogamia) e maximize o índice genético.
# =============================================================================

print("--- INICIANDO O ALGORITMO DE OTIMIZAÇÃO ---")

# =============================================================================
# 1. LEITURA E PREPARAÇÃO DOS DADOS
# =============================================================================
# Nesta etapa, carregamos os arquivos CSV e limpamos os dados para evitar erros
# durante o processamento das matrizes.

try:
    # Carrega os arquivos CSV para DataFrames do Pandas
    df_indices = pd.read_csv('indices.csv')
    df_coanc = pd.read_csv('coancestralidade.csv')
    
    # Renomeia as colunas para um padrão interno
    df_indices = df_indices.rename(columns={
        'Animal': 'AnimalID', 
        'Indice de seleção': 'Index'
    })
    df_coanc = df_coanc.rename(columns={
        'Coef': 'Coancestrality',
        'Animal_1': 'Par1_Nome', 
        'Animal_2': 'Par2_Nome'  
    })

    # Remove linhas onde o 'Index' é NaN (vazio), pois não podemos otimizar sem valor genético
    df_indices = df_indices.dropna(subset=['Index'])
    print("Arquivos carregados com sucesso.")

except FileNotFoundError:
    print("!!! ERRO: Arquivos 'indices.csv' ou 'coancestralidade.csv' não encontrados.")
    exit()

# =============================================================================
# 2. ORDENAÇÃO E IDENTIFICAÇÃO ÚNICA (MAPPING)
# =============================================================================
# O algoritmo trabalha com matrizes numéricas (numpy). Precisamos converter
# os nomes dos animais (ex: "Touro_A") para índices numéricos (0, 1, 2...).

print("Identificando animais e gerando ordenação...")

try:
    machos_set = set()
    femeas_set = set()
    
    # Coleta todos os nomes de pares únicos que aparecem no arquivo de coancestralidade
    # Exemplo esperado de nome no CSV: "101_202" (Macho_Femea)
    todos_nomes_pares = pd.concat([df_coanc['Par1_Nome'], df_coanc['Par2_Nome']]).unique()
    
    for nome_par in todos_nomes_pares:
        if pd.isna(nome_par): continue
        try:
            # Assume que o formato é "MachoID_FemeaID"
            macho_str, femea_str = str(nome_par).split('_')
            machos_set.add(int(macho_str))
            femeas_set.add(int(femea_str))
        except (ValueError, TypeError):
            # Ignora linhas que não seguem o padrão esperado
            pass 

    # Ordena as listas para garantir que o índice 0 seja sempre o menor ID, etc.
    femeas_nomes = sorted(list(femeas_set))
    machos_nomes = sorted(list(machos_set))
    
    num_femeas = len(femeas_nomes)
    num_machos = len(machos_nomes)

    if num_femeas == 0 or num_machos == 0:
        print("ERRO: Não foi possível identificar população válida.")
        exit()

    print(f"População identificada: {num_femeas} Fêmeas e {num_machos} Machos.")

except Exception as e:
    print(f"Erro na identificação: {e}")
    exit()

# Cria dicionários para busca rápida: Nome -> Índice na Matriz
# Ex: map_femea_idx[105] = 0
map_femea_idx = {nome: i for i, nome in enumerate(femeas_nomes)}
map_macho_idx = {nome: i for i, nome in enumerate(machos_nomes)}


# =============================================================================
# 3. CONSTRUÇÃO DA MATRIZ DE VALOR (ÍNDICES DA PROLE)
# =============================================================================
# Aqui calculamos o valor genético esperado de cada possível acasalamento.
# Valor da Prole = (Valor do Pai + Valor da Mãe) / 2

print("Construindo Matriz de Índices (Valor)...")

# Transforma o DF em dicionário para acesso O(1)
indices_arquivo = pd.Series(df_indices.Index.values, index=df_indices.AnimalID).to_dict()
indices_completos = {}

# Preenchimento de dados faltantes (Fallback)
print("  -> Verificando índices faltantes...")
count_missing = 0
for animal_id in (femeas_nomes + machos_nomes):
    if animal_id in indices_arquivo:
        indices_completos[animal_id] = indices_arquivo[animal_id]
    else:
        # Se um animal está na genealogia mas não tem índice no arquivo,
        # geramos um valor aleatório para não travar o código.
        indices_completos[animal_id] = random.uniform(0, 20)
        count_missing += 1
if count_missing > 0:
    print(f"     (Aviso: {count_missing} animais sem índice preenchidos aleatoriamente)")

# Inicializa matriz de zeros: Linhas = Fêmeas, Colunas = Machos
matriz_indices = np.zeros((num_femeas, num_machos))

# Preenche a matriz com a média dos pais
for f_nome, f_idx in map_femea_idx.items():
    for m_nome, m_idx in map_macho_idx.items():
        val_f = indices_completos[f_nome]
        val_m = indices_completos[m_nome]
        matriz_indices[f_idx, m_idx] = (val_f + val_m) / 2.0


# =============================================================================
# 4. CONSTRUÇÃO DA MATRIZ DE COANCESTRALIDADE
# =============================================================================
# Esta é a parte mais pesada. A matriz de coancestralidade aqui não é Indivíduo x Indivíduo,
# mas sim CASAL x CASAL (ou PROLE x PROLE hipotética).
# Ela diz: "O quão aparentado o filho do casal A é do filho do casal B?"

print("Construindo Matriz de Coancestralidade (Matriz de Adjacências)...")
num_pares_total = num_femeas * num_machos

# Matriz quadrada gigante: (N_Fêmeas * N_Machos)²
matriz_coanc = np.zeros((num_pares_total, num_pares_total), dtype=np.float32)

map_par_idx = {}         # Tupla (idx_femea, idx_macho) -> índice linear global (0 a N_Pares)
map_nome_par_idx = {}    # String "Macho_Femea" -> índice linear global

idx_counter = 0
# Cria o mapeamento linear de todos os acasalamentos possíveis
for f_nome, f_idx in map_femea_idx.items():
    for m_nome, m_idx in map_macho_idx.items():
        map_par_idx[(f_idx, m_idx)] = idx_counter
        # Constrói string idêntica à do arquivo CSV para fazer o "match"
        nome_str = f"{m_nome}_{f_nome}"
        map_nome_par_idx[nome_str] = idx_counter
        idx_counter += 1

start_t = time.time()
# Preenche a matriz lendo o CSV linha a linha
for _, row in df_coanc.iterrows():
    # Busca os índices lineares dos dois pares envolvidos na linha do CSV
    p1 = map_nome_par_idx.get(row['Par1_Nome'])
    p2 = map_nome_par_idx.get(row['Par2_Nome'])
    val = row['Coancestrality']
    
    if p1 is not None and p2 is not None:
        # A matriz é simétrica (A se relaciona com B igual B se relaciona com A)
        matriz_coanc[p1, p2] = val
        matriz_coanc[p2, p1] = val 

print(f"Matriz construída em {time.time()-start_t:.2f}s.")

# --- Relatório Estatístico Inicial (Item 7.1) ---
print("\n--- ESTATÍSTICAS DA BASE DE DADOS (Item 7.1) ---")
print(f"Total Fêmeas: {num_femeas}, Total Machos: {num_machos}")

# Estatísticas sobre os Valores Genéticos (Índices)
print(f"Índice Prole -> Min: {matriz_indices.min():.4f}, Max: {matriz_indices.max():.4f}")
print(f"                Média: {matriz_indices.mean():.4f}, Mediana: {np.median(matriz_indices):.4f}")

# Estatísticas sobre a Coancestralidade
vals_coanc_csv = df_coanc['Coancestrality'].values
if len(vals_coanc_csv) > 0:
    print(f"Coancestralidade -> Min: {vals_coanc_csv.min():.4f}, Max: {vals_coanc_csv.max():.4f}")
    print(f"                    Média: {vals_coanc_csv.mean():.4f}, Mediana: {np.median(vals_coanc_csv):.4f}")
else:
    print("Coancestralidade -> Base vazia.")
print("="*60)


# =============================================================================
# 5. MODELO: SIMULATED ANNEALING 
# =============================================================================

# Constante de penalização.
# Como queremos minimizar a Energia, e a Coancestralidade é muito pequena (0.0x),
# multiplicamos por um valor gigante para que ela tenha peso maior que o Índice.
PESO_COANC = 1_000_000.0 

def criar_solucao_inicial(num_f, num_m, max_uso):
    """
    Gera uma solução aleatória válida onde cada fêmea tem 1 macho,
    respeitando o limite de uso de cada macho.
    """
    solucao = [-1] * num_f  # Vetor onde índice=Fêmea, valor=ID_Macho
    uso_machos = np.zeros(num_m, dtype=int)
    
    # Validação preliminar de viabilidade
    if num_f > (num_m * max_uso):
        raise ValueError("Inviável: Fêmeas > Capacidade total dos machos.")

    for f in range(num_f):
        # Lista de machos que ainda não atingiram a cota (max_uso)
        disponiveis = [m for m in range(num_m) if uso_machos[m] < max_uso]
        
        if not disponiveis:
            # Fallback seguro (teoricamente não deve ocorrer devido ao if acima)
            m_escolhido = random.randint(0, num_m - 1)
        else:
            m_escolhido = random.choice(disponiveis)
            
        solucao[f] = m_escolhido
        uso_machos[m_escolhido] += 1
        
    return solucao, uso_machos

def gerar_vizinho(solucao_atual, uso_atual, max_uso):
    """
    Realiza um movimento de perturbação na solução atual.
    Tipos de movimento:
    1. Swap: Troca os machos de duas fêmeas.
    2. Reassign: Troca o macho de uma fêmea por outro disponível.
    """
    vizinho = solucao_atual[:]
    uso_vizinho = uso_atual.copy()
    num_f = len(vizinho)
    num_m = len(uso_vizinho)

    # 50% de chance para cada tipo de movimento
    if random.random() < 0.5:
        # --- SWAP (Troca) ---
        # Escolhe duas fêmeas aleatórias e troca seus parceiros
        # Isso não altera a contagem de uso dos machos.
        f1, f2 = random.sample(range(num_f), 2)
        vizinho[f1], vizinho[f2] = vizinho[f2], vizinho[f1]
    else:
        # --- REASSIGN (Realocação) ---
        # Escolhe uma fêmea e tenta dar um novo parceiro para ela
        f_alvo = random.randint(0, num_f - 1)
        m_antigo = vizinho[f_alvo]
        
        # Busca machos que não estão cheios (e não são o atual)
        disponiveis = [m for m in range(num_m) if uso_vizinho[m] < max_uso and m != m_antigo]
        
        if disponiveis:
            m_novo = random.choice(disponiveis)
            vizinho[f_alvo] = m_novo
            uso_vizinho[m_antigo] -= 1
            uso_vizinho[m_novo] += 1
        else:
            # Se não há machos livres, faz um Swap para não desperdiçar iteração
            f1, f2 = random.sample(range(num_f), 2)
            vizinho[f1], vizinho[f2] = vizinho[f2], vizinho[f1]
            
    return vizinho, uso_vizinho

def calcular_custo(solucao, mat_idx, mat_coanc, map_pares):
    """
    Calcula a 'Energia' do sistema. No Simulated Annealing, buscamos o MÍNIMO de energia.
    
    Fórmula: Energia = (Peso * Coancestralidade_Total) - Indice_Total
    
    Observação: Subtraímos o Índice porque queremos maximizá-lo (Minimizar o negativo é igual a Maximizar o positivo).
    """
    total_idx = 0.0
    total_coanc = 0.0
    n_f = len(solucao)
    
    # 1. Soma dos Índices Genéticos dos casais formados
    pares_indices = [] 
    for f_idx, m_idx in enumerate(solucao):
        total_idx += mat_idx[f_idx, m_idx]
        # Guarda o índice linear desse par para calcular coancestralidade depois
        pares_indices.append(map_pares[(f_idx, m_idx)])
        
    # 2. Soma da Coancestralidade entre TODOS os casais da solução (Somatório duplo)
    # Isso penaliza se a população resultante for muito consanguínea entre si.
    for i in range(n_f):
        p1 = pares_indices[i]
        for j in range(i + 1, n_f):
            p2 = pares_indices[j]
            total_coanc += mat_coanc[p1, p2]
            
    energia = (PESO_COANC * total_coanc) - total_idx
    return energia, total_coanc, total_idx

def simulated_annealing(max_uso, temp_ini=1000, temp_fim=0.1, alfa=0.99, iter_temp=100):
    """
    Loop principal do algoritmo.
    temp_ini: Temperatura inicial (alta = aceita soluções ruins para explorar)
    alfa: Taxa de resfriamento (ex: 0.99 reduz temperatura em 1% a cada passo)
    iter_temp: Quantas tentativas fazer em cada temperatura
    """
    try:
        curr_sol, curr_uso = criar_solucao_inicial(num_femeas, num_machos, max_uso)
    except ValueError:
        return None # Retorna None se for impossível criar solução
        
    curr_energy, curr_coanc, curr_idx = calcular_custo(curr_sol, matriz_indices, matriz_coanc, map_par_idx)
    
    # Armazena a "Melhor Solução Global" encontrada até agora
    best_sol = curr_sol[:]
    best_coanc = curr_coanc
    best_idx = curr_idx
    
    temp = temp_ini
    historico = [] 
    
    # MODIFICAÇÃO: Contador de iterações em vez de tempo
    total_iter = 0
    
    # Loop de Resfriamento
    while temp > temp_fim:
        # Loop de Equilíbrio Térmico (várias iterações na mesma temperatura)
        for _ in range(iter_temp):
            new_sol, new_uso = gerar_vizinho(curr_sol, curr_uso, max_uso)
            new_energy, new_coanc, new_idx = calcular_custo(new_sol, matriz_indices, matriz_coanc, map_par_idx)
            
            delta = new_energy - curr_energy
            
            # Critério de Metrópolis
            aceitar = False
            if delta < 0:
                # Se melhorou (energia diminuiu), aceita sempre
                aceitar = True
            else:
                # Se piorou, aceita com uma probabilidade baseada na temperatura
                # Quanto maior a temp, maior a chance de aceitar piora (evita mínimo local)
                if random.random() < math.exp(-delta / temp):
                    aceitar = True
            
            if aceitar:
                curr_sol, curr_uso = new_sol[:], new_uso
                curr_energy, curr_coanc, curr_idx = new_energy, new_coanc, new_idx
                
                # Atualiza o "Best Global"
                # Critério: Prioriza Menor Coanc. Se empate, Maior Índice.
                if (curr_coanc < best_coanc) or (abs(curr_coanc - best_coanc) < 1e-9 and curr_idx > best_idx):
                    best_sol, best_coanc, best_idx = curr_sol[:], curr_coanc, curr_idx
        
        # MODIFICAÇÃO: Atualiza contador de iterações e salva no histórico
        total_iter += iter_temp
        historico.append((total_iter, best_coanc, best_idx))
        
        temp *= alfa # Resfria
        
    return best_sol, best_coanc, best_idx, historico


# =============================================================================
# 6. EXECUÇÃO, TESTES E APRESENTAÇÃO DOS RESULTADOS
# =============================================================================

def plotar_graficos(historias, titulo):
    """
    Plota a evolução da Coancestralidade e do Índice ao longo do tempo.
    Usa eixo Y duplo pois as escalas são muito diferentes.
    """
    plt.figure(figsize=(10, 6))
    
    # Eixo da esquerda (Azul) -> Coancestralidade
    ax1 = plt.gca()
    # MODIFICAÇÃO: Eixo X agora é Iterações
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Coancestralidade (Minimizar)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Eixo da direita (Vermelho) -> Índice
    ax2 = ax1.twinx()
    ax2.set_ylabel('Índice (Maximizar)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Ajuste de escala caso índices sejam todos zero
    todos_indices = [pt[2] for h in historias for pt in h]
    if all(v == 0 for v in todos_indices): ax2.set_ylim(-1, 1)

    # Plota todas as execuções (runs) sobrepostas
    for i, hist in enumerate(historias):
        # pt[0] agora contém iterações, não tempo
        iters = [pt[0] for pt in hist]
        coancs = [pt[1] for pt in hist]
        idxs = [pt[2] for pt in hist]
        ax1.plot(iters, coancs, color='blue', alpha=0.3)
        ax2.plot(iters, idxs, color='red', alpha=0.3)
        
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

# Parâmetros de Teste: Diferentes limites de uso por macho
TESTES_MAX_USO = [12, 18, 20]
NUM_RUNS = 1 # Executa 1 vez para cada cenário (aumente se desejar média)

print("\n--- EXECUTANDO TESTES ---")

for max_uso in TESTES_MAX_USO:
    print(f"\n>>> Teste com max_uso = {max_uso} <<<")
    
    resultados_run = [] 
    historicos_run = [] 
    
    for r in range(NUM_RUNS):
        print(f"  Run {r+1}/{NUM_RUNS}...", end=" ", flush=True)
        res = simulated_annealing(max_uso)
        
        if res:
            sol, coanc, idx, hist = res
            resultados_run.append((coanc, idx))
            historicos_run.append(hist)
            print(f"Ok (Coanc: {coanc:.2f})")
            
            # =========================================================
            # MODIFICAÇÃO: VISUALIZAÇÃO DOS ACASALAMENTOS (TABELA)
            # =========================================================
            print(f"\n   >>> ACASALAMENTOS SUGERIDOS (Rodada {r+1}, MaxUso {max_uso}):")
            print(f"   {'Fêmea':<20} | {'Macho':<20} | {'Índice Estimado':<15}")
            print("   " + "-"*60)
            
            # Itera pela solução e mapeia de volta para os nomes
            for i_femea, i_macho in enumerate(sol):
                nome_femea = femeas_nomes[i_femea]
                nome_macho = machos_nomes[i_macho]
                
                # Opcional: Mostra o valor genético desse casal específico
                valor_casal = matriz_indices[i_femea, i_macho]
                
                print(f"   {str(nome_femea):<20} x {str(nome_macho):<20} | {valor_casal:.4f}")
            print("   " + "-"*60 + "\n")
            # =========================================================

        else:
            print("Falhou (Inviável - Fêmeas demais para poucos machos).")

    if not resultados_run:
        continue

    # --- Estatísticas Finais das Rodadas (Item 7.3) ---
    vals_coanc = [r[0] for r in resultados_run]
    vals_idx = [r[1] for r in resultados_run]
    
    print(f"\n  Estatísticas Finais (max_uso={max_uso}):")
    
    print(f"  Coancestralidade -> Min: {min(vals_coanc):.4f}, Max: {max(vals_coanc):.4f}")
    print(f"                      Média: {np.mean(vals_coanc):.4f}, Mediana: {np.median(vals_coanc):.4f}")
    
    print(f"  Índice Total     -> Min: {min(vals_idx):.4f}, Max: {max(vals_idx):.4f}")
    print(f"                      Média: {np.mean(vals_idx):.4f}, Mediana: {np.median(vals_idx):.4f}")
    
    # Gera o gráfico de convergência
    plotar_graficos(historicos_run, f"Convergência - Max Uso {max_uso}")

print("\n=== FIM DA EXECUÇÃO ===")
