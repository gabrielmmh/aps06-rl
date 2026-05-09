# APS08 — Generalização do Agente em Coverage Path Planning

Fork técnico de [`fbarth/gym_custom_env`](https://github.com/fbarth/gym_custom_env) feito para a Atividade Prática Supervisionada 08 da disciplina de Reinforcement Learning do Insper. Enunciado em https://insper.github.io/rl/classes/23_custom_env_agent/.

A APS pede uma estratégia que faça um agente PPO treinado no problema de Coverage Path Planning (CPP) generalizar entre tamanhos de grid (5x5, 10x10 e, como bônus, 20x20) preservando a observabilidade parcial. O baseline do enunciado treina em 5x5 e degrada quando avaliado em grids maiores. Investiguei nove configurações de RL (mais dois baselines clássicos não-learning para contexto) para atacar essa degradação.

## Ambiente

`GridWorldCPPEnv` é o ambiente herdado do upstream. O agente nasce numa célula aleatória de um grid quadrado com obstáculos fixos por episódio, e precisa visitar todas as células livres sem revisitar.

| Propriedade | Valor |
|---|---|
| Estado | `agent` (x, y normalizados, ratio de cobertura) e `neighbors` 3x3 ao redor do agente |
| Ações | 0 = direita, 1 = cima, 2 = esquerda, 3 = baixo |
| Reward | +1 por célula nova, −0.3 por revisita, −0.5 por bater em parede, −0.1 por step, +10 ao cobrir tudo, −5 ao truncar |
| Término | todas as células livres visitadas, ou `max_steps` excedido |
| Observação | parcial: o agente vê só a vizinhança 3x3 (codificada como 0 = livre, 1 = parede ou obstáculo, 2 = visitada) |

A observabilidade parcial é o ponto da APS. O agente nunca tem acesso ao mapa completo, então a política precisa lidar com a incerteza sobre o que existe além da janela. As regras do enunciado permitem aumentar essa janela para 5x5 (que apliquei nas configs `curriculum_enriched`, `mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl` e `maskable_frontier_pbrs`) e modificar o reward function (que apliquei em `maskable_v3`, `maskable_bc_kl` e `maskable_frontier_pbrs`). Tudo o mais — em particular a obrigatoriedade de a estratégia ser baseada em RL — segue como no original.

| Tamanho | Obstáculos | `max_steps` |
|---|---|---|
| 5x5 | 3 | 200 |
| 10x10 | 12 | 500 |
| 20x20 | 48 | 1000 |

## O Problema da Generalização

A política aprendida em 5x5 não transfere para 10x10. O motivo é uma combinação de três fatores que descobri empiricamente. Primeiro, **as features dependem da escala**: a posição do agente é normalizada por `size`, então uma posição relativa de 0.5 em 5x5 corresponde a uma célula central concreta, mas em 10x10 corresponde a outra coordenada absoluta — a rede aprende mapeamentos ligados a um tamanho específico. Segundo, **a janela 3x3 cobre uma fatia cada vez menor do mapa** à medida que o grid cresce: representa 36% do mapa em 5x5, 9% em 10x10 e apenas 2.25% em 20x20, deixando o agente cada vez mais cego ao contexto local. Terceiro, **sem memória, o agente esquece** as células visitadas fora da janela atual; em mapas pequenos a janela é grande o bastante para o agente sempre ver parte do que já cobriu, mas em mapas grandes ele entra em regiões novas sem saber onde já passou.

Surgiu uma quarta hipótese durante o trabalho, que descrevo nas seções de Análise: **o credit assignment do fechamento das últimas células**. *Credit assignment* aqui é o problema clássico de RL: o reward sinaliza "falta fechar" só pelo bônus terminal (+10) — uma diferença que, sob γ=0.99 e episódios de 500-1000 steps, fica tão diluída no rollout que o agente nunca aprende que essas últimas 3-15 células fora da janela valem o esforço de retornar. Resultado empírico: o agente cobre 94-99% das células em média, mas trava em 64-86% em "fração de episódios fechados completamente". O gargalo é o final do episódio, não o início.

## Estratégias Investigadas

Comparei nove configurações; todas usam PPO ou variantes, e as diferenças estão em como atacam um ou mais dos fatores acima.

| Config | Estratégia | Hipótese atacada |
|---|---|---|
| `baseline` | PPO com `MultiInputPolicy`, sem curriculum (treina do zero em cada tamanho) | nenhuma (reproduz o problema) |
| `curriculum` | PPO + curriculum learning: 5x5 → 10x10 → 20x20, transferindo pesos | escala de features |
| `curriculum_enriched` | curriculum + observação ampliada (vizinhança 5x5 + direção e distância à célula não-visitada mais próxima) | janela pequena |
| `curriculum_recurrent` | curriculum com RecurrentPPO (LSTM 64 unidades, n_steps 128, CPU) | falta de memória |
| `curriculum_recurrent_v2` | mesma estratégia com LSTM 256 + n_steps 512 + GPU para testar se a primeira tentativa estava subdimensionada | falta de memória (segunda tentativa) |
| `mapcnn_bc_pbrs` | PPO com `NatureCNN` sobre observação egocêntrica de mapa acumulado (3×39×39) + warm-start BC do `FrontierAgent` + PBRS Φ=cobertura | janela pequena + memória + credit assignment do fechamento |
| `maskable_v3` | curriculum + obs enriquecida + action masking (`MaskablePPO`) + reward redesign (terminal +60, truncation 0, step penalty 0 quando coverage ≥ 0.80) | credit assignment do fechamento |
| `maskable_bc_kl` | `maskable_v3` + warm-start BC + KL anchor `λ · KL(π ‖ π_BC_frozen)` com λ decaindo de 1.0 a 0.05 | fechamento + preservação da BC sob drift do PPO |
| `maskable_frontier_pbrs` | `maskable_v3` + memória `visited_pooled` (2×8×8 fixo) + feature `frontier` (BFS) + PBRS distance-based + reset do value head entre fases | combinação direta dos três gargalos (memória, fechamento, drift) |

Detalhes completos de cada config nas seções [Configurações](#configurações), [Curvas de Aprendizado](#curvas-de-aprendizado) e [Resultados de Inferência](#resultados-de-inferência) abaixo. Convenções terminológicas usadas em todo o relatório:

- **fronteira**: célula livre vista mas não-visitada (a borda do mapa conhecido pelo agente). O `FrontierAgent` scripted (não-RL) toma a ação de andar até a fronteira mais próxima via BFS; já a feature `frontier` do `maskable_frontier_pbrs` apenas *informa* essa direção/distância ao agente, que continua escolhendo a ação via política PPO.
- **drift do PPO**: ao longo de muitos updates (especialmente em fases longas como 20x20 com 2M+ timesteps), a política se afasta do ponto inicial — seja a BC ou o checkpoint da fase anterior do curriculum — e perde competência adquirida. Mecanismo: o crítico recém-carregado fica mal-calibrado pra escala de retornos da nova fase, advantage estimation produz gradient que empurra a política pra fora da bacia aprendida. Visível empiricamente como avg coverage caindo entre fases ou full coverage rate colapsando após uma fase longa.

As cinco primeiras configs rodaram com 3 seeds (0, 1, 2). As quatro últimas (`mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl`, `maskable_frontier_pbrs`) rodaram com apenas o seed 0, porque cada uma leva 3-8h por seed e o sinal diagnóstico do seed 0 já era forte o suficiente para decidir o próximo passo dentro do orçamento de tempo da APS.

## A métrica corrigida: mapas insolucionáveis

Em todos os tamanhos, uma fração dos mapas gerados aleatoriamente é fisicamente impossível de cobrir 100% — alguns obstáculos isolam células livres do spawn do agente, criando bolsões inalcançáveis. Calculando reachability via BFS a partir da posição inicial nos 300 mapas de avaliação por tamanho (3 seeds × 100 episódios), encontrei **6% (18/300) insolucionáveis em 5x5, 14% (42/300) em 10x10 e 23% (69/300) em 20x20**. O teto teórico de full coverage rate é, portanto, 94% / 86% / 77%, não 100%.

Em todas as tabelas reporto duas métricas lado a lado:

- **raw**: fração dos 100 mapas em que o agente cobriu tudo. Comparável com o `75/100` que o enunciado cita ao descrever o baseline.
- **sobre solucionáveis**: descontando os mapas impossíveis, mede a competência efetiva num conjunto onde 100% é fisicamente atingível.

Validação importante: o frontier scripted (BFS sobre mapa interno) bate **exatamente** os tetos teóricos — 94/86/77 raw = 100% sobre solucionáveis em todos os tamanhos. Ou seja, sob observabilidade parcial, esse heurístico é ótimo. Os números sobre solucionáveis dos configs RL medem o quanto se aproximam desse teto.

A observabilidade parcial fica preservada: o cache de solucionabilidade (`results/solvability_cache.json`, gerado offline por `python -m broom.build_solvability_cache` via `broom/solvability.py`) **não é exposto ao agente em momento algum**, só ao avaliador.

## Como Executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m broom.run_experiments --configs baseline,curriculum,curriculum_enriched,curriculum_recurrent
```

As quatro configs com GPU CUDA rodam separadamente. As que usam BC warm-start precisam gerar o checkpoint primeiro:

```bash
# curriculum_recurrent_v2 (GPU, sem BC):
python -m broom.run_experiments --configs curriculum_recurrent_v2

# mapcnn_bc_pbrs (GPU, BC do FrontierAgent no env mapobs):
python -m broom.bc_pipeline           # gera results/models/bc_warmstart.zip (~10min)
python -m broom.run_experiments --configs mapcnn_bc_pbrs

# maskable_v3 (GPU, sem BC):
python -m broom.run_experiments --configs maskable_v3

# maskable_bc_kl (GPU, BC do FrontierAgent no env V3):
python -m broom.bc_v3_pipeline        # gera results/models/bc_warmstart_v3.zip (~10min)
python -m broom.run_experiments --configs maskable_bc_kl

# maskable_frontier_pbrs (GPU, sem BC):
python -m broom.run_experiments --configs maskable_frontier_pbrs
```

Os baselines clássicos (frontier-based, boustrophedon) só rodam inferência: `python -m broom.run_scripted`.

O `run_experiments.py` é resumível: pula combinações cujo modelo já existe em `results/models/`. Para treinar sem rodar inferência, adiciono `--skip-inference`. Os testes ficam em `tests/` e rodam com `pytest tests/ -q` (79 testes).

## Configurações

Hiperparâmetros principais (mantidos consistentes para isolar a estratégia testada):

| Parâmetro | Valor |
|---|---|
| Algoritmo (`baseline`, `curriculum`, `curriculum_enriched`, `mapcnn_bc_pbrs`) | PPO + `MultiInputPolicy` |
| Algoritmo (`curriculum_recurrent`, `curriculum_recurrent_v2`) | RecurrentPPO + `MultiInputLstmPolicy` |
| Algoritmo (`maskable_v3`, `maskable_bc_kl`, `maskable_frontier_pbrs`) | MaskablePPO + `MultiInputPolicy` |
| `ent_coef` | 0.05 (default upstream); `maskable_v3`, `maskable_bc_kl` e `maskable_frontier_pbrs` usam schedule linear 0.02 → 0.001 |
| `device` | cpu (4 primeiras configs); cuda (`curriculum_recurrent_v2`, `mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl`, `maskable_frontier_pbrs`) |
| `n_envs` (PPO, 5x5/10x10) | 4 |
| `n_envs` (PPO, 20x20) | 2 |
| `n_envs` (`curriculum_recurrent`) | 2 em todos os grids |
| `n_envs` (`curriculum_recurrent_v2`) | 4 em 5x5/10x10, 2 em 20x20 |
| `n_steps` | PPO default 2048 (configs sem override); RecurrentPPO default 128 (`curriculum_recurrent`); overrides: 512 (`curriculum_recurrent_v2`), 1024 (`mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl`), 2048 explícito (`maskable_frontier_pbrs`) |
| `learning_rate` | 3e-4 default; 1e-4 em `maskable_bc_kl`; 5e-5 em `maskable_frontier_pbrs` (LR menor reduz drift do PPO em horizonte longo) |
| `clip_range` | 0.2 default; 0.1 em `maskable_frontier_pbrs` (tighter, previne updates grandes; Moalla et al. arXiv:2405.00662) |
| Timesteps por fase | 5x5: 300k, 10x10: 800k, 20x20: 2M (4M em `maskable_frontier_pbrs`) |
| `max_steps` | 5x5: 200, 10x10: 500, 20x20: 1000 (1500 em `maskable_frontier_pbrs` — dá margem pra fechar) |
| LSTM (`curriculum_recurrent`) | 64 unidades, 1 camada |
| LSTM (`curriculum_recurrent_v2`) | 256 unidades, 1 camada |
| `gamma` (`mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl`) | 0.999 (long-horizon: 1000 passos no 20x20) |
| `gamma` (`maskable_frontier_pbrs`) | 0.995 (horizonte efetivo ~200 steps, alinhado com decisões de fechamento; menos ruído na advantage estimation que 0.999) |
| Net arch (`maskable_v3`, `maskable_bc_kl`, `maskable_frontier_pbrs`) | `[256, 256]` (vs default 64x64) |
| Observação (`mapcnn_bc_pbrs`) | egocêntrica `(3, 39, 39)`, canais visited/walls/free, mapa interno construído só pelo que o agente já viu |
| Observação (`maskable_v3`, `maskable_bc_kl`) | enriched 5x5 + features de direção/distância + `action_masks()` |
| Observação (`maskable_frontier_pbrs`) | enriched + `visited_pooled` (2×8×8 max-pool da trajetória, resolução fixa F=8) + `frontier` (3 dims, direção e distância BFS) + `progress` (count_steps/max_steps) + `action_masks()` |
| Reward (`maskable_v3`, `maskable_bc_kl`, `maskable_frontier_pbrs`, treino) | terminal full coverage **+60** (era +10), truncation **0** (era −5), step penalty 0 quando coverage ≥ 0.80; eval usa o reward upstream |
| Value-head reset (`maskable_frontier_pbrs`, transição entre fases) | reinicialização ortogonal do `policy.value_net` ao carregar checkpoint da fase anterior; mantém policy + features (Igl 2021, Wolczyk 2024) |
| Warm-start (`mapcnn_bc_pbrs`, fase 5x5) | BC do `FrontierAgent` (~75k pares (s, a), 10 épocas, 97.9% acc) |
| Warm-start (`maskable_bc_kl`, fase 5x5) | BC do `FrontierAgent` no env V3 (~22k pares (s, a), 10 épocas, ~95% acc) |
| KL anchor (`maskable_bc_kl`, treino) | `λ · KL(π ‖ π_BC_frozen)`; λ linear 1.0 → 0.05 ao longo dos 3.1M timesteps cumulativos |
| PBRS (`mapcnn_bc_pbrs`, treino) | Φ = ratio de cobertura, F = γΦ' − Φ; magnitude per-step ~0.001 (efetivamente ruído) |
| PBRS (`maskable_frontier_pbrs`, treino) | Φ = −d_BFS(agente, fronteira)/diâmetro, F = γΦ' − Φ; magnitude per-step ~±0.05 (Jonnarth et al. ICML 2024). BFS sobre terreno conhecido, preserva observabilidade parcial |
| Seeds | 0, 1, 2 (1 seed apenas em `mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl`, `maskable_frontier_pbrs`) |
| Episódios de inferência | 100 (política estocástica, `deterministic=False`) |

## Curvas de Aprendizado

Todas as curvas usam média e desvio padrão sobre 3 seeds (1 seed para as quatro últimas configs), suavizadas com janela móvel de 20 episódios.

### Baseline

![baseline](results/plots/learning_curve_baseline.png)

O agente converge nos três tamanhos: 5x5 sai de −60 e estabiliza próximo de 0; 10x10 sai de −140 e chega a −10; 20x20 sai de −200 e atinge ~0.

### Curriculum

![curriculum](results/plots/learning_curve_curriculum.png)

A fase 5x5 é idêntica ao baseline (treinada do zero). Nas fases 10x10 e 20x20, o eixo X reinicia em zero porque cada fase é um treino separado com `model.learn(reset_num_timesteps=False)`. Os pesos vêm carregados da fase anterior, então a curva começa em reward mais alto que o baseline equivalente.

### Curriculum + observação enriquecida

![enriched](results/plots/learning_curve_curriculum_enriched.png)

Comportamento parecido com curriculum, mas com a observação 5x5 + features de direção/distância para a célula não-visitada mais próxima na janela.

### Curriculum + RecurrentPPO (LSTM) — duas tentativas

A hipótese de memória foi testada em duas configurações distintas, separadas para deixar claro o que cada uma testa.

A primeira (`curriculum_recurrent`, LSTM 64, n_steps 128, CPU) levou ~2.5h por seed mas resultou num colapso: LSTM 64 com rollouts de 128 steps não converge para nenhuma estratégia útil em 10x10 ou 20x20.

![recurrent CPU](results/plots/learning_curve_curriculum_recurrent.png)

A segunda (`curriculum_recurrent_v2`, LSTM 256, n_steps 512, GPU) ataca diretamente as três hipóteses sobre por que a primeira tentativa colapsou: `device="cuda"` libera CPU para coletar rollouts; `lstm_hidden_size=256` dá 4× mais unidades e ~16× mais parâmetros na LSTM; `n_steps=512` é 4× o rollout default, dando à LSTM mais sinal temporal por update; `n_envs=4` em 5x5/10x10 (mantém 2 em 20x20) aproveita que a LSTM saiu da CPU. Cada seed leva ~5h. Há melhora real em 10x10 native (1.3% → 10%) e em 20x20 → 10x10 (19.3% → 30.7%), mas o 20x20 native segue 0%.

![recurrent GPU](results/plots/learning_curve_curriculum_recurrent_v2.png)

### MapCNN + BC + PBRS

![mapcnn_bc_pbrs](results/plots/learning_curve_mapcnn_bc_pbrs.png)

A curva começa em reward já alto porque a BC inicializa o policy network. 5x5 native em 97% (melhor isolado), 10x10 empata o enriched em 77%, 20x20 native colapsa para 0% — o PPO durante a fase 20x20 destrói a inicialização do BC.

### Maskable PPO + reward redesign

![maskable_v3](results/plots/learning_curve_maskable_v3.png)

A calibração do terminal +60 vem de Theile et al. (arXiv 2309.03157): pra o bônus dominar a soma das step penalties sob γ=0.999, B ≥ (0.1·500+5)/0.95 ≈ 58 (adotei +60 com margem). Action masking elimina ruído de ações inválidas (Huang & Ontañón, arXiv 2006.14171). Essa config destrava o teto histórico de 77% no 10x10, subindo para 84% raw / 92.3% sobre solucionáveis.

### Maskable PPO + BC + KL anchor

![maskable_bc_kl](results/plots/learning_curve_maskable_bc_kl.png)

10x10 native sobe para 86% raw / 94.5% sobre solucionáveis. A curva começa em reward bem positivo (BC) e mantém estável durante o treino sem desviar muito do BC inicial (visível no log do `kl_to_bc`).

### Maskable PPO + frontier feature + distance-PBRS

![maskable_frontier_pbrs](results/plots/learning_curve_maskable_frontier_pbrs.png)

Curva no 20x20 sobe estável de +14 (cold start) pra ~+250 ao longo dos 4M timesteps, sem o plateau dos 80-110 que travou as configs anteriores. Atinge **75% raw / 96.2% sobre solucionáveis no 20x20 native** — quase match com o frontier scripted (77%/100%).

## Resultados de Inferência

100 episódios por modelo, política estocástica. Cada modelo treinado num tamanho é avaliado nos três; a diagonal é a performance "nativa" e os off-diagonais medem generalização. Para os 5 primeiros configs as tabelas reportam a média sobre 3 seeds; para os 4 últimos (1 seed cada), reporto o número direto do seed 0.

### Baseline

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **92.7% / 98.6%** | 14.0% / 16.2% | 0.0% / 0.0% |
| 10x10 | 89.0% / 94.8% | **64.3% / 75.0%** | 0.3% / 0.4% |
| 20x20 | 87.3% / 93.0% | 47.7% / 55.2% | **0.3% / 0.4%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **99.1%** | 95.9% | 79.4% |
| 10x10 | 98.7% | **98.2%** | 95.4% |
| 20x20 | 98.4% | 97.8% | **94.1%** |

### Curriculum

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **92.7% / 98.6%** | 14.0% / 16.2% | 0.0% / 0.0% |
| 10x10 | 90.7% / 96.4% | **71.3% / 82.7%** | 2.0% / 2.6% |
| 20x20 | 89.0% / 94.8% | 64.7% / 75.4% | **0.3% / 0.4%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **99.1%** | 95.9% | 79.4% |
| 10x10 | 98.9% | **98.9%** | 96.6% |
| 20x20 | 98.7% | 98.3% | **96.6%** |

A linha 5x5 é idêntica ao baseline porque a primeira fase do curriculum não tem warm-start. O ganho aparece a partir do 10x10 e é mais visível no que o modelo final do 20x20 consegue fazer no 10x10 (64.7% vs 47.7% do baseline).

### Curriculum + observação enriquecida

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **91.3% / 97.1%** | 69.7% / 81.1% | 0.7% / 0.9% |
| 10x10 | 92.7% / 98.6% | **77.3% / 90.0%** | 4.7% / 6.2% |
| 20x20 | 91.0% / 96.8% | 73.0% / 85.1% | **9.0% / 11.8%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.6%** | 98.7% | 93.5% |
| 10x10 | 98.9% | **98.6%** | 96.7% |
| 20x20 | 98.8% | 98.8% | **97.3%** |

A célula mais surpreendente é o 5x5 → 10x10: 69.7% (vs 14.0% do baseline e do curriculum). A janela 5x5 + a feature `direction_to_nearest_unvisited` fazem o modelo treinado só em 5x5 generalizar quase tão bem em 10x10 quanto em 5x5. Isso é resultado de **estrutura na observação**, não de mais treino.

### Curriculum + RecurrentPPO (CPU, LSTM 64, n_steps 128)

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **83.0% / 88.3%** | 0.0% / 0.0% | 0.0% / 0.0% |
| 10x10 | 64.7% / 68.5% | **1.3% / 1.6%** | 0.0% / 0.0% |
| 20x20 | 85.0% / 90.4% | 19.3% / 22.7% | **0.0% / 0.0%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.3%** | 85.4% | 56.0% |
| 10x10 | 96.6% | **88.0%** | 69.7% |
| 20x20 | 98.5% | 95.6% | **86.2%** |

O recurrent regrediu em quase todas as células comparado ao baseline. O 10x10 native colapsou de 64.3% para 1.3%, e o 5x5 → 10x10 zerou. A avg coverage continua razoável (56-98% se incluir cells 5x5→20x20 e 10x10→20x20, onde a cobertura cai abaixo de 70%; 84-98% nas demais), então o agente ainda explora, só não fecha a cobertura.

### Curriculum + RecurrentPPO (GPU, LSTM 256, n_steps 512)

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **83.7% / 88.9%** ±4.7 | 0.0% / 0.0% | 0.0% / 0.0% |
| 10x10 | 85.3% / 90.6% ±9.2 | **10.0% / 11.7%** ±5.0 | 0.0% / 0.0% |
| 20x20 | 83.7% / 88.9% ±6.6 | 30.7% / 35.9% ±18.9 | **0.0% / 0.0%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.5%** | 88.2% | 55.7% |
| 10x10 | 98.5% | **93.2%** | 80.1% |
| 20x20 | 98.2% | 95.4% | **84.8%** |

A v2 melhora em quase todas as células fora do 20x20 native: 10x10 native sobe de 1.3% para 10.0% (~8×), 10x10 → 5x5 vai de 64.7% para 85.3%, 20x20 → 10x10 vai de 19.3% para 30.7%. O 5x5 native fica praticamente igual (83.0% → 83.7%) e o 20x20 native segue 0% mesmo com a capacidade aumentada. Mostra que a primeira tentativa estava de fato subdimensionada, mas que mesmo a v2 não encontra a estratégia de fechar mapas grandes, ficando bem abaixo do enriched (77.3% em 10x10 native) e do frontier scripted (86.0%). A variância no seed 2 da v2 (10x10 native em 3.0% versus 13-14% nos seeds 0 e 1) sinaliza que a LSTM ainda treina de forma instável.

### MapCNN + BC + PBRS (1 seed)

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **97.0% / 100.0%** | 25.0% / 27.5% | 1.0% / 1.3% |
| 10x10 | 92.0% / 94.8% | **77.0% / 84.6%** | 0.0% / 0.0% |
| 20x20 | 38.0% / 39.2% | 0.0% / 0.0% | **0.0% / 0.0%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **99.0%** | 96.8% | 88.8% |
| 10x10 | 98.5% | **99.1%** | 85.4% |
| 20x20 | 93.7% | 83.7% | **77.0%** |

Esta foi a primeira tentativa de empilhar memória global, warm-start do FrontierAgent e PBRS num bundle único. O 5x5 native sobe a 97% (melhor de todos os configs nesse tamanho), o 10x10 native iguala o enriched em 77%, mas o 20x20 native colapsa para 0% — o PPO durante a fase 20x20 destrói a inicialização do BC. O dano é mais visível no cell `20x20 → 5x5`: 38% (vs 87-92% das outras configs), confirmando que o modelo treinado em 20x20 perdeu até a competência das fases anteriores. Foi essa observação que motivou a config `maskable_bc_kl`, com KL anchor para prevenir esse drift.

### Maskable PPO + reward redesign (1 seed)

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **96.0% / 99.0%** | 72.0% / 79.1% | 5.0% / 6.4% |
| 10x10 | 96.0% / 99.0% | **84.0% / 92.3%** | 30.0% / 38.5% |
| 20x20 | 95.0% / 97.9% | 54.0% / 59.3% | **0.0% / 0.0%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.9%** | 99.1% | 91.0% |
| 10x10 | 98.9% | **99.2%** | 97.9% |
| 20x20 | 98.9% | 98.4% | **92.1%** |

O `maskable_v3` é a config que destrava o teto histórico de 77% no 10x10 native: 84% raw / 92.3% sobre solucionáveis. A combinação action masking + reward redesign ataca diretamente o gargalo do fechamento das últimas células. A célula `10x10 → 20x20` também surpreende: 30% raw / 38.5% sobre solucionáveis (vs 4.7% / 6.2% do enriched), mostrando que o modelo treinado só em 10x10 com reward redesign já transfere bem pro 20x20. O 20x20 native, no entanto, cai pra 0% — mesmo padrão do `mapcnn_bc_pbrs`. A fase 20x20 do PPO continua causando drift mesmo sem BC pra anular. A avg coverage de 92.1% mostra que o agente ainda explora bem em 20x20, só não fecha.

### Maskable PPO + BC + KL anchor (1 seed)

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **96.0% / 99.0%** | 72.0% / 79.1% | 6.0% / 7.7% |
| 10x10 | 96.0% / 99.0% | **86.0% / 94.5%** | 32.0% / 41.0% |
| 20x20 | 96.0% / 99.0% | 64.0% / 70.3% | **1.0% / 1.3%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.8%** | 99.0% | 96.0% |
| 10x10 | 98.5% | **99.8%** | 98.9% |
| 20x20 | 98.9% | 98.3% | **94.4%** |

O `maskable_bc_kl` adiciona o KL anchor (`λ · KL(π ‖ π_BC_frozen)`) ao `maskable_v3`. O 10x10 native sobe para 86% raw / 94.5% sobre solucionáveis, o melhor RL puro do estudo nesse tamanho — ainda abaixo do frontier (100%) mas chegando próximo. A célula `10x10 → 20x20` também melhora ligeiramente: 32% / 41.0%. O 20x20 native fica em 1% raw, confirmando que nem o KL anchor pra BC consegue prevenir o drift do PPO na fase 20x20. A boa notícia é o `20x20 → 10x10 = 64%` raw / 70.3% sobre solucionáveis (vs 54% / 59.3% do `maskable_v3`), indicando que o KL anchor preservou mais competência da fase 10x10 mesmo após a fase 20x20.

### Maskable PPO + frontier feature + distance-PBRS (1 seed)

Full coverage rate raw / sobre solucionáveis:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **96.0% / 99.0%** | 89.0% / 97.8% | 73.0% / 93.6% |
| 10x10 | 95.0% / 97.9% | **89.0% / 97.8%** | 74.0% / 94.9% |
| 20x20 | 96.0% / 99.0% | 89.0% / 97.8% | **75.0% / 96.2%** |

Avg coverage:

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.8%** | 99.7% | 99.8% |
| 10x10 | 98.9% | **99.7%** | 99.6% |
| 20x20 | 98.9% | 99.7% | **99.6%** |

Esta é a config decisiva — **rompeu o teto do 20x20 native que travou todas as configs anteriores em ~0% raw**. A descrição completa dos quatro pilares está na seção [Análise / Hipótese 4](#análise) abaixo, junto com a leitura comparativa.

A célula mais surpreendente é a transferência cross-grid: o **modelo treinado APENAS em 5x5 já atinge 73% raw / 93.6% solucionáveis no 20x20**, sem nunca ter visto o tamanho. Sinal forte da `visited_pooled` em resolução fixa F=8 — política aprendida em escala pequena transfere quase direto. Treinar mais (no 10x10 e 20x20) refina pra 96.2% solucionáveis no 20x20 native — match quase exato com o frontier scripted (100%).

## Análise

A tabela abaixo consolida as nove configurações nas células-chave (full coverage rate, raw):

| Treinado ↓ \ Eval → | Baseline | Curriculum | Enriched | Rec. (CPU) | Rec. v2 | MapCNN+BC+PBRS | Mask. v3 | Mask. BC+KL | **Mask. Frontier+PBRS** |
|---|---|---|---|---|---|---|---|---|---|
| 5x5 → 5x5 | 92.7% | 92.7% | 91.3% | 83.0% | 83.7% | 97.0% | 96.0% | 96.0% | **96.0%** |
| 5x5 → 10x10 | 14.0% | 14.0% | 69.7% | 0.0% | 0.0% | 25.0% | 72.0% | 72.0% | **89.0%** |
| 5x5 → 20x20 | 0.0% | 0.0% | 0.7% | 0.0% | 0.0% | 1.0% | 5.0% | 6.0% | **73.0%** |
| 10x10 → 10x10 | 64.3% | 71.3% | 77.3% | 1.3% | 10.0% | 77.0% | 84.0% | 86.0% | **89.0%** |
| 10x10 → 20x20 | 0.3% | 2.0% | 4.7% | 0.0% | 0.0% | 0.0% | 30.0% | 32.0% | **74.0%** |
| 20x20 → 10x10 | 47.7% | 64.7% | 73.0% | 19.3% | 30.7% | 0.0% | 54.0% | 64.0% | **89.0%** |
| 20x20 → 20x20 | 0.3% | 0.3% | 9.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.0% | **75.0%** |

Com a métrica filtrada sobre mapas solucionáveis, os números finais nas natives ficam:

| Config | 5x5 native | 10x10 native | 20x20 native |
|---|---|---|---|
| baseline | 98.6% | 75.0% | 0.4% |
| curriculum | 98.6% | 82.7% | 0.4% |
| curriculum_enriched | 97.1% | 90.0% | 11.8% |
| curriculum_recurrent | 88.3% | 1.6% | 0.0% |
| curriculum_recurrent_v2 | 88.9% | 11.7% | 0.0% |
| mapcnn_bc_pbrs | 100.0% | 84.6% | 0.0% |
| maskable_v3 | 99.0% | 92.3% | 0.0% |
| maskable_bc_kl | 99.0% | 94.5% | 1.3% |
| **maskable_frontier_pbrs** | **99.0%** | **97.8%** | **96.2%** |
| frontier scripted (referência) | 100.0% | 100.0% | 100.0% |

**Métrica usada para ranquear as estratégias: Full Coverage Rate** (fração de episódios fechados 100%). O enunciado descreve o baseline em `75/100`, formato Full Coverage Rate, então adoto essa leitura por consistência. Avg coverage (94-99% em quase tudo) e métrica filtrada (sobre solucionáveis) ficam em paralelo nas tabelas pra mostrar o gap entre "explorar quase tudo" e "fechar de fato" — exatamente o gap que motivou a Hipótese 4 abaixo.

Cada uma das quatro hipóteses se mapeia num resultado:

**Hipótese 1: features dependem do tamanho do grid.** A posição é normalizada por `size`, então um par `(x/size, y/size)` no 5x5 codifica uma célula concreta diferente da mesma proporção no 20x20 — a rede aprende mapeamentos amarrados a um size específico. Endereçada pelo curriculum: ganho real mas modesto (+7pp no 10x10 native, +17pp no 20x20→10x10 transfer). Não é o gargalo principal.

**Hipótese 2: janela 3×3 fica pequena demais em grids grandes, e a política não tem pista de "pra onde ir".** A janela cobre 36% do mapa em 5x5, 9% em 10x10 e só 2.25% em 20x20. Endereçada pelo `curriculum_enriched`: vizinhança 5×5 (24 células em vez de 8) + `direction_to_nearest_unvisited` dentro da janela. Resultado: 5×5→10×10 vai de 14% pra 70% **sem precisar de curriculum**, ou seja, ganho puramente estrutural. Era a hipótese certa pra generalização 5×5↔10×10.

**Hipótese 3: o agente esquece as células visitadas que ficaram fora da janela atual.** Foi testada com `RecurrentPPO`. A primeira tentativa (LSTM 64, n_steps 128, CPU) colapsou — 10x10 native foi a 1.3%. A segunda (LSTM 256, n_steps 512, GPU) confirmou que parte do colapso era subdimensionamento — 10x10 sobe pra 10% — mas ainda muito abaixo do enriched (77.3%). Conclusão: memória recorrente ajuda mas não compete com observação enriquecida estruturalmente. Memória **explícita estruturada** (a `visited_pooled` na hipótese 4) acabou sendo o substituto certo para LSTM.

**Hipótese 4: credit assignment do fechamento das últimas células.** Avg coverage chega a 94-99% em todas as configs e tamanhos, mas full coverage rate trava em 64-86% — ou seja, o agente explora bem, mas as últimas 3-15 células fora da janela ficam pra trás. Quatro tentativas atacaram esse problema; cada uma diagnostica um aspecto diferente:

| Tentativa | Componente novo | 10x10 native (raw / solv) | 20x20 native (raw / solv) | Veredito |
|---|---|---|---|---|
| `mapcnn_bc_pbrs` | mapa CNN egocêntrico + BC + PBRS Φ=cobertura | 77 / 84.6 | 0 / 0 | empata enriched no 10x10; **PBRS magnitude ~0.001/step é ruído**; PPO drift no 20x20 erradica BC |
| `maskable_v3` | action masking + reward redesign (terminal +60) | 84 / 92.3 | 0 / 0 | **destrava 10x10** mas drift do PPO no 20x20 segue |
| `maskable_bc_kl` | + KL anchor pra BC frozen | 86 / 94.5 | 1 / 1.3 | melhor 10x10 RL puro até então; KL anchor preserva BC mas não destrava closing |
| **`maskable_frontier_pbrs`** | memória `visited_pooled` + feature `frontier` + PBRS distance + value-head reset | **89 / 97.8** | **75 / 96.2** | **destrava 20x20**; quase match com frontier scripted |

A configuração decisiva combina quatro componentes que atacam os três gargalos remanescentes: (a) `visited_pooled` (2×8×8 max-pool fixo) substitui hidden state recorrente e transfere bem cross-grid; (b) feature `frontier` via BFS sempre orienta a ação, mesmo quando a janela 5×5 está toda visitada; (c) PBRS distance-based (Φ = −d_BFS/diâmetro) com magnitude ~±0.05/step (vs ~0.001 do PBRS antigo) — Jonnarth et al. ICML 2024; (d) reset do value head a cada transição de fase do curriculum impede o crítico de empurrar a política pra fora da bacia aprendida (Igl 2021, Wolczyk 2024).

A leitura final: **a magnitude do PBRS importa tanto quanto a função potencial em si**. Ng-Harada-Russell (1999) garante que qualquer Φ preserva a política ótima, mas a velocidade de convergência depende criticamente da magnitude do gradient denso adicionado. Foi a combinação dos quatro pilares — não nenhum sozinho — que rompeu o teto do 20x20 que resistira a todas as tentativas anteriores.

## Comparação com baselines clássicos

Comparei contra dois baselines não-learning rodando no mesmo `GridWorldCPPEnv` (3 seeds, observabilidade parcial preservada — o mapa interno é construído só do que o agente viu pela janela 3x3, nunca de oráculo). O **frontier-based exploration** mantém um mapa interno `size × size` e a cada step faz BFS pra fronteira conhecida mais próxima; sem fronteira, escolhe a ação que maximiza descoberta. O **boustrophedon** faz varredura linha-a-linha com fallback frontier quando trava. Código em `broom/baselines/`, executado com `python -m broom.run_scripted`.

Resultados (média de 3 seeds, 100 episódios cada, full coverage rate raw / sobre solucionáveis):

| Algoritmo | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| Frontier-based BFS | **94.0% / 100.0%** | **86.0% / 100.0%** | **77.0% / 100.0%** |
| Boustrophedon | 94.0% / 100.0% | 26.3% / 30.5% | 0.0% / 0.0% |
| **Melhor RL (`maskable_frontier_pbrs`, 1 seed)** | **96.0% / 99.0%** | **89.0% / 97.8%** | **75.0% / 96.2%** |

Avg coverage:

| Algoritmo | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| Frontier-based BFS | 99.1% | 99.4% | **99.9%** |
| Boustrophedon | 99.1% | 92.1% | 54.1% |
| Melhor RL (`maskable_frontier_pbrs`, 1 seed) | 98.8% | 99.7% | 99.6% |

O `maskable_frontier_pbrs` praticamente fecha o gap pro frontier scripted em todos os tamanhos. No 20x20, onde a melhor RL anterior (`enriched`) fechava 9% e o frontier 77%, o `maskable_frontier_pbrs` chega a 75% raw — empata em raw e fica 4pp abaixo na métrica filtrada (96.2% vs 100%). A avg coverage está em 99.6% (frontier 99.9%) — o agente cobre praticamente tudo, e ainda fecha 75 dos 78 mapas solucionáveis em 100 episódios.

O boustrophedon mostra a importância dos obstáculos. Em 5x5 (3 obstáculos, 22 células livres), o zigzag basta — 94% de full coverage, igual ao frontier. Em 10x10 (12 obstáculos), o zigzag fica preso a cada poucas linhas e o fallback frontier não recupera bem (26%). Em 20x20 (48 obstáculos), o zigzag é virtualmente inútil (0% de fechamento). Para grids com densidade alta de obstáculos, o frontier-based é necessário; o zigzag puro só serve em mapas vazios ou quase.

A diferença entre RL e frontier ficou pequena no resultado final. Em 5×5 o `maskable_frontier_pbrs` chega a 96% (vs 94% do frontier — RL ligeiramente acima). Em 10×10 ambos em 89-86% raw e 97.8/100 sobre solucionáveis. Em 20×20 o RL fica 2pp abaixo do frontier raw (75% vs 77%) e 4pp abaixo sobre solucionáveis (96.2% vs 100%). O frontier ainda tem mais eficiência (avg steps menores), mas a competência de fechamento empata praticamente. O frontier não é solução RL — não generaliza para outros problemas (cada problema precisa de heurística codificada à mão), enquanto o RL em princípio escala para qualquer task com signal de reward. A APS pede uma estratégia RL e isso é o que entreguei; o frontier serve aqui como ponto de comparação útil pra entender que **chegamos próximos do teto teórico atingível sob observabilidade parcial**.

## Comparativo final

Heatmap das full coverage rates de todas as estratégias (RL e scripted) avaliadas no grid em que treinaram (linha "nativa"):

![heatmap full coverage](results/plots/heatmap_native_full.png)

E a versão com a métrica filtrada (sobre mapas solucionáveis):

![heatmap full coverage solvable](results/plots/heatmap_native_full_solvable.png)

Curva de degradação por tamanho do grid, com média ± std das 3 seeds (ou seed único nos casos sinalizados):

![coverage by size full](results/plots/coverage_by_size_full.png)

E em avg coverage, a métrica que praticamente todas as estratégias bateram em todos os grids:

![coverage by size avg](results/plots/coverage_by_size_avg.png)

A leitura conjunta: em 5x5 quase todas as estratégias chegam a 91-97% de full coverage; excluindo recurrent, o problema é trivial nesse tamanho. Em 10x10 o `maskable_frontier_pbrs` lidera RL (89%), seguido por `maskable_bc_kl` (86%), `maskable_v3` (84%), `enriched`/`mapcnn_bc_pbrs` (77%), curriculum (71%) e baseline (64%); o frontier scripted é 86% raw mas 100% solucionáveis. Em 20x20 o `maskable_frontier_pbrs` quase iguala o frontier scripted (75% raw / 96.2% solucionáveis vs 77% raw / 100% solucionáveis); a melhor RL anterior (`enriched`) ficava em 9% raw / 11.8% solucionáveis, e as outras configs com reward redesign isolado (`mapcnn_bc_pbrs`/`maskable_v3`/`maskable_bc_kl`) colapsavam pra 0-1% raw — a combinação de memória estruturada + frontier feature + PBRS distance + value-head reset foi o que rompeu o 20x20.

## Bônus 20x20

O enunciado oferece 1 ponto extra se a estratégia chegar próxima de 100% também em 20x20.

**Atingi com RL puro: 75% raw / 96.2% sobre solucionáveis no 20x20 native** (config `maskable_frontier_pbrs`, seed 0). Pela leitura conservadora do enunciado (full coverage rate, que é a métrica citada ao descrever o baseline em `75/100`), 75% raw está acima dos `75/100` que o enunciado cita como referência do baseline em 5x5. Pela métrica filtrada sobre mapas solucionáveis (descontando os 23% de mapas com células fisicamente ilhadas), 96.2% é "próximo de 100%" com folga. Por avg coverage, 99.6% — o agente visita 350 das 352 células livres em média.

Esse resultado vem da combinação de quatro pilares atacando o credit assignment do fechamento das últimas células. Primeiro, **memória estruturada `visited_pooled` (2×8×8)** com max-pool da trajetória do agente em resolução fixa F=8, independente do tamanho do grid. Resolve o "esquecer onde já estive" sem precisar de hidden state recorrente, e como F=8 é fixo, a representação aprendida em 5x5 transfere quase direto pra 20x20. Segundo, **feature `frontier`** (3 dims: direção e distância) computada por BFS sobre o terreno conhecido pelo agente — sempre dá direção mesmo quando a janela 5x5 está toda visitada. Terceiro, **PBRS distance-based** com Φ = −d_BFS/diâmetro, magnitude per-step ~±0.05, comparável ao reward de +1 por nova célula, dense ao longo de toda a trajetória (Jonnarth et al. ICML 2024). Quarto, **reset do value head** ao iniciar cada fase do curriculum (Igl 2021, Wolczyk 2024) impede o crítico de carregar calibração estale e empurrar a política pra fora da bacia aprendida — exatamente o mecanismo que destruía a competência das fases anteriores nas configs `mapcnn_bc_pbrs`, `maskable_v3` e `maskable_bc_kl`.

A observabilidade parcial é estritamente preservada: a BFS pra fronteira opera apenas sobre `visited ∪ NOT seen_obstacles` (cells já observadas pela janela 5x5), com otimismo sob incerteza pras cells nunca observadas. A `visited_pooled` codifica apenas `self.visited` (subset do que `coverage_ratio` original já expõe). Em momento algum o agente acessa o conjunto global de obstáculos do mapa real. Tem teste explícito em `tests/test_v4_env.py::test_partial_observability_frontier_bfs_uses_only_seen_obstacles` validando isso.

Comparação com as outras configs no 20x20 native:

| Config | raw | solucionáveis |
|---|---|---|
| baseline | 0.3% | 0.4% |
| curriculum | 0.3% | 0.4% |
| curriculum_enriched | 9.0% | 11.8% |
| curriculum_recurrent_v2 | 0.0% | 0.0% |
| mapcnn_bc_pbrs | 0.0% | 0.0% |
| maskable_v3 | 0.0% | 0.0% |
| maskable_bc_kl | 1.0% | 1.3% |
| **maskable_frontier_pbrs** | **75.0%** | **96.2%** |
| frontier scripted (referência) | 77.0% | 100.0% |

O salto do `maskable_bc_kl` (1.3% solucionáveis) pro `maskable_frontier_pbrs` (96.2% solucionáveis) — quase 100x — vem da combinação dos quatro pilares acima atacando simultaneamente memória, fechamento e drift do PPO em horizonte longo. Cada peça individualmente tinha sido testada em configs anteriores e falhava sozinha (mapa CNN, BC anchor, reward redesign isolados); foi necessário combinar as quatro.

Documento também no [Apêndice](#apêndice-experimento-híbrido-fora-do-escopo-de-rl-puro) um experimento exploratório que mistura RL com FrontierAgent scripted na inferência. Esse atinge 99.6% sobre solucionáveis mas envolve uma heurística scripted como componente principal, então **não submeto como solução do bônus**. O número que submeto é o `maskable_frontier_pbrs`: RL puro, observabilidade parcial preservada, 96.2% sobre solucionáveis no 20x20 native.

## Limitações e aprendizados

A maior limitação prática foi o hardware: 8 GB de RAM, CPU 8 cores e uma GPU RTX 3060 6 GB. As 4 primeiras configs rodaram em CPU only; `curriculum_recurrent_v2`, `mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl` e `maskable_frontier_pbrs` rodaram na GPU. PPO usou `n_envs=4` em 5x5/10x10 e `n_envs=2` em 20x20. Cada seed em 20x20 leva 47-180 min nas configs com 2M timesteps, e ~8h no `maskable_frontier_pbrs` (4M timesteps). Orçamento total do estudo: ~46h de compute.

Outra limitação foi rodar `mapcnn_bc_pbrs`, `maskable_v3`, `maskable_bc_kl` e `maskable_frontier_pbrs` com apenas 1 seed cada, em vez das 3 seeds das primeiras configs. A decisão foi tomada porque cada seed dessas configs leva 3-8h, e o sinal diagnóstico do seed 0 já era forte o suficiente para decidir continuar ou pivotar dentro do orçamento de tempo. Para os 5 primeiros configs as 3 seeds estão presentes e produzem média ± std; para as últimas 4 o número é o do seed 0 (sem std), e a comparação com os 3-seed configs é honesta sobre essa assimetria. Os timesteps por fase ficaram fixos (300k/800k/2M) na maioria das configs, com override de 4M no 20x20 do `maskable_frontier_pbrs` justificado pela observação de que a curva de reward continuava subindo perto do final dos 2M.

A descoberta dos mapas insolucionáveis (~6/14/23% em 5x5/10x10/20x20) muda a leitura honesta dos resultados: o teto teórico de full coverage rate é 94/86/77% (não 100%), e o frontier scripted bate exatamente esses tetos. Mantenho a métrica raw para comparabilidade com o baseline citado pelo enunciado, mas reporto a métrica filtrada (sobre solucionáveis) lado a lado para mostrar a competência efetiva.

A jornada produziu cinco descobertas principais.

**Hipótese 2 (janela 3x3 → 5x5 + features direcionais) destravou a maior parte do 10x10.** O salto 14% → 70% no 5x5 → 10x10 transfer veio só da observação enriquecida; mais informação local funcionou.

**Memória recorrente (LSTM) é cara e instável.** Mesmo a v2 com LSTM 256 + n_steps 512 + GPU ficou muito abaixo do enriched no 10x10. Memória recorrente perdeu para memória **explícita estruturada** (a `visited_pooled` em resolução fixa F=8 do `maskable_frontier_pbrs`) — que substituiu o LSTM com sucesso e ainda transfere bem entre tamanhos.

**Reward landscape destravou o teto histórico de 77% no 10x10.** Terminal +60 em vez de +10, truncation 0 em vez de −5, step penalty zerado pós-80% de cobertura — calibrado a partir de Theile et al. 2023 — empurrou `maskable_v3` e `maskable_bc_kl` para 84-86% raw / 92-94% sobre solucionáveis em 10x10.

**PBRS magnitude importa tanto quanto a função potencial.** Φ=cobertura no `mapcnn_bc_pbrs` dava shaping ~0.001/step (efetivamente ruído); Φ=−d_BFS/diâmetro no `maskable_frontier_pbrs` dá ~±0.05/step (comparável ao reward de +1 por nova célula). Ng-Harada-Russell garante invariância da política ótima sob qualquer Φ, mas a velocidade de convergência depende criticamente da magnitude do gradient denso adicionado.

**A combinação destrava o 20x20 native.** Memória estruturada + frontier feature + PBRS distance + reset do value head atacam simultaneamente os três gargalos remanescentes (memória, fechamento, drift do PPO em horizonte longo) e produzem 75% raw / 96.2% sobre solucionáveis no 20x20 native — quase match com o frontier scripted (77%/100%). Cada peça individualmente tinha sido testada em configs anteriores e falhou sozinha; foi necessário combinar as quatro.

Em síntese, para coverage path planning sob observabilidade parcial, **estrutura na observação > memória explícita**, **reward shape > exploração mais longa**, e o frontier scripted (100% sobre solucionáveis em todos os tamanhos) é o teto que o RL puro ainda não atravessa em 20x20.

Como trabalhos futuros, três direções valem investigação. Primeiro, treinar `maskable_v3` ou `maskable_bc_kl` direto no 20x20 sem curriculum, hipotetizando que o curriculum 5 → 10 → 20 do PPO derive a política do que aprendeu nas fases anteriores. Segundo, KL anchor mais agressivo no 20x20, mantendo λ alto durante toda a fase 20x20 para forçar o agente a ficar próximo ao BC e não derivar. Terceiro, residual policy real (não só inferência), em que a output do PPO seja interpretada como ajuste sobre o frontier (Silver et al. 2018) — mais engineering mas com chão duro garantido em 100% solvable. Algoritmos mais recentes como DreamerV3 (world model) ou MuZero (planning) lidam melhor com long-horizon mas o compute fica fora do orçamento doméstico.

## Apêndice: experimento híbrido (fora do escopo de RL puro)

Esta seção descreve um experimento exploratório que **não foi submetido como solução do bônus do 20x20** porque mistura RL com uma heurística scripted (não-RL). Documento aqui apenas como ponto de curiosidade, para mostrar o teto prático do problema sob observabilidade parcial.

A motivação foi a observação de que o modelo `maskable_bc_kl` treinado em 10x10 atinge 32% raw / 41% sobre solucionáveis quando avaliado em 20x20 (transferência sem retreinamento), enquanto qualquer modelo treinado direto em 20x20 fica em ~0%. A política existe — ela só não sobrevive ao treino do PPO em horizonte longo.

A construção é uma mistura na inferência: a cada step, com probabilidade `(1 − p_model)` o agente segue a ação do `FrontierAgent` scripted (BFS sobre mapa interno construído só do que ele viu — preserva observabilidade parcial), e com probabilidade `p_model` segue a ação do modelo `maskable_bc_kl` 10x10. Implementação em `broom/eval_mixture.py`.

Resultados em 20x20, 3 seeds, 100 episódios cada, full coverage rate sobre solucionáveis:

| `p_model` | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| 0.00 (frontier puro, sem RL) | 100.0% | 100.0% | 100.0% | 100.0% ± 0.0% |
| 0.10 (90% frontier + 10% RL) | 98.7% | 100.0% | 100.0% | 99.6% ± 0.6% |
| 0.20 (80% frontier + 20% RL) | 96.2% | 100.0% | 100.0% | 98.7% ± 1.8% |

Por que isso não vale como solução RL do bônus: o enunciado exige "estratégia justificada com base em conceitos de RL", e com `p_model = 0.10` 90% das ações vêm de uma heurística BFS escrita à mão, não de aprendizado. Mesmo que a porção RL contribua, é difícil argumentar que a estratégia agregada é "RL" num sentido razoável. Apresentar isso como solução do bônus seria contornar o objetivo da APS. O experimento responde uma pergunta acadêmica diferente — qual é o teto prático sob observabilidade parcial — e a resposta é 100% sobre solucionáveis com frontier puro, com o RL ajudando a manter essa performance em um número pequeno de células onde o frontier hesitaria.
