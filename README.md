# APS07 — Generalização do Agente em Coverage Path Planning

Fork técnico de [`fbarth/gym_custom_env`](https://github.com/fbarth/gym_custom_env) feito para a Atividade Prática Supervisionada 07 da disciplina de Reinforcement Learning do Insper. Enunciado em https://insper.github.io/rl/classes/23_custom_env_agent/.

A APS pede uma estratégia que faça um agente PPO treinado no problema de Coverage Path Planning (CPP) generalizar entre tamanhos de grid (5x5, 10x10 e, como bônus, 20x20) preservando a observabilidade parcial. O baseline do enunciado treina em 5x5 e degrada quando avaliado em grids maiores. Aqui investigamos quatro configurações para atacar essa degradação.

O repositório foi reduzido aos arquivos relacionados ao Coverage Path Planning. Os exemplos do upstream para outros ambientes (grid world básico, 3D, com obstáculos, com renderização) foram removidos para deixar a leitura focada na APS. O histórico do upstream segue acessível pelo `git log` e via remote `upstream`.

## Ambiente

`GridWorldCPPEnv` é o ambiente de Coverage Path Planning herdado do upstream. O agente nasce numa célula aleatória de um grid quadrado com obstáculos fixos por episódio, e precisa visitar todas as células livres sem revisitar.

| Propriedade | Valor |
|---|---|
| Estado | `agent` (x, y normalizados, ratio de cobertura) e `neighbors` 3x3 ao redor do agente |
| Ações | 0 = direita, 1 = cima, 2 = esquerda, 3 = baixo |
| Reward | +1 por célula nova, −0.3 por revisita, −0.5 por bater em parede, −0.1 por step, +10 ao cobrir tudo, −5 ao truncar |
| Término | todas as células livres visitadas, ou `max_steps` excedido |
| Observação | parcial: o agente vê só a vizinhança 3x3 (codificada como 0 = livre, 1 = parede ou obstáculo, 2 = visitada) |

A observabilidade parcial é o ponto da APS. O agente nunca tem acesso ao mapa completo, então a política precisa lidar com a incerteza sobre o que existe além da janela.

| Tamanho | Obstáculos | `max_steps` |
|---|---|---|
| 5x5 | 3 | 200 |
| 10x10 | 12 | 500 |
| 20x20 | 48 | 1000 |

## O Problema da Generalização

A política aprendida em 5x5 não transfere para 10x10. O motivo é uma combinação de três fatores que descobrimos empiricamente:

1. **Features dependem da escala.** A posição é normalizada por `size` (`x/5`, `y/5`), então uma posição relativa de 0.5 em 5x5 corresponde a uma célula no centro, mas em 10x10 corresponde a outra coordenada absoluta. A rede aprende mapeamentos ligados ao 5x5.

2. **A janela 3x3 cobre uma fatia cada vez menor.** Em 5x5 a vizinhança 3x3 representa 36% das células do mapa. Em 10x10, só 9%. Em 20x20, 2.25%. Quanto maior o grid, menos contexto local o agente tem para decidir.

3. **Sem memória, o agente esquece.** A política markoviana só vê a janela atual, não o histórico de células visitadas fora dela. Em mapas pequenos, a janela é grande o suficiente para que o agente sempre veja parte do que já visitou; em mapas grandes, ele entra em regiões novas sem saber onde já passou.

## Estratégias Investigadas

Quatro configurações comparadas. Todas usam PPO; as diferenças estão em como atacam um ou mais dos três fatores acima.

| Config | Estratégia | Hipótese atacada |
|---|---|---|
| `baseline` | PPO com `MultiInputPolicy`, sem curriculum (treina do zero em cada tamanho) | nenhuma (reproduz o problema) |
| `curriculum` | PPO + curriculum learning: 5x5 → 10x10 → 20x20, transferindo pesos | escala de features |
| `curriculum_enriched` | curriculum + observação ampliada (vizinhança 5x5 + direção e distância à célula não-visitada mais próxima) | janela pequena |
| `curriculum_recurrent` | curriculum com RecurrentPPO (LSTM 64 unidades) | falta de memória |

Cada configuração roda com 3 seeds (0, 1, 2) e é avaliada em todos os três tamanhos.

## Como Executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m broom.run_experiments --configs baseline,curriculum,curriculum_enriched,curriculum_recurrent
```

O `run_experiments.py` é resumível: pula combinações cujo modelo já existe em `results/models/`. Para rodar uma config isolada:

```bash
python -m broom.run_experiments --configs baseline
```

Para treinar sem rodar inferência:

```bash
python -m broom.run_experiments --configs baseline --skip-inference
```

Os testes ficam em `tests/`:

```bash
pytest tests/ -q
```

## Configurações

Hiperparâmetros principais (mantidos consistentes para isolar a estratégia):

| Parâmetro | Valor |
|---|---|
| Algoritmo (configs 1–3) | PPO + `MultiInputPolicy` |
| Algoritmo (config 4) | RecurrentPPO + `MultiInputLstmPolicy` |
| `ent_coef` | 0.05 (igual upstream) |
| `device` | cpu |
| `n_envs` (PPO, 5x5/10x10) | 4 |
| `n_envs` (PPO, 20x20) | 2 |
| `n_envs` (RecurrentPPO, todos) | 2 |
| Timesteps por fase | 5x5: 300k, 10x10: 800k, 20x20: 2M |
| LSTM (config 4) | 64 unidades, 1 camada |
| Seeds | 0, 1, 2 |
| Episódios de inferência | 100 (política estocástica, `deterministic=False`) |

## Curvas de Aprendizado

Todas as curvas usam média e desvio padrão sobre 3 seeds, suavizadas com janela móvel de 20 episódios.

### Baseline (PPO sem curriculum)

![baseline](results/plots/learning_curve_baseline.png)

O agente converge nos três tamanhos. 5x5 sai de −60 e estabiliza próximo de 0; 10x10 sai de −140 e chega a −10; 20x20 sai de −200 e atinge ~0.

### Curriculum (PPO com warm start entre fases)

![curriculum](results/plots/learning_curve_curriculum.png)

A fase 5x5 é idêntica ao baseline (treinada do zero). Nas fases 10x10 e 20x20, o eixo X reinicia em zero porque cada fase é um treino separado com `model.learn(reset_num_timesteps=False)`. Os pesos vêm carregados da fase anterior, então a curva começa em reward mais alto que o baseline equivalente.

### Curriculum + observação enriquecida

![enriched](results/plots/learning_curve_curriculum_enriched.png)

Comportamento parecido com curriculum, mas com a observação 5x5 + features de direção/distância para a célula não-visitada mais próxima na janela.

## Resultados de Inferência

100 episódios por modelo, política estocástica. Cada modelo treinado num tamanho é avaliado nos três. Todos os números são médias sobre 3 seeds. A diagonal é a performance "nativa" (mesmo tamanho); os off-diagonais medem generalização.

### Baseline

#### Full coverage rate

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **92.7%** | 14.0% | 0.0% |
| 10x10 | 89.0% | **64.3%** | 0.3% |
| 20x20 | 87.3% | 47.7% | **0.3%** |

#### Avg coverage

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **99.1%** | 95.9% | 79.4% |
| 10x10 | 98.7% | **98.2%** | 95.4% |
| 20x20 | 98.4% | 97.8% | **94.1%** |

### Curriculum

#### Full coverage rate

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **92.7%** | 14.0% | 0.0% |
| 10x10 | 90.7% | **71.3%** | 2.0% |
| 20x20 | 89.0% | 64.7% | **0.3%** |

#### Avg coverage

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **99.1%** | 95.9% | 79.4% |
| 10x10 | 98.9% | **98.9%** | 96.6% |
| 20x20 | 98.7% | 98.3% | **96.6%** |

A linha 5x5 é idêntica ao baseline porque a primeira fase do curriculum não tem warm-start (o modelo é criado do zero). O ganho aparece a partir do 10x10 e é mais visível em quem o modelo final do 20x20 consegue fazer no 10x10 (64.7% vs 47.7% do baseline).

### Curriculum + observação enriquecida

#### Full coverage rate

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **91.3%** | 69.7% | 0.7% |
| 10x10 | 92.7% | **77.3%** | 4.7% |
| 20x20 | 91.0% | 73.0% | **9.0%** |

#### Avg coverage

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **98.6%** | 98.7% | 93.5% |
| 10x10 | 98.9% | **98.6%** | 96.7% |
| 20x20 | 98.8% | 98.8% | **97.3%** |

A célula mais surpreendente é o 5x5/10x10: 69.7% (vs 14.0% do baseline e do curriculum). A janela 5x5 + a feature `direction_to_nearest_unvisited` fazem o modelo treinado só em 5x5 generalizar quase tão bem em 10x10 quanto em 5x5. Isso é resultado de **estrutura na observação**, não de mais treino.

## Análise

Comparando as três configurações nas células-chave (linha de comparação direta com o enunciado):

| Cell | Baseline | Curriculum | Enriched |
|---|---|---|---|
| 5x5 trained, 10x10 eval | 14.0% | 14.0% | **69.7%** |
| 10x10 trained, 10x10 eval | 64.3% | 71.3% | **77.3%** |
| 20x20 trained, 10x10 eval | 47.7% | 64.7% | **73.0%** |
| 20x20 trained, 20x20 eval | 0.3% | 0.3% | **9.0%** |
| 5x5 trained, 5x5 eval | 92.7% | 92.7% | 91.3% |

Cada hipótese da seção "O Problema da Generalização" se mapeia num resultado:

**Hipótese 1: features dependem da escala.** O curriculum endereça isso ao carregar pesos de 5x5 → 10x10 → 20x20. Ganho real mas modesto: +7.0pp no 10x10 native, +17.0pp no eval do 10x10 a partir do modelo final do 20x20. O 5x5 native não muda porque a primeira fase do curriculum equivale ao baseline. A escala de features parece ser parte do problema mas não a maior parte.

**Hipótese 2: janela 3x3 fica pequena em grids grandes + falta de pista direcional.** É aqui que o enriched faz diferença. O 5x5/10x10 vai de 14% para 70% sem precisar de curriculum — é estrutura. A janela 5x5 mostra mais células, e `direction_to_nearest_unvisited` resolve o "para onde devo ir" que a janela 3x3 sozinha não responde. Esta era a hipótese certa para a generalização entre 5x5 e 10x10.

**Hipótese 3: agente esquece células visitadas fora da janela.** Será testada com a config `curriculum_recurrent`, que substitui o MLP por um LSTM. O treino dela está em curso e a seção de resultados é atualizada conforme os experimentos terminam.

### O bônus 20x20

O 20x20 native continua difícil mesmo com enriched (9.0%). O salto de 0.3% para 9.0% mostra que a estrutura ajuda, mas não é suficiente para fechar a cobertura nos 1000 passos disponíveis. Discussão completa na seção [Bônus 20x20](#bônus-20x20).

### Trade-off de avg coverage vs full coverage rate

Avg coverage fica em 94-99% em todas as configurações e em todos os tamanhos. O agente encontra a maioria das células. O que diferencia as estratégias é a capacidade de **fechar** a cobertura, ou seja, encontrar as últimas 1-5 células antes do `max_steps`. Esse é um problema de eficiência, não de exploração.

## Bônus 20x20

Resultado em construção.

## Limitações e trabalhos futuros

Em construção.
