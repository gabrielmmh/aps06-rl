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

### Baseline (PPO sem curriculum)

![baseline](results/plots/learning_curve_baseline.png)

A curva de cada tamanho usa média e desvio padrão sobre as 3 seeds. O agente converge nos três tamanhos: 5x5 sai de −60 e estabiliza próximo de 0; 10x10 sai de −140 e chega a −10; 20x20 sai de −200 e atinge ~0. As três outras configurações entram nesta seção conforme os épicos seguintes ficam prontos.

## Resultados de Inferência

### Baseline

100 episódios por modelo, política estocástica. Cada modelo treinado num tamanho é avaliado nos três. Todos os números são médias sobre 3 seeds.

#### Full coverage rate (% de episódios que cobriram tudo)

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **92.7%** | 14.0% | 0.0% |
| 10x10 | 89.0% | **64.3%** | 0.3% |
| 20x20 | 87.3% | 47.7% | **0.3%** |

#### Avg coverage (média de cobertura ao final do episódio)

| Treinado em ↓ \ Avaliado em → | 5x5 | 10x10 | 20x20 |
|---|---|---|---|
| 5x5 | **99.1%** | 95.9% | 79.4% |
| 10x10 | 98.7% | **98.2%** | 95.4% |
| 20x20 | 98.4% | 97.8% | **94.1%** |

A diagonal mostra a performance "nativa" (treinado e avaliado no mesmo tamanho). Os off-diagonais mostram a generalização.

## Análise

A degradação prevista pelo enunciado aparece nítida:

* **5x5 → 10x10:** o agente treinado em 5x5 vai de 92.7% de full coverage para 14.0% quando avaliado em 10x10. A avg coverage cai bem menos (99.1% → 95.9%), o que indica que o agente continua explorando, mas não consegue fechar a cobertura nos cantos do mapa maior.

* **5x5 → 20x20:** zero episódios completos. O agente cobre em média 79.4% das células, mas nunca termina. A janela 3x3 vira muito pequena em proporção e a normalização da posição perde sentido.

* **Treinar direto no 20x20 não basta.** Mesmo o modelo treinado em 20x20 fecha a cobertura em apenas 0.3% dos episódios. Avg coverage é 94.1%, o que mostra que o PPO aprende uma política exploratória decente, mas a observabilidade parcial num grid grande impede o fechamento total. Isso é coerente com o enunciado, que menciona ~65% como referência para o baseline em 10x10.

* **Treinar em 10x10 ou 20x20 não recupera 5x5.** O agente treinado em 20x20 cai para 87.3% no 5x5 (vs 92.7% nativo). Sugere overfitting às características do grid maior.

A baseline confirma a hipótese global da APS: PPO com `MultiInputPolicy` numa janela 3x3 e posição normalizada não generaliza entre escalas. As três configurações seguintes atacam esses pontos.

## Bônus 20x20

Resultado em construção.

## Limitações e trabalhos futuros

Em construção.
