# Co-movimento de ativos por Wavelets (B3 / yfinance)

Análise de **co-movimento entre ativos em múltiplas escalas de tempo** usando apenas preços de fechamento (log-retornos) e transformadas wavelet em Python.

O script baixa as cotações via `yfinance`, trata os dados e responde perguntas como:

- Em quais horizontes (semana, mês, trimestre...) dois ativos se movem juntos?
- Quem **lidera** quem, e em quantos pregões?
- Quanto desse co-movimento é só efeito do **Ibovespa** ou do **dólar**?
- Quais ativos formam grupos em cada horizonte, e quais diversificam melhor?

> Projeto de pesquisa/estudo. **Não é recomendação de investimento.**

## Funcionalidades

| Módulo | O que faz |
|---|---|
| **Dados** | Fechamento ajustado (proventos/splits), cache diário, retry, calendário comum de pregões, tratamento de buracos, remoção de ativos ilíquidos, contenção de outliers (MAD) e padronização |
| **Correlação por escala (MODWT)** | Decompõe cada ativo em escalas (2-4d, 4-8d, ~semana, ~mês, ~trimestre...) e correlaciona escala a escala, com IC de Fisher e exclusão dos coeficientes de borda |
| **Coerência wavelet cruzada (CWT Morlet)** | Mapa tempo × período com significância Monte Carlo (nulo AR(1)), cone de influência (COI) e setas de fase |
| **Lead-lag** | Defasagem média (em pregões) por faixa de período, a partir da fase da coerência |
| **Coerência parcial e múltipla** | Isola o efeito de Ibovespa e/ou dólar sobre o par; mede quanto da variância de um ativo é explicada por B + fatores |
| **Muitos ativos** | Coerência média por faixa (curto 2-8d, médio 8-64d, longo 64d+) para todos os pares, redes de co-movimento, comunidades, clusters hierárquicos e score de diversificação |

A MODWT e a coerência wavelet (incluindo a parcial/múltipla) são implementadas diretamente em NumPy/SciPy/PyWavelets, sem depender de bibliotecas de coerência externas.

## Instalação

Requer Python 3.9+ (recomendado 3.11 a 3.13; versões muito recentes podem ainda não ter pacotes científicos prontos).

```bash
git clone <URL-DO-SEU-REPOSITORIO>
cd <pasta-do-repositorio>
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
scipy
matplotlib
PyWavelets
yfinance
networkx
statsmodels   # opcional (teste ADF de estacionariedade)
```

## Uso

Execução padrão (8 ações da B3, par PETR4 × VALE3, controlando Ibovespa e dólar):

```bash
python comovimento_wavelet.py
```

Personalizada (**em uma única linha** no PowerShell; no bash pode quebrar com `\`):

```bash
python comovimento_wavelet.py --tickers PETR4 VALE3 ITUB4 BBDC4 ABEV3 WEGE3 BBAS3 SUZB3 --pair PETR4 VALE3 --start 2016-01-01 --n-surrogates 300
```

Teste offline com dados simulados (não usa internet):

```bash
python comovimento_wavelet.py --synthetic --n-surrogates 40
```

Tickers da B3 podem ser passados sem sufixo (`PETR4`); o `.SA` é adicionado automaticamente. Os fatores de controle são `^BVSP` (Ibovespa) e `BRL=X` (dólar).

### Opções

| Opção | Padrão | Descrição |
|---|---|---|
| `--tickers` | 8 ações da B3 | Ativos da análise |
| `--pair A B` | `PETR4 VALE3` | Par analisado em detalhe (mapa de coerência, parcial, lead-lag) |
| `--no-factors` | — | Não baixa/usa Ibovespa e dólar |
| `--start`, `--end` | `2016-01-01`, hoje | Período dos dados |
| `--outdir` | `output_wavelet` | Pasta de saída |
| `--no-cache` | — | Ignora o cache de preços |
| `--wavelet` | `sym4` | Wavelet da MODWT (qualquer ortogonal do PyWavelets) |
| `--levels` | `7` | Nº máximo de níveis da MODWT (reduzido automaticamente se a série for curta) |
| `--max-period` | `N/4` | Maior período da CWT (pregões) |
| `--n-surrogates` | `200` | Simulações Monte Carlo para significância (0 desliga) |
| `--edge-quantile` | `0.70` | Redes: mantém pares acima deste quantil de coerência |
| `--clusters` | `3` | Nº de clusters hierárquicos por faixa |
| `--synthetic` | — | Usa dados simulados |
| `--show` | — | Abre os gráficos além de salvá-los |
| `--seed` | `42` | Semente aleatória |

## Saídas

Tudo é gravado em `--outdir`.

**Figuras**

| Arquivo | Conteúdo |
|---|---|
| `fig0_modwt_correlacao_par.png` | Correlação por escala do par, com IC de 95% |
| `fig1_coerencia_mapa.png` | Mapa tempo × período: coerência, contorno de significância, COI, setas de fase |
| `fig2_coerencia_parcial_mapas.png` | Coerência bruta vs. parcial (controlando Ibovespa / dólar / ambos) |
| `fig3_coerencia_media_por_escala.png` | Coerência média por período: bruta, parciais, múltiplas e limiar 95% |
| `fig4_coerencia_por_faixa_heatmap.png` | Matriz de coerência média por faixa (curto/médio/longo) |
| `fig5_redes_comovimento.png` | Redes de co-movimento por faixa, coloridas por comunidade |
| `fig6_modwt_heatmaps_por_escala.png` | Um heatmap de correlação por escala MODWT |

**Tabelas (CSV)**: `log_retornos.csv`, `modwt_correlacao_par.csv`, `coerencia_media_por_periodo.csv`, `lead_lag_por_faixa.csv`, `matriz_coerencia_<faixa>.csv` e `clusters_e_diversificacao_por_faixa.csv`.

## Como interpretar

- **Períodos** estão em pregões: ~5 = semana, ~21 = mês, ~63 = trimestre, ~252 = ano.
- **Coerência²** vai de 0 a 1 (análoga a um R² local no tempo-frequência). Compare sempre com o limiar de significância e com a coerência média sob o nulo (impressa no terminal).
- **Setas de fase** (eixo vertical = período crescente):
  - `→` em fase; `←` anti-fase;
  - `↑` o 1º ativo do par **lidera** o 2º; `↓` o 2º lidera o 1º.
- **Parcial vs. bruta:** se a coerência cai muito ao controlar o Ibovespa, o co-movimento é majoritariamente efeito de mercado; se permanece, há relação específica entre o par.
- **Score de diversificação:** quanto menor a coerência média de um ativo com os demais numa faixa, melhor ele diversifica naquele horizonte.

## Cuidados e limitações

- **Análise exploratória em amostra.** A CWT/MODWT sobre a série inteira usa informação futura; não use diretamente como sinal de trading. Para isso, recalcule em janelas deslizantes (walk-forward) e faça backtest com custos.
- **Efeitos de borda:** resultados dentro do COI (hachura) e os primeiros coeficientes da MODWT são descartados.
- **Não-sincronia de fechamento:** o dólar (`BRL=X`) fecha em horário diferente da B3, gerando defasagem artificial de ~1 pregão no lead-lag com ele.
- **Comparações múltiplas:** o limiar de 95% é pontual; com muitos pares/células há falsos positivos. As redes são descritivas, não inferenciais.
- **Escalas longas** (> 1 ano) exigem séries longas (5+ anos).
- A significância assume nulo de ruído AR(1) independente, uma simplificação.
- O script foi validado com dados sintéticos (defasagem conhecida de 4 pregões recuperada corretamente). Resultados reais dependem dos dados do Yahoo Finance, que podem conter falhas.

## Estrutura

```
.
├── comovimento_wavelet.py   # script principal
├── requirements.txt
└── README.md
```

## Referências

- Torrence, C. & Compo, G. P. (1998). *A Practical Guide to Wavelet Analysis*. Bulletin of the AMS.
- Torrence, C. & Webster, P. J. (1999). *Interdecadal changes in the ENSO-monsoon system*. Journal of Climate.
- Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). *Application of the cross wavelet transform and wavelet coherence to geophysical time series*. Nonlinear Processes in Geophysics.
- Percival, D. B. & Walden, A. T. (2000). *Wavelet Methods for Time Series Analysis*. Cambridge University Press.
- Gençay, R., Selçuk, F. & Whitcher, B. (2001). *An Introduction to Wavelets and Other Filtering Methods in Finance and Economics*. Academic Press.

## Licença

Defina a licença do seu repositório (por exemplo, MIT) e adicione o arquivo `LICENSE`.
