# Previsão de Carga do SIN: multi-horizonte, com modelo vivo

**📊 [Ver a análise completa e interativa](https://arnaldochebl.github.io/ONS/)** (GitHub Pages)

Previsão horária da carga elétrica do Sistema Interligado Nacional (dados abertos do
ONS) para horizontes de **1h a 24h à frente**, com a disciplina completa de um projeto
de forecasting sério: pré-processamento auditável, features sem vazamento, validação
cruzada temporal com tunagem, um **canário de ruído** para poda de features, baselines
ingênuos que precisam ser batidos, backtest em holdout intocado e um **modelo vivo**
que roda toda semana no GitHub Actions registrando previsões antes de o realizado ser
publicado.

## Por que este desenho

- **Multi-horizonte direto**: um único LightGBM global recebe `horizonte` como feature
  e prevê de 1h a 24h. As features autorregressivas usam somente informação disponível
  na origem; as únicas features do instante-alvo são de calendário.
- **Validação temporal, nunca aleatória**: folds trimestrais de 2025 com treino
  expansivo. Embaralhar séries temporais é vazamento.
- **Canário de ruído**: uma feature `N(0,1)` entra na tunagem; qualquer feature com
  ganho menor ou igual ao do ruído é cortada e o modelo enxuto é re-validado. Se uma
  variável informa menos que ruído puro, ela não merece o posto.
- **Baselines com dignidade**: persistência e sazonal-semanal (mesma hora, 7 dias
  atrás). O sazonal é forte em carga elétrica; bater ele é o que justifica o modelo.
- **Quantis (P10/P50/P90)**: operação decide com intervalo, não com ponto. A cobertura
  empírica do intervalo é medida no holdout.
- **Holdout intocado**: origens de 2026 nunca participam de nenhuma escolha.
- **Modelo vivo a custo zero**: um workflow agendado (segundas, 12:00 UTC) baixa o dado
  novo, registra a previsão das próximas 24h em `docs/live/` e reconcilia o track
  record público. O carimbo do commit prova que a previsão precede o realizado (a
  publicação do ONS tem ~1 semana de defasagem).

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate        # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python src/data.py           # baixa dados do ONS (2021+) e monta a série limpa
python src/train.py          # CV temporal + tunagem + canário + modelos finais
python src/evaluate.py       # backtest no holdout e artefatos da página
python src/build_charts.py --splice   # regenera os gráficos de docs/index.html
python src/score_live.py     # um ciclo do modelo vivo (o Actions faz isso sozinho)
```

## Estrutura

| Caminho | O que é |
|---|---|
| `src/data.py` | Aquisição (S3 público do ONS) e limpeza com relatório de qualidade |
| `src/features.py` | Features multi-horizonte anti-vazamento + canário de ruído |
| `src/train.py` | CV temporal, busca aleatória, filtro por ruído, quantílicos finais |
| `src/evaluate.py` | Backtest no holdout vs baselines, cobertura, artefatos |
| `src/score_live.py` | Ciclo semanal do modelo vivo (previsão + reconciliação) |
| `models/` | Boosters finais + config (commitados: o Actions precisa deles) |
| `docs/` | Página estática (Pages) + `docs/live/` com o track record público |

## Decisões de dados (resumo do relatório de qualidade)

Janela 2021+ (regime pós-COVID). Duplicatas, valores impossíveis e drift de esquema
entre anos (carga como texto em arquivos antigos) são tratados com coerção explícita e
contabilizados. Lacunas curtas (≤3h) são interpoladas; longas ficam como estão e são
reportadas. Outliers são apenas sinalizados por desvio robusto contra o perfil
(dia-da-semana, hora): um apagão é realidade operacional, não erro de medição.

Fonte dos dados: [Portal de Dados Abertos do ONS](https://dados.ons.org.br), dataset
"Curva de Carga Horária". Clima não entra nesta versão: em operação, exigiria previsão
meteorológica no momento do score; fica documentado como evolução.

Stack: Python · pandas · LightGBM · scikit-learn · GitHub Actions (cron) · GitHub Pages.
