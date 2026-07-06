# -*- coding: utf-8 -*-
"""
Engenharia de features para previsao multi-horizonte (1h a 24h) da carga do SIN.

Desenho anti-vazamento: cada linha do dataset supervisionado e um par
(origem t, horizonte h). TODAS as features derivadas da serie usam apenas
informacao disponivel ate a origem t (shifts >= 0). As unicas features do
instante-alvo (t+h) sao de CALENDARIO, que e conhecido com antecedencia
infinita (hora, dia da semana, feriado...).

Grupos de features:
  - Autorregressivas na origem: lags de 1-6h, 24h, 48h, 168h e 336h.
  - Janelas na origem: media/min/max de 24h, media/desvio de 168h, variacoes.
  - Referencias "mesma hora": carga na mesma hora de ontem e de 7 dias atras
    relativas ao ALVO (shifts 24-h e 168-h, sempre >= 0 para h <= 24).
  - Calendario do alvo: hora, dia da semana, mes, codificacao ciclica,
    fim de semana, feriado nacional, vespera e pos-feriado.
  - Nome do feriado com controle de cardinalidade: os mais frequentes viram
    categorias proprias; o resto vira "outros" (feriados moveis raros nao
    ganham categoria propria para nao virar ruido de alta cardinalidade).
  - Horizonte h como feature (um unico modelo global para os 24 horizontes).
  - "ruido_aleatorio": N(0,1) independente do alvo. E o canario da mina:
    qualquer feature com importancia menor ou igual a do ruido nao esta
    contribuindo com sinal real e e cortada.
"""
from pathlib import Path

import holidays as pyholidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONTES = list(range(1, 25))
TOP_FERIADOS = 8
SEED = 42

LAGS = [1, 2, 3, 4, 5, 6, 24, 48, 168, 336]


def _tabela_feriados(anos) -> dict:
    br = pyholidays.Brazil(years=list(anos))
    return {pd.Timestamp(d): nome for d, nome in br.items()}


def _calendario_alvo(idx_alvo: pd.DatetimeIndex,
                     feriados: dict, nomes_top: list) -> pd.DataFrame:
    datas = idx_alvo.normalize()
    nome = pd.Series([feriados.get(d, "") for d in datas], index=idx_alvo)
    e_feriado = (nome != "").astype(np.int8)
    nome_cat = nome.where(nome.isin(nomes_top) | (nome == ""), "outros")
    nome_cat = nome_cat.replace("", "nao_feriado")

    ontem = datas - pd.Timedelta(days=1)
    amanha = datas + pd.Timedelta(days=1)
    pos_feriado = pd.Series([1 if o in feriados else 0 for o in ontem],
                            index=idx_alvo, dtype=np.int8)
    vespera = pd.Series([1 if a in feriados else 0 for a in amanha],
                        index=idx_alvo, dtype=np.int8)

    hora = idx_alvo.hour.astype(np.int8)
    doy = idx_alvo.dayofyear.astype(np.int16)
    cal = pd.DataFrame({
        "hora_alvo": hora,
        "dow_alvo": idx_alvo.dayofweek.astype(np.int8),
        "mes_alvo": idx_alvo.month.astype(np.int8),
        "fim_de_semana": (idx_alvo.dayofweek >= 5).astype(np.int8),
        "hora_sin": np.sin(2 * np.pi * hora / 24).astype(np.float32),
        "hora_cos": np.cos(2 * np.pi * hora / 24).astype(np.float32),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
        "e_feriado": e_feriado.values,
        "vespera_feriado": vespera.values,
        "pos_feriado": pos_feriado.values,
        "nome_feriado": pd.Categorical(nome_cat.values),
    }, index=idx_alvo)
    return cal


def nomes_feriados_top(serie: pd.Series, k: int = TOP_FERIADOS) -> list:
    """Feriados mais frequentes no periodo da serie (controle de cardinalidade)."""
    anos = range(serie.index.min().year - 1, serie.index.max().year + 2)
    fer = _tabela_feriados(anos)
    datas = pd.Series(list(fer.keys()))
    dentro = datas[(datas >= serie.index.min().normalize())
                   & (datas <= serie.index.max().normalize())]
    nomes = pd.Series([fer[d] for d in dentro])
    return nomes.value_counts().head(k).index.tolist()


def features_origem(serie: pd.Series) -> pd.DataFrame:
    """Features conhecidas na origem t (so passado)."""
    f = pd.DataFrame(index=serie.index)
    for lag in LAGS:
        f[f"lag_{lag}h"] = serie.shift(lag).astype(np.float32)
    f["carga_origem"] = serie.astype(np.float32)          # lag 0
    f["media_24h"] = serie.rolling(24).mean().astype(np.float32)
    f["min_24h"] = serie.rolling(24).min().astype(np.float32)
    f["max_24h"] = serie.rolling(24).max().astype(np.float32)
    f["desvio_24h"] = serie.rolling(24).std().astype(np.float32)
    f["media_168h"] = serie.rolling(168).mean().astype(np.float32)
    f["var_1h"] = (serie - serie.shift(1)).astype(np.float32)
    f["var_24h"] = (serie - serie.shift(24)).astype(np.float32)
    f["var_168h"] = (serie - serie.shift(168)).astype(np.float32)
    return f


def monta_supervisionado(df_serie: pd.DataFrame,
                         horizontes=HORIZONTES,
                         incluir_ruido: bool = True,
                         nomes_top: list | None = None,
                         seed: int = SEED) -> pd.DataFrame:
    """Empilha pares (origem, horizonte) com alvo y = carga(origem+h)."""
    serie = df_serie["carga_mw"]
    if nomes_top is None:
        nomes_top = nomes_feriados_top(serie)
    anos = range(serie.index.min().year - 1, serie.index.max().year + 2)
    feriados = _tabela_feriados(anos)

    base = features_origem(serie)
    blocos = []
    for h in horizontes:
        b = base.copy()
        b["horizonte"] = np.int8(h)
        b["y"] = serie.shift(-h).astype(np.float32)
        # referencias na mesma hora do alvo (sempre passado para h <= 24)
        b["ref_ontem_mesma_hora"] = serie.shift(24 - h).astype(np.float32)
        b["ref_semana_mesma_hora"] = serie.shift(168 - h).astype(np.float32)
        idx_alvo = serie.index + pd.Timedelta(hours=h)
        cal = _calendario_alvo(idx_alvo, feriados, nomes_top)
        cal.index = serie.index
        blocos.append(pd.concat([b, cal], axis=1))

    sup = pd.concat(blocos)
    sup.index.name = "origem"
    sup = sup.dropna(subset=["y", f"lag_{max(LAGS)}h",
                             "ref_semana_mesma_hora"])
    # nome_feriado precisa de categorias unificadas pos-concat
    sup["nome_feriado"] = pd.Categorical(sup["nome_feriado"])

    if incluir_ruido:
        rng = np.random.default_rng(seed)
        sup["ruido_aleatorio"] = rng.normal(size=len(sup)).astype(np.float32)
    return sup


COL_ALVO = "y"
COLS_NAO_FEATURE = {COL_ALVO}


def colunas_features(sup: pd.DataFrame) -> list:
    return [c for c in sup.columns if c not in COLS_NAO_FEATURE]
