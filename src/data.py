# -*- coding: utf-8 -*-
"""
Aquisicao e limpeza da curva de carga horaria do SIN (dados abertos do ONS).

Fonte: https://dados.ons.org.br (dataset "curva-carga", parquet anual no S3
publico do ONS). A serie alvo e a carga total do SIN em MW medio horario,
obtida somando os 4 subsistemas (SE/CO, S, NE, N).

Filosofia de qualidade de dados (documentada tambem na pagina do projeto):
  - Duplicatas exatas de (instante, subsistema): mantem a primeira.
  - Valores fisicamente impossiveis (carga <= 0): viram NaN.
  - Reindexacao para grade horaria completa: lacunas ficam explicitas.
  - Lacunas curtas (ate 3h): interpolacao linear; longas: permanecem NaN e
    sao reportadas (nao inventamos dado).
  - Outliers: detectados por desvio robusto contra o perfil tipico do mesmo
    (dia-da-semana, hora), mas apenas SINALIZADOS, nunca removidos: um
    apagao e realidade operacional, nao erro de medicao. A distincao entre
    "outlier de dado" e "outlier de realidade" fica registrada no relatorio.

Uso:
    python src/data.py            # baixa/atualiza cache e gera processed
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
for p in (RAW, PROC, ART):
    p.mkdir(parents=True, exist_ok=True)

URL = ("https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
       "curva-carga-ho/CURVA_CARGA_{ano}.parquet")
ANO_INICIO = 2021          # regime pos-COVID; anos anteriores mudam o padrao
GAP_MAX_INTERP = 3         # horas: lacunas ate este tamanho sao interpoladas


def baixa_ano(ano: int, forca: bool = False) -> pd.DataFrame:
    """Baixa (ou le do cache) o parquet anual do ONS."""
    dest = RAW / f"CURVA_CARGA_{ano}.parquet"
    if dest.exists() and not forca:
        return pd.read_parquet(dest)
    df = pd.read_parquet(URL.format(ano=ano))
    df.to_parquet(dest, index=False)
    return df


def monta_serie(ano_fim: int | None = None,
                atualiza_corrente: bool = True) -> tuple[pd.Series, dict]:
    """Serie horaria da carga total do SIN + relatorio de qualidade."""
    ano_fim = ano_fim or date.today().year
    frames = []
    for ano in range(ANO_INICIO, ano_fim + 1):
        # o ano corrente (e o anterior, por seguranca de revisoes) e sempre
        # rebaixado; anos fechados ficam no cache
        forca = atualiza_corrente and ano >= ano_fim - 1
        frames.append(baixa_ano(ano, forca=forca))
    df = pd.concat(frames, ignore_index=True)

    rel = {"linhas_brutas": int(len(df))}

    # drift de esquema entre anos: em alguns arquivos a carga vem como texto
    # e o instante como string; coercao explicita, com falhas contabilizadas
    df["din_instante"] = pd.to_datetime(df["din_instante"], errors="coerce")
    antes_na = df["val_cargaenergiahomwmed"].isna().sum()
    df["val_cargaenergiahomwmed"] = pd.to_numeric(
        df["val_cargaenergiahomwmed"], errors="coerce")
    rel["falhas_coercao_numerica"] = int(
        df["val_cargaenergiahomwmed"].isna().sum() - antes_na)
    rel["instantes_invalidos"] = int(df["din_instante"].isna().sum())
    df = df.dropna(subset=["din_instante"])

    # duplicatas exatas por (instante, subsistema)
    dup = df.duplicated(subset=["din_instante", "id_subsistema"]).sum()
    rel["duplicatas_removidas"] = int(dup)
    df = df.drop_duplicates(subset=["din_instante", "id_subsistema"],
                            keep="first")

    # valores fisicamente impossiveis
    imp = int((df["val_cargaenergiahomwmed"] <= 0).sum())
    rel["valores_impossiveis"] = imp
    df.loc[df["val_cargaenergiahomwmed"] <= 0,
           "val_cargaenergiahomwmed"] = np.nan

    # carga total do SIN por hora (soma valida apenas com os 4 subsistemas)
    piv = df.pivot_table(index="din_instante", columns="id_subsistema",
                         values="val_cargaenergiahomwmed", aggfunc="first")
    completos = piv.notna().all(axis=1)
    rel["horas_subsistema_incompleto"] = int((~completos).sum())
    sin = piv.sum(axis=1).where(completos)   # hora incompleta vira NaN
    sin.name = "carga_mw"

    # grade horaria completa
    idx = pd.date_range(sin.index.min(), sin.index.max(), freq="h")
    sin = sin.reindex(idx)
    rel["horas_grade"] = int(len(sin))
    rel["lacunas_totais"] = int(sin.isna().sum())

    # interpola apenas lacunas curtas
    na = sin.isna()
    grupo = (na != na.shift()).cumsum()
    tam_gap = na.groupby(grupo).transform("sum").where(na, 0)
    interpolaveis = na & (tam_gap <= GAP_MAX_INTERP)
    sin_i = sin.interpolate(limit=GAP_MAX_INTERP, limit_area="inside")
    sin = sin.where(~interpolaveis, sin_i)
    rel["lacunas_interpoladas"] = int(interpolaveis.sum())
    rel["lacunas_restantes"] = int(sin.isna().sum())

    # outliers: desvio robusto contra o perfil (dow, hora) em janela movel
    perfil = sin.groupby([sin.index.dayofweek, sin.index.hour])
    mediana = perfil.transform("median")
    mad = perfil.transform(lambda s: (s - s.median()).abs().median())
    z_rob = (sin - mediana) / (1.4826 * mad.replace(0, np.nan))
    flag = z_rob.abs() > 6
    rel["outliers_sinalizados"] = int(flag.sum())
    top = (z_rob.abs().nlargest(5).index.strftime("%Y-%m-%d %H:%M").tolist()
           if flag.any() else [])
    rel["maiores_desvios"] = top

    rel["inicio"] = str(sin.index.min())
    rel["fim"] = str(sin.index.max())
    rel["carga_media_mw"] = round(float(sin.mean()), 1)

    out = sin.to_frame()
    out["flag_outlier"] = flag.astype(int)
    return out, rel


if __name__ == "__main__":
    print(">> Baixando e montando a serie do SIN...")
    df, rel = monta_serie()
    df.to_parquet(PROC / "carga_sin.parquet")
    (ART / "qualidade_dados.json").write_text(
        json.dumps(rel, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rel, indent=1, ensure_ascii=False))
    print(f"\nOK: {PROC / 'carga_sin.parquet'}")
