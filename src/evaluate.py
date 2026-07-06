# -*- coding: utf-8 -*-
"""
Backtest final no holdout intocado (origens de 2026-01-01 em diante).

Compara, por horizonte de 1h a 24h:
  - Modelo (LightGBM quantilico, P50 como previsao pontual)
  - Baseline de persistencia: previsao = carga na origem
  - Baseline sazonal-semanal: previsao = carga na mesma hora ha 7 dias
    (o baseline forte de series com dupla sazonalidade; e ele que precisa
    ser batido para o modelo merecer existir)

Metricas: MAE, RMSE, MAPE por horizonte; skill = 1 - MAE/MAE_sazonal;
cobertura empirica do intervalo P10-P90 (nominal: 80%).

Gera tambem os artefatos consumidos pela pagina (docs/), sempre a partir
dos numeros reais deste backtest.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import monta_supervisionado

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

CFG = json.loads((ART / "config_final.json").read_text())
FEATS = CFG["features"]
HOLDOUT_INICIO = CFG["holdout_inicio"]


def carrega_modelos():
    return {q: lgb.Booster(model_file=str(MODELS / f"lgbm_{q}.txt"))
            for q in ("p10", "p50", "p90")}


def main():
    print(">> Backtest no holdout...")
    df = pd.read_parquet(PROC / "carga_sin.parquet")
    sup = monta_supervisionado(df, nomes_top=CFG["nomes_feriados_top"],
                               incluir_ruido=False)
    hold = sup[sup.index >= HOLDOUT_INICIO].copy()
    print(f"   {len(hold):,} pares | origens {hold.index.min()} a "
          f"{hold.index.max()}")

    modelos = carrega_modelos()
    X = hold[FEATS]
    for q, b in modelos.items():
        hold[q] = b.predict(X)
    # coerencia dos quantis (cruzamentos raros, mas corrigimos por sort)
    tripla = np.sort(hold[["p10", "p50", "p90"]].values, axis=1)
    hold[["p10", "p50", "p90"]] = tripla

    # calibracao conformal (CQR): alarga o intervalo pela folga q_hat[h]
    # aprendida em nov-dez/2025 (fatia nunca vista pelo ajuste)
    conf_path = MODELS / "conformal.json"
    cobertura_bruta = ((hold["y"] >= hold["p10"]) & (hold["y"] <= hold["p90"])
                       ).groupby(hold["horizonte"]).mean() * 100
    if conf_path.exists():
        q_hat = {int(k): v for k, v in json.loads(
            conf_path.read_text(encoding="utf-8"))["q_hat_mw"].items()}
        folga = hold["horizonte"].map(q_hat)
        hold["p10"] = hold["p10"] - folga
        hold["p90"] = hold["p90"] + folga

    # baselines
    hold["naive_persistencia"] = hold["carga_origem"]
    hold["naive_sazonal"] = hold["ref_semana_mesma_hora"]

    def metricas(pred):
        err = hold["y"] - pred
        return pd.DataFrame({
            "mae": err.abs().groupby(hold["horizonte"]).mean(),
            "rmse": (err ** 2).groupby(hold["horizonte"]).mean() ** 0.5,
            "mape": (err.abs() / hold["y"]).groupby(hold["horizonte"]).mean() * 100,
        })

    m_modelo = metricas(hold["p50"])
    m_pers = metricas(hold["naive_persistencia"])
    m_saz = metricas(hold["naive_sazonal"])

    cobertura = ((hold["y"] >= hold["p10"]) & (hold["y"] <= hold["p90"])
                 ).groupby(hold["horizonte"]).mean() * 100
    pinball = {}
    for q, a in (("p10", .1), ("p50", .5), ("p90", .9)):
        d = hold["y"] - hold[q]
        pinball[q] = float(np.mean(np.maximum(a * d, (a - 1) * d)))

    skill = (1 - m_modelo["mae"] / m_saz["mae"]) * 100

    print(f"   MAE medio 1-24h: modelo {m_modelo['mae'].mean():.0f} MW | "
          f"sazonal {m_saz['mae'].mean():.0f} | persistencia "
          f"{m_pers['mae'].mean():.0f}")
    print(f"   skill medio vs sazonal: {skill.mean():.1f}% | cobertura "
          f"P10-P90: {cobertura.mean():.1f}% (nominal 80%)")

    # ------------------------------------------------------------- artefatos
    (ART / "metricas_horizonte.json").write_text(json.dumps({
        "horizontes": m_modelo.index.tolist(),
        "modelo": {c: [round(v, 2) for v in m_modelo[c]] for c in m_modelo},
        "persistencia": {c: [round(v, 2) for v in m_pers[c]] for c in m_pers},
        "sazonal": {c: [round(v, 2) for v in m_saz[c]] for c in m_saz},
        "skill_pct": [round(v, 2) for v in skill],
        "cobertura_p10_p90_pct": [round(v, 2) for v in cobertura],
        "cobertura_bruta_pct": [round(v, 2) for v in cobertura_bruta],
        "pinball": pinball,
        "n_pares": int(len(hold)),
        "janela": [str(hold.index.min()), str(hold.index.max())],
    }, indent=1), encoding="utf-8")

    # fan chart: previsao feita numa origem fixa exemplar + realizado
    origem_ex = hold.index.unique().sort_values()[-24 * 7]  # ~1 semana antes do fim
    ex = hold.loc[[origem_ex]].sort_values("horizonte")
    alvo_idx = [str(origem_ex + pd.Timedelta(hours=int(h)))
                for h in ex["horizonte"]]
    # contexto: 48h antes da origem
    ctx = df.loc[origem_ex - pd.Timedelta(hours=48):origem_ex, "carga_mw"]
    (ART / "page_fanchart.json").write_text(json.dumps({
        "origem": str(origem_ex),
        "contexto_datas": [str(i) for i in ctx.index],
        "contexto": [round(float(v), 1) for v in ctx],
        "alvo_datas": alvo_idx,
        "p10": [round(float(v), 1) for v in ex["p10"]],
        "p50": [round(float(v), 1) for v in ex["p50"]],
        "p90": [round(float(v), 1) for v in ex["p90"]],
        "realizado": [round(float(v), 1) for v in ex["y"]],
        "sazonal": [round(float(v), 1) for v in ex["naive_sazonal"]],
    }, indent=1), encoding="utf-8")

    # perfil da serie para a pagina (semana media por hora/dow, e 4 semanas recentes)
    rec = df["carga_mw"].loc[str(hold.index.max() - pd.Timedelta(days=28)):]
    rec_w = rec.resample("3h").mean()
    (ART / "page_serie.json").write_text(json.dumps({
        "datas": [str(i) for i in rec_w.index],
        "carga": [round(float(v), 1) for v in rec_w],
    }, indent=1), encoding="utf-8")

    resumo = {
        "mae_modelo_medio": round(float(m_modelo["mae"].mean()), 1),
        "mae_sazonal_medio": round(float(m_saz["mae"].mean()), 1),
        "mae_persistencia_medio": round(float(m_pers["mae"].mean()), 1),
        "mape_modelo_medio_pct": round(float(m_modelo["mape"].mean()), 2),
        "skill_medio_pct": round(float(skill.mean()), 1),
        "skill_h1_pct": round(float(skill.iloc[0]), 1),
        "skill_h24_pct": round(float(skill.iloc[-1]), 1),
        "cobertura_media_pct": round(float(cobertura.mean()), 1),
        "cobertura_bruta_media_pct": round(float(cobertura_bruta.mean()), 1),
        "n_pares_holdout": int(len(hold)),
    }
    (ART / "summary_backtest.json").write_text(
        json.dumps(resumo, indent=1), encoding="utf-8")
    print(json.dumps(resumo, indent=1))
    print("\nOK.")


if __name__ == "__main__":
    main()
