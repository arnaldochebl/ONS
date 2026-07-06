# -*- coding: utf-8 -*-
"""
Modelos finais quantilicos + calibracao conformal (CQR) por horizonte.

Motivacao: os quantis brutos do gradient boosting sairam estreitos (cobertura
de 66% para um intervalo nominal de 80% no primeiro backtest). Em vez de
aceitar ou de alargar no olho, aplicamos Conformalized Quantile Regression
(Romano, Patterson & Candes, 2019):

  1. Ajuste: P10/P50/P90 treinados ate 31/08/2025, com early stopping em
     set-out/2025.
  2. Calibracao: nov-dez/2025, fatia que o ajuste NUNCA viu. Score conformal
     s = max(P10 - y, y - P90); o quantil (1 - alfa) ajustado de s, por
     horizonte, vira a folga q_hat[h].
  3. Intervalo final: [P10 - q_hat, P90 + q_hat]. Garantia de cobertura
     marginal ~80% sob trocabilidade; o holdout de 2026 testa na pratica.

Uso: python src/finalize.py     (substitui os modelos finais do train.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import monta_supervisionado
from train import fit_lgbm, QUANTIS, HOLDOUT_INICIO

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

ALFA = 0.20                       # intervalo nominal de 80%
FIM_AJUSTE = "2025-09-01"         # treino dos boosters: antes daqui
FIM_ES = "2025-11-01"             # early stopping: set-out/2025
# calibracao conformal: nov-dez/2025 (nunca vista pelo ajuste)


def main():
    cfg = json.loads((MODELS / "config_final.json").read_text(encoding="utf-8"))
    feats, params = cfg["features"], cfg["params"]

    print(">> Montando dataset...")
    df = pd.read_parquet(PROC / "carga_sin.parquet")
    sup = monta_supervisionado(df, nomes_top=cfg["nomes_feriados_top"],
                               incluir_ruido=False)
    treino = sup[sup.index < HOLDOUT_INICIO]

    tr = treino[treino.index < FIM_AJUSTE]
    va = treino[(treino.index >= FIM_AJUSTE) & (treino.index < FIM_ES)]
    calib = treino[treino.index >= FIM_ES]
    print(f"   ajuste: {len(tr):,} | early stopping: {len(va):,} | "
          f"calibracao conformal: {len(calib):,}")

    print(">> Treinando quantilicos...")
    pred_cal = {}
    quantis_meta = {}
    for nome, alpha in QUANTIS.items():
        m = fit_lgbm(tr[feats], tr["y"], va[feats], va["y"], params,
                     alpha=alpha)
        m.booster_.save_model(str(MODELS / f"lgbm_{nome}.txt"),
                              num_iteration=m.best_iteration_)
        pred_cal[nome] = m.predict(calib[feats],
                                   num_iteration=m.best_iteration_)
        quantis_meta[nome] = {"best_iteration": int(m.best_iteration_)}
        print(f"   {nome}: {m.best_iteration_} arvores")

    print(">> Calibracao conformal (CQR) por horizonte...")
    lo = np.minimum(pred_cal["p10"], pred_cal["p90"])
    hi = np.maximum(pred_cal["p10"], pred_cal["p90"])
    y = calib["y"].values
    score = np.maximum(lo - y, y - hi)
    h = calib["horizonte"].values
    q_hat = {}
    for hz in range(1, 25):
        s = np.sort(score[h == hz])
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - ALFA))) - 1
        q_hat[hz] = float(s[min(max(k, 0), n - 1)])
    cobertura_bruta = float(np.mean((y >= lo) & (y <= hi)) * 100)
    print(f"   cobertura bruta na calibracao: {cobertura_bruta:.1f}% | "
          f"q_hat mediano: {np.median(list(q_hat.values())):.0f} MW "
          f"(h1 {q_hat[1]:.0f} | h24 {q_hat[24]:.0f})")

    (MODELS / "conformal.json").write_text(json.dumps({
        "alfa": ALFA,
        "janela_calibracao": [FIM_ES, str(calib.index.max())],
        "cobertura_bruta_calibracao_pct": round(cobertura_bruta, 1),
        "q_hat_mw": {str(k): round(v, 1) for k, v in q_hat.items()},
    }, indent=1), encoding="utf-8")

    cfg["quantis"] = quantis_meta
    cfg["conformal"] = True
    txt = json.dumps(cfg, indent=1)
    (MODELS / "config_final.json").write_text(txt, encoding="utf-8")
    (ART / "config_final.json").write_text(txt, encoding="utf-8")
    print(f"OK. Boosters e conformal.json em {MODELS}")


if __name__ == "__main__":
    main()
