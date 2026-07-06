# -*- coding: utf-8 -*-
"""
Treino do modelo global multi-horizonte de carga do SIN.

Protocolo (nesta ordem, sem excecoes):
  1. Holdout temporal INTOCADO: origens a partir de 2026-01-01 ficam
     reservadas para o backtest final (src/evaluate.py). Nada aqui as ve.
  2. Validacao cruzada TEMPORAL dentro do treino: 4 folds com janelas de
     validacao trimestrais em 2025 e treino expansivo (so passado preve
     futuro; embaralhar seria vazamento).
  3. Tunagem de hiperparametros por busca aleatoria (seed fixa) sobre o
     LightGBM (objetivo MAE), com early stopping por fold.
  4. Filtro pelo canario de ruido: apos a tunagem, o modelo vencedor e
     ajustado com a feature "ruido_aleatorio"; toda feature com ganho
     menor ou igual ao do ruido e descartada, e a configuracao enxuta e
     re-validada nos mesmos folds para provar que nada de sinal se perdeu.
  5. Modelos finais quantilicos (P10/P50/P90) com os hiperparametros
     vencedores e as features sobreviventes, treinados no treino inteiro
     com early stopping em uma cauda de validacao (nov-dez/2025).

Uso: python src/train.py            (~20-40 min na primeira execucao)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import monta_supervisionado, colunas_features, nomes_feriados_top

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

SEED = 42
HOLDOUT_INICIO = "2026-01-01"
N_CONFIGS = 20
FOLDS = [  # (inicio_val, fim_val); treino = tudo antes de inicio_val
    ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"),
]
QUANTIS = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

rng = np.random.default_rng(SEED)


def espaco_busca(n=N_CONFIGS):
    """Busca aleatoria com semente fixa: reproducivel e sem vies de grade."""
    configs = []
    for _ in range(n):
        configs.append({
            "num_leaves": int(rng.choice([31, 63, 127, 255])),
            "learning_rate": float(rng.choice([0.03, 0.05, 0.08, 0.12])),
            "min_child_samples": int(rng.choice([50, 100, 200, 400])),
            "feature_fraction": float(rng.choice([0.6, 0.75, 0.9, 1.0])),
            "bagging_fraction": float(rng.choice([0.7, 0.85, 1.0])),
            "bagging_freq": 1,
            "lambda_l2": float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        })
    return configs


def fit_lgbm(Xtr, ytr, Xva, yva, params, objetivo="l1", alpha=None,
             n_estimators=3000, es=100):
    p = dict(objective=objetivo, metric="mae", verbosity=-1,
             n_estimators=n_estimators, random_state=SEED,
             n_jobs=-1, **params)
    if alpha is not None:
        p.update(objective="quantile", alpha=alpha, metric="quantile")
    modelo = lgb.LGBMRegressor(**p)
    modelo.fit(Xtr, ytr, eval_set=[(Xva, yva)],
               callbacks=[lgb.early_stopping(es, verbose=False),
                          lgb.log_evaluation(0)])
    return modelo


def cv_config(sup, feats, params, passo_tuning=2):
    """MAE medio nos folds temporais. No tuning, as origens do TREINO sao
    subamostradas (1 a cada `passo_tuning` horas) por custo; a validacao e
    sempre completa."""
    maes, iters = [], []
    for ini, fim in FOLDS:
        tr = sup[sup.index < ini]
        va = sup[(sup.index >= ini) & (sup.index <= fim + " 23:00")]
        if passo_tuning > 1:
            tr = tr[tr.index.hour % passo_tuning == 0]
        m = fit_lgbm(tr[feats], tr["y"], va[feats], va["y"], params)
        pred = m.predict(va[feats], num_iteration=m.best_iteration_)
        maes.append(float(np.mean(np.abs(va["y"] - pred))))
        iters.append(int(m.best_iteration_))
    return float(np.mean(maes)), maes, int(np.mean(iters))


def main():
    print(">> Montando dataset supervisionado...")
    df = pd.read_parquet(PROC / "carga_sin.parquet")
    nomes_top = nomes_feriados_top(df["carga_mw"])
    sup = monta_supervisionado(df, nomes_top=nomes_top)
    print(f"   {len(sup):,} pares (origem, horizonte) | "
          f"{sup.index.min()} a {sup.index.max()}")

    treino = sup[sup.index < HOLDOUT_INICIO]
    print(f"   treino: {len(treino):,} | holdout (intocado): "
          f"{(sup.index >= HOLDOUT_INICIO).sum():,}")

    feats = colunas_features(sup)

    # ---------------------------------------------- tunagem (busca aleatoria)
    print(f">> Tunagem: {N_CONFIGS} configs x {len(FOLDS)} folds temporais...")
    resultados = []
    for i, cfg in enumerate(espaco_busca(), 1):
        mae, maes_fold, it = cv_config(treino, feats, cfg)
        resultados.append({"config": cfg, "mae_cv": mae,
                           "mae_folds": maes_fold, "iteracoes": it})
        print(f"   [{i:02d}/{N_CONFIGS}] MAE {mae:8.1f} MW | {cfg}")
    resultados.sort(key=lambda r: r["mae_cv"])
    melhor = resultados[0]
    print(f"   vencedora: MAE {melhor['mae_cv']:.1f} MW | {melhor['config']}")

    # ------------------------------------------- canario de ruido: o filtro
    print(">> Filtro pelo canario de ruido...")
    ini_va, fim_va = FOLDS[-1]
    tr = treino[treino.index < ini_va]
    va = treino[(treino.index >= ini_va) & (treino.index <= fim_va + " 23:00")]
    m_full = fit_lgbm(tr[feats], tr["y"], va[feats], va["y"], melhor["config"])
    imp = pd.Series(m_full.booster_.feature_importance("gain"),
                    index=feats).sort_values(ascending=False)
    ganho_ruido = float(imp["ruido_aleatorio"])
    cortadas = [f for f in feats
                if imp[f] <= ganho_ruido and f != "ruido_aleatorio"]
    feats_ok = [f for f in feats
                if f not in cortadas and f != "ruido_aleatorio"]
    print(f"   ganho do ruido: {ganho_ruido:,.0f} | features cortadas "
          f"({len(cortadas)}): {cortadas}")

    mae_full, _, _ = cv_config(treino, feats, melhor["config"])
    mae_enxuto, maes_enxuto, it_enxuto = cv_config(treino, feats_ok,
                                                   melhor["config"])
    print(f"   MAE CV com todas ({len(feats)}): {mae_full:.1f} | "
          f"enxuto ({len(feats_ok)}): {mae_enxuto:.1f}")

    # ------------------------------------------------- modelos finais (P10/50/90)
    print(">> Modelos finais quantilicos (treino completo, es em nov-dez/25)...")
    tr_fin = treino[treino.index < "2025-11-01"]
    va_fin = treino[treino.index >= "2025-11-01"]
    metricas_finais = {}
    for nome, alpha in QUANTIS.items():
        m = fit_lgbm(tr_fin[feats_ok], tr_fin["y"],
                     va_fin[feats_ok], va_fin["y"],
                     melhor["config"], alpha=alpha)
        m.booster_.save_model(str(MODELS / f"lgbm_{nome}.txt"),
                              num_iteration=m.best_iteration_)
        metricas_finais[nome] = {"best_iteration": int(m.best_iteration_)}
        print(f"   {nome}: {m.best_iteration_} arvores")

    # ------------------------------------------------------------- artefatos
    imp_norm = (imp / imp.sum() * 100).round(3)
    (ART / "cv_resultados.json").write_text(json.dumps({
        "n_configs": N_CONFIGS,
        "folds": FOLDS,
        "vencedora": melhor,
        "todas": resultados,
    }, indent=1), encoding="utf-8")
    (ART / "importancias.json").write_text(json.dumps({
        "ganho_pct": imp_norm.to_dict(),
        "ganho_ruido_pct": float(imp_norm["ruido_aleatorio"]),
        "cortadas": cortadas,
        "mantidas": feats_ok,
        "mae_cv_todas": mae_full,
        "mae_cv_enxuto": mae_enxuto,
        "mae_folds_enxuto": maes_enxuto,
    }, indent=1), encoding="utf-8")
    cfg_final = json.dumps({
        "params": melhor["config"],
        "features": feats_ok,
        "nomes_feriados_top": nomes_top,
        "quantis": metricas_finais,
        "holdout_inicio": HOLDOUT_INICIO,
    }, indent=1)
    # artifacts/ e local (gitignore); models/ vai para o repo porque o ciclo
    # ao vivo no Actions precisa da config junto com os boosters
    (ART / "config_final.json").write_text(cfg_final, encoding="utf-8")
    (MODELS / "config_final.json").write_text(cfg_final, encoding="utf-8")
    print(f"\nOK. Modelos em {MODELS}, artefatos em {ART}")


if __name__ == "__main__":
    main()
