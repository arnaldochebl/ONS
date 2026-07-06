# -*- coding: utf-8 -*-
"""
Ciclo "vivo" do modelo: roda semanalmente no GitHub Actions (custo zero).

A cada execucao:
  1. Baixa a serie mais recente do ONS (a publicacao tem ~1 semana de
     defasagem em relacao ao relogio).
  2. Gera a previsao P10/P50/P90 para as 24h seguintes ao ultimo instante
     disponivel e a REGISTRA em docs/live/previsoes.json. O carimbo do
     commit e a prova de que a previsao foi feita antes de o realizado
     ser publicado.
  3. Reconcilia previsoes antigas cujo realizado ja chegou e atualiza o
     track record publico (docs/live/track_record.json), que a pagina
     desenha: previsto vs realizado acumulando semana a semana.

Uso: python src/score_live.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import monta_serie

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "docs" / "live"
LIVE.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
CFG = json.loads((MODELS / "config_final.json").read_text(encoding="utf-8"))


def carrega(nome_arq, default):
    p = LIVE / nome_arq
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main():
    agora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f">> Ciclo ao vivo: {agora}")
    df, _ = monta_serie()
    serie = df["carga_mw"].dropna()
    t_ult = serie.index.max()
    print(f"   ultimo dado publicado pelo ONS: {t_ult}")

    # features da ULTIMA origem disponivel (previsao das proximas 24h);
    # para a origem final nao existe y, entao o frame e montado a mao com
    # exatamente as mesmas funcoes de feature usadas no treino
    from features import features_origem, _calendario_alvo, _tabela_feriados
    janela = df.loc[t_ult - pd.Timedelta(days=30):]
    fo = features_origem(janela["carga_mw"]).loc[[t_ult]]
    anos = range(t_ult.year - 1, t_ult.year + 2)
    fer = _tabela_feriados(anos)
    linhas = []
    for h in range(1, 25):
        b = fo.copy()
        b["horizonte"] = np.int8(h)
        b["ref_ontem_mesma_hora"] = janela["carga_mw"].shift(24 - h).loc[t_ult]
        b["ref_semana_mesma_hora"] = janela["carga_mw"].shift(168 - h).loc[t_ult]
        cal = _calendario_alvo(pd.DatetimeIndex([t_ult + pd.Timedelta(hours=h)]),
                               fer, CFG["nomes_feriados_top"])
        cal.index = [t_ult]
        linhas.append(pd.concat([b, cal], axis=1))
    X = pd.concat(linhas)
    X["nome_feriado"] = pd.Categorical(
        X["nome_feriado"],
        categories=sorted(set(CFG["nomes_feriados_top"]
                              + ["nao_feriado", "outros"])))
    X = X[CFG["features"]]

    boosters = {q: lgb.Booster(model_file=str(MODELS / f"lgbm_{q}.txt"))
                for q in ("p10", "p50", "p90")}
    pred = {q: b.predict(X) for q, b in boosters.items()}
    tripla = np.sort(np.column_stack([pred["p10"], pred["p50"],
                                      pred["p90"]]), axis=1)
    # folga conformal (CQR) por horizonte, a mesma do backtest
    conf = json.loads((MODELS / "conformal.json").read_text(encoding="utf-8"))
    q_hat = np.array([conf["q_hat_mw"][str(h)] for h in range(1, 25)])
    tripla[:, 0] -= q_hat
    tripla[:, 2] += q_hat

    novo_lote = {
        "gerado_em": agora,
        "origem": str(t_ult),
        "previsoes": [{
            "alvo": str(t_ult + pd.Timedelta(hours=h)),
            "h": h,
            "p10": round(float(tripla[h - 1, 0]), 1),
            "p50": round(float(tripla[h - 1, 1]), 1),
            "p90": round(float(tripla[h - 1, 2]), 1),
        } for h in range(1, 25)],
    }

    historico = carrega("previsoes.json", [])
    # evita duplicar lote da mesma origem (reruns manuais)
    historico = [l for l in historico if l["origem"] != novo_lote["origem"]]
    historico.append(novo_lote)
    (LIVE / "previsoes.json").write_text(
        json.dumps(historico, indent=1), encoding="utf-8")
    print(f"   lote registrado (origem {novo_lote['origem']}); "
          f"{len(historico)} lotes no historico")

    # ------------------------------------------------- reconciliacao
    aval = []
    for lote in historico:
        for p in lote["previsoes"]:
            alvo = pd.Timestamp(p["alvo"])
            if alvo in serie.index:
                real = float(serie.loc[alvo])
                aval.append({**p, "origem": lote["origem"],
                             "gerado_em": lote["gerado_em"],
                             "realizado": round(real, 1)})
    tr = {"atualizado_em": agora, "n_avaliadas": len(aval)}
    if aval:
        da = pd.DataFrame(aval)
        da["erro_abs"] = (da["realizado"] - da["p50"]).abs()
        da["dentro"] = ((da["realizado"] >= da["p10"])
                        & (da["realizado"] <= da["p90"]))
        tr.update({
            "mae_mw": round(float(da["erro_abs"].mean()), 1),
            "mape_pct": round(float((da["erro_abs"] / da["realizado"])
                                    .mean() * 100), 2),
            "cobertura_p10_p90_pct": round(float(da["dentro"].mean() * 100), 1),
            "mae_por_h": {int(h): round(float(g["erro_abs"].mean()), 1)
                          for h, g in da.groupby("h")},
            "pontos": da.sort_values("alvo").tail(400)[
                ["alvo", "p10", "p50", "p90", "realizado"]
            ].to_dict("records"),
        })
        print(f"   track record: {len(aval)} previsoes avaliadas | "
              f"MAE {tr['mae_mw']} MW | cobertura {tr['cobertura_p10_p90_pct']}%")
    (LIVE / "track_record.json").write_text(
        json.dumps(tr, indent=1), encoding="utf-8")
    print("OK.")


if __name__ == "__main__":
    main()
