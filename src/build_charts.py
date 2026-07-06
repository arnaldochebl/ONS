# -*- coding: utf-8 -*-
"""
Gera os SVGs de docs/index.html a partir dos artefatos reais e injeta cada um
entre marcadores <!--SVG:nome:begin--> ... <!--SVG:nome:end-->.

Uso:
    python src/build_charts.py            # gera docs/_charts/*.svg
    python src/build_charts.py --splice   # gera e injeta em docs/index.html
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "_charts"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#334155", "#64748b", "#e6ecf4"
AZUL, AQUA, AMARELO, VERMELHO, VIOLETA = ("#2a78d6", "#1baf7a", "#eda100",
                                          "#e34948", "#4a3aa7")
FONT = "font-family='Segoe UI,system-ui,sans-serif'"


def fmt_br(v, nd=0):
    return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def nice_ticks(lo, hi, n=5):
    import math
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10, 20):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


class Chart:
    def __init__(self, w=660, h=320, ml=64, mr=100, mt=16, mb=40):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.parts = []

    def scales(self, x0, x1, y0, y1):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def X(self, v):
        return round(self.ml + (v - self.x0) / (self.x1 - self.x0)
                     * (self.w - self.ml - self.mr), 1)

    def Y(self, v):
        return round(self.h - self.mb - (v - self.y0) / (self.y1 - self.y0)
                     * (self.h - self.mt - self.mb), 1)

    def gridy(self, ticks, fmt=lambda v: f"{v:g}"):
        g = [f"<g stroke='{GRID}' stroke-width='1'>"]
        lbl = [f"<g font-size='11' fill='{MUTED}' text-anchor='end'>"]
        for t in ticks:
            y = self.Y(t)
            g.append(f"<line x1='{self.ml}' y1='{y}' x2='{self.w-self.mr}' y2='{y}'/>")
            lbl.append(f"<text x='{self.ml-7}' y='{y+4}'>{fmt(t)}</text>")
        self.parts += g + ["</g>"] + lbl + ["</g>"]

    def gridx(self, ticks, fmt=lambda v: f"{v:g}", title=""):
        lbl = [f"<g font-size='11' fill='{MUTED}' text-anchor='middle'>"]
        for t in ticks:
            lbl.append(f"<text x='{self.X(t)}' y='{self.h-self.mb+18}'>{fmt(t)}</text>")
        self.parts += lbl + ["</g>"]
        if title:
            self.parts.append(
                f"<text x='{(self.ml+self.w-self.mr)/2}' y='{self.h-6}' "
                f"font-size='11' fill='{MUTED}' text-anchor='middle'>{title}</text>")

    def line(self, xs, ys, color, width=2, dash="", opacity=1.0):
        pts = " ".join(f"{self.X(x)},{self.Y(y)}" for x, y in zip(xs, ys))
        d = f" stroke-dasharray='{dash}'" if dash else ""
        o = f" opacity='{opacity}'" if opacity < 1 else ""
        self.parts.append(f"<polyline points='{pts}' fill='none' stroke='{color}' "
                          f"stroke-width='{width}' stroke-linejoin='round' "
                          f"stroke-linecap='round'{d}{o}/>")

    def area(self, xs, y_lo, y_hi, color, opacity=0.18):
        sup = " ".join(f"{self.X(x)},{self.Y(y)}" for x, y in zip(xs, y_hi))
        inf = " ".join(f"{self.X(x)},{self.Y(y)}"
                       for x, y in zip(reversed(xs), reversed(y_lo)))
        self.parts.append(f"<polygon points='{sup} {inf}' fill='{color}' "
                          f"opacity='{opacity}'/>")

    def label(self, x, y, text, color, size=11.5, anchor="start", bold=True):
        w = " font-weight='700'" if bold else ""
        self.parts.append(f"<text x='{x}' y='{y}' font-size='{size}' "
                          f"fill='{color}' text-anchor='{anchor}'{w}>{text}</text>")

    def svg(self, aria):
        return (f"<svg class='viz' viewBox='0 0 {self.w} {self.h}' role='img' "
                f"{FONT} aria-label='{aria}'>" + "".join(self.parts) + "</svg>")


def spread(items, gap=15, hi=None):
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i-1] + gap)
    if hi is not None and ys and ys[-1] > hi:
        ys = [y - (ys[-1] - hi) for y in ys]
        for i in range(len(ys)-2, -1, -1):
            ys[i] = min(ys[i], ys[i+1] - gap)
    return [(y, *t[1:]) for y, t in zip(ys, items)]


MET = json.loads((ART / "metricas_horizonte.json").read_text())
IMP = json.loads((ART / "importancias.json").read_text())
FAN = json.loads((ART / "page_fanchart.json").read_text())
SER = json.loads((ART / "page_serie.json").read_text())


def chart_mae_horizonte():
    hs = MET["horizontes"]
    c = Chart(h=330)
    series = [("persistência", MET["persistencia"]["mae"], "#8a94a6", "6 5"),
              ("sazonal (7 dias)", MET["sazonal"]["mae"], VERMELHO, ""),
              ("modelo (P50)", MET["modelo"]["mae"], AQUA, "")]
    ymax = max(max(s[1]) for s in series) * 1.08
    c.scales(1, 24, 0, ymax)
    c.gridy(nice_ticks(0, ymax, 5), fmt=lambda v: fmt_br(v))
    c.gridx([1, 4, 8, 12, 16, 20, 24], title="horizonte da previsão (horas à frente)")
    ends = []
    for nome, ys, cor, dash in series:
        c.line(hs, ys, cor, 2.4, dash=dash)
        ends.append((c.Y(ys[-1]), nome, cor))
    for y, nome, cor in spread(ends, hi=c.h - c.mb - 3):
        c.label(c.w - c.mr + 8, y + 4, nome, cor, 11)
    return c.svg("Erro absoluto medio em MW por horizonte de previsao: modelo "
                 "contra persistencia e baseline sazonal de 7 dias")


def chart_fanchart():
    c = Chart(h=330, mr=24)
    xs_ctx = list(range(-len(FAN["contexto"]) + 1, 1))
    xs_alvo = list(range(1, 25))
    todos = FAN["contexto"] + FAN["p10"] + FAN["p90"] + FAN["realizado"]
    lo, hi = min(todos) * 0.98, max(todos) * 1.02
    c.scales(xs_ctx[0], 24, lo, hi)
    c.gridy(nice_ticks(lo, hi, 5), fmt=lambda v: fmt_br(v / 1000, 0))
    c.gridx([-48, -24, 0, 6, 12, 18, 24],
            fmt=lambda v: ("origem" if v == 0 else f"{v:+.0f}h"),
            title="horas em relação à origem da previsão")
    x0 = c.X(0)
    c.parts.append(f"<line x1='{x0}' y1='{c.mt}' x2='{x0}' y2='{c.h-c.mb}' "
                   f"stroke='#94a3b8' stroke-width='1.2' stroke-dasharray='4 4'/>")
    c.area(xs_alvo, FAN["p10"], FAN["p90"], AQUA, 0.20)
    c.line(xs_ctx, FAN["contexto"], "#8a94a6", 1.8)
    c.line(xs_alvo, FAN["p50"], AQUA, 2.6)
    c.line(xs_alvo, FAN["realizado"], "#0f172a", 2.2)
    c.line(xs_alvo, FAN["sazonal"], VERMELHO, 1.6, dash="5 4")
    c.label(c.X(12), c.mt + 14, "intervalo P10-P90", "#0e7a55", 11, "middle")
    c.label(c.X(xs_ctx[0]) + 4, c.Y(FAN["contexto"][0]) - 8,
            "últimas 48h observadas", MUTED, 10.5, bold=False)
    ends = [(c.Y(FAN["p50"][-1]), "P50", AQUA),
            (c.Y(FAN["realizado"][-1]), "realizado", "#0f172a"),
            (c.Y(FAN["sazonal"][-1]), "sazonal", VERMELHO)]
    for y, nome, cor in spread(ends, 14, hi=c.h - c.mb - 3):
        c.label(c.w - c.mr + 6, y + 4, nome, cor, 10.5)
    return c.svg("Exemplo real do holdout: previsao quantilica das proximas "
                 "24 horas contra o realizado e o baseline sazonal")


def chart_importancias():
    ganhos = IMP["ganho_pct"]
    ruido = IMP["ganho_ruido_pct"]
    top = sorted(ganhos.items(), key=lambda kv: -kv[1])[:16]
    if not any(k == "ruido_aleatorio" for k, _ in top):
        top = top[:15] + [("ruido_aleatorio", ruido)]
    c = Chart(w=660, h=26 * len(top) + 60, ml=190, mr=70, mt=12, mb=34)
    vmax = max(v for _, v in top) * 1.06
    c.scales(0, vmax, 0, 1)
    xt = nice_ticks(0, vmax, 5)
    lbl = [f"<g font-size='10.5' fill='{MUTED}' text-anchor='middle'>"]
    for t in xt:
        x = c.X(t)
        c.parts.append(f"<line x1='{x}' y1='{c.mt}' x2='{x}' "
                       f"y2='{c.h-c.mb}' stroke='{GRID}'/>")
        lbl.append(f"<text x='{x}' y='{c.h-c.mb+16}'>{fmt_br(t, 1)}%</text>")
    c.parts += lbl + ["</g>"]
    c.parts.append(f"<text x='{(c.ml+c.w-c.mr)/2}' y='{c.h-4}' font-size='11' "
                   f"fill='{MUTED}' text-anchor='middle'>participação no ganho "
                   f"total do modelo (%)</text>")
    x_ruido = c.X(ruido)
    c.parts.append(f"<line x1='{x_ruido}' y1='{c.mt}' x2='{x_ruido}' "
                   f"y2='{c.h-c.mb}' stroke='{VERMELHO}' stroke-width='1.5' "
                   f"stroke-dasharray='4 3'/>")
    for i, (nome, v) in enumerate(top):
        y = c.mt + 6 + i * 26
        cor = VERMELHO if nome == "ruido_aleatorio" else (
            "#c3cddc" if nome in IMP["cortadas"] else AQUA)
        c.parts.append(f"<rect x='{c.ml}' y='{y}' width='{c.X(v)-c.ml}' "
                       f"height='16' rx='4' fill='{cor}'/>")
        c.label(c.ml - 8, y + 13, nome, INK if cor == AQUA else MUTED, 10.5,
                "end", bold=(cor == AQUA))
        c.label(c.X(v) + 6, y + 13, f"{fmt_br(v, 1)}%", MUTED, 10, bold=False)
    return c.svg("Importancia por ganho das principais features, com o ruido "
                 "aleatorio marcado como regua de corte")


def chart_serie():
    xs = list(range(len(SER["carga"])))
    c = Chart(h=280, mr=24)
    lo, hi = min(SER["carga"]) * 0.97, max(SER["carga"]) * 1.03
    c.scales(0, xs[-1], lo, hi)
    c.gridy(nice_ticks(lo, hi, 5), fmt=lambda v: fmt_br(v / 1000, 0))
    # ticks semanais
    import pandas as pd
    datas = pd.to_datetime(SER["datas"])
    lbl = [f"<g font-size='10.5' fill='{MUTED}' text-anchor='middle'>"]
    for i, d in enumerate(datas):
        if d.dayofweek == 0 and d.hour == 0:
            x = c.X(i)
            c.parts.append(f"<line x1='{x}' y1='{c.mt}' x2='{x}' "
                           f"y2='{c.h-c.mb}' stroke='{GRID}'/>")
            lbl.append(f"<text x='{x}' y='{c.h-c.mb+16}'>{d.strftime('%d/%m')}</text>")
    c.parts += lbl + ["</g>"]
    c.line(xs, SER["carga"], AZUL, 1.8)
    return c.svg("Carga do SIN nas ultimas quatro semanas do periodo avaliado, "
                 "em medias de tres horas: ciclo diario e semanal visiveis")


def main():
    frags = {
        "mae": chart_mae_horizonte(),
        "fan": chart_fanchart(),
        "importancias": chart_importancias(),
        "serie": chart_serie(),
    }
    for k, v in frags.items():
        (OUT / f"{k}.svg").write_text(v, encoding="utf-8")
        print(f"   docs/_charts/{k}.svg ({len(v)/1024:.0f} KB)")
    if "--splice" in sys.argv:
        page = ROOT / "docs" / "index.html"
        html = page.read_text(encoding="utf-8")
        for k, v in frags.items():
            pat = re.compile(f"(<!--SVG:{k}:begin-->).*?(<!--SVG:{k}:end-->)", re.S)
            if not pat.search(html):
                print(f"   [aviso] marcador SVG:{k} ausente")
                continue
            html = pat.sub(lambda m, v=v: m.group(1) + v + m.group(2), html)
        page.write_text(html, encoding="utf-8")
        print("   docs/index.html atualizado")


if __name__ == "__main__":
    main()
