"""Parte I: analise estrutural do wiki-Talk (G85).

Exato: vertices, arestas, graus, densidade, componentes, clusterizacao, triangulos.
Estimado: diametro, raio e comprimento medio de caminhos (BFS de uma amostra de origens).
"""
import os
import random
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tratamento import carregar

SEED = 42
N_AMOSTRA = 300      # origens para estimar caminhos (nao dirigido)
N_AMOSTRA_DIR = 100  # origens para a versao dirigida
RES = "results"
FIG = f"{RES}/figures"
rng = np.random.default_rng(SEED)
random.seed(SEED)  # o igraph usa o gerador do Python (layouts)
met = {}


def stats_grau(nome, graus):
    graus = np.asarray(graus)
    met[f"{nome}_min"] = int(graus.min())
    met[f"{nome}_max"] = int(graus.max())
    met[f"{nome}_medio"] = float(graus.mean())
    met[f"{nome}_mediana"] = float(np.median(graus))


def ccdf(graus):
    x = np.sort(np.asarray(graus))
    y = 1.0 - np.arange(len(x)) / len(x)
    return x, y


def dist_amostra(g, fontes, mode="all"):
    """Distancias por BFS a partir de cada fonte (inalcancaveis ignorados)."""
    soma, pares, ecc = 0, 0, []
    for s in fontes:
        d = np.array(g.distances(source=int(s), mode=mode)[0], dtype=float)
        d = d[np.isfinite(d) & (d > 0)]
        soma += d.sum()
        pares += len(d)
        ecc.append(d.max() if len(d) else 0)
    return soma / pares, np.array(ecc), pares


def double_sweep(g, inicio):
    """Limite inferior do diametro: BFS ate o mais distante e BFS de novo."""
    d = np.array(g.distances(source=int(inicio))[0], dtype=float)
    d[~np.isfinite(d)] = -1
    a = int(d.argmax())
    d2 = np.array(g.distances(source=a)[0], dtype=float)
    d2[~np.isfinite(d2)] = -1
    return int(d2.max())


def main():
    os.makedirs(FIG, exist_ok=True)
    gd, gu = carregar("dir"), carregar("undir")
    gdl, gul = carregar("dir_lcc"), carregar("undir_lcc")
    n, m = gd.vcount(), gd.ecount()

    # ---- tamanho e graus (grafo tratado completo) ----
    met["vertices"], met["arestas_dir"], met["arestas_nao_dir"] = n, m, gu.ecount()
    met["densidade_dir"] = m / (n * (n - 1))
    met["densidade_nao_dir"] = 2 * gu.ecount() / (n * (n - 1))
    gin, gout, gtot = gd.indegree(), gd.outdegree(), gu.degree()
    stats_grau("grau_in", gin)
    stats_grau("grau_out", gout)
    stats_grau("grau_total_nao_dir", gtot)
    met["vertices_in0"] = int(np.sum(np.array(gin) == 0))
    met["vertices_out0"] = int(np.sum(np.array(gout) == 0))
    met["top_hub_grau_total"] = int(max(gtot))
    top1 = np.argsort(gtot)[-int(0.01 * n):]
    met["frac_arestas_top1pct_vertices"] = float(np.array(gtot)[top1].sum() / (2 * gu.ecount()))

    # ---- componentes ----
    cw = gd.connected_components(mode="weak")
    cs = gd.connected_components(mode="strong")
    tw, ts = np.array(cw.sizes()), np.array(cs.sizes())
    met["comp_fracas"], met["comp_fortes"] = len(tw), len(ts)
    met["maior_comp_fraca"], met["maior_comp_forte"] = int(tw.max()), int(ts.max())
    met["frac_maior_fraca"], met["frac_maior_forte"] = tw.max() / n, ts.max() / n
    met["comp_fortes_tamanho1"] = int((ts == 1).sum())
    pd.Series(tw).value_counts().sort_index().rename("qtd").to_csv(
        f"{RES}/tamanhos_componentes_fracas.csv", index_label="tamanho")
    pd.Series(ts).value_counts().sort_index().rename("qtd").head(30).to_csv(
        f"{RES}/tamanhos_componentes_fortes.csv", index_label="tamanho")

    # ---- clusterizacao e triangulos (nao dirigido) ----
    met["transitividade_global"] = gu.transitivity_undirected()
    cc = np.array(gu.transitivity_local_undirected(mode="zero"))
    met["clusterizacao_media_zero"] = float(cc.mean())
    deg = np.array(gtot, dtype=float)
    met["triangulos_via_clusterizacao"] = int(round((cc * deg * (deg - 1) / 2).sum() / 3))
    met["triangulos"] = len(gu.list_triangles())  # contagem exata
    ok = deg >= 2
    met["clusterizacao_media_grau_ge2"] = float(cc[ok].mean())

    # ---- caminhos (estimados na maior componente) ----
    met["lcc_vertices"], met["lcc_arestas_nao_dir"] = gul.vcount(), gul.ecount()
    fontes = rng.choice(gul.vcount(), N_AMOSTRA, replace=False)
    L, ecc, pares = dist_amostra(gul, fontes)
    met["caminho_medio_est"] = float(L)
    met["diametro_lb_amostra"] = int(ecc.max())
    met["raio_ub_amostra"] = int(ecc.min())
    met["diametro_lb_double_sweep"] = max(double_sweep(gul, f) for f in fontes[:50])
    met["origens_amostradas"] = N_AMOSTRA
    # raio: os hubs costumam ser os centros; excentricidade dos 50 maiores graus
    hubs = np.argsort(gul.degree())[-50:]
    ecc_hubs = np.array(gul.eccentricity(vertices=hubs.tolist()))
    met["raio_ub_hubs"] = int(ecc_hubs.min())
    met["raio_ub"] = min(met["raio_ub_amostra"], met["raio_ub_hubs"])
    met["diametro_lb"] = max(met["diametro_lb_amostra"], met["diametro_lb_double_sweep"],
                             int(ecc_hubs.max()))

    # dirigido (a): origens com grau de saida > 0 na maior componente fraca
    outdeg = np.array(gdl.outdegree())
    cand = np.nonzero(outdeg > 0)[0]
    met["dir_lcc_vertices_out_gt0"] = len(cand)
    fd = rng.choice(cand, N_AMOSTRA_DIR, replace=False)
    Ld, eccd, paresd = dist_amostra(gdl, fd, mode="out")
    met["dir_caminho_medio_est"] = float(Ld)
    met["dir_diametro_lb_amostra"] = int(eccd.max())
    met["dir_alcancaveis_medio_por_origem"] = paresd / N_AMOSTRA_DIR
    met["dir_frac_pares_alcancaveis"] = paresd / (N_AMOSTRA_DIR * (gdl.vcount() - 1))

    # dirigido (b): dentro da maior componente fortemente conexa
    gs = gd.connected_components(mode="strong").giant()
    fs = rng.choice(gs.vcount(), N_AMOSTRA_DIR, replace=False)
    Ls, eccs, _ = dist_amostra(gs, fs, mode="out")
    met["scc_vertices"], met["scc_arestas"] = gs.vcount(), gs.ecount()
    met["scc_caminho_medio_est"] = float(Ls)
    met["scc_diametro_lb"] = int(eccs.max())
    met["scc_raio_ub"] = int(eccs.min())

    pd.Series(met).to_csv(f"{RES}/parte1_metricas.csv", header=["valor"], index_label="metrica")
    print(pd.Series(met).to_string())

    # ---- figuras ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (nome, gr) in zip(ax, [("entrada", gin), ("saída", gout), ("total", gtot)]):
        v = np.bincount(gr)
        k = np.nonzero(v)[0]
        k = k[k > 0]
        a.loglog(k, v[k] / len(gr), ".", ms=3)
        a.set_title("Grau total" if nome == "total" else f"Grau de {nome}")
        a.set_xlabel("grau k")
        a.set_ylabel("P(k)")
    plt.tight_layout()
    plt.savefig(f"{FIG}/distribuicao_graus.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3.6))
    for nome, gr in [("entrada", gin), ("saída", gout), ("total", gtot)]:
        x, y = ccdf(gr)
        plt.loglog(x[x > 0], y[x > 0], label=nome)
    plt.xlabel("grau k")
    plt.ylabel("P(K ≥ k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG}/ccdf_graus.png", dpi=200)
    plt.close()

    # visualizacao reduzida: 300 vertices de maior grau e as arestas entre eles
    # (conectado na pratica, pois os hubs se ligam entre si); fica legivel.
    top = np.argsort(gtot)[-300:]
    sub = gu.induced_subgraph(top)
    sub = sub.induced_subgraph(max(sub.connected_components(), key=len))
    lay = np.array(sub.layout_fruchterman_reingold())
    d = np.array(sub.degree())
    plt.figure(figsize=(6, 6))
    for e in sub.es:
        plt.plot(lay[[e.source, e.target], 0], lay[[e.source, e.target], 1],
                 color="gray", lw=0.2, alpha=0.5)
    plt.scatter(lay[:, 0], lay[:, 1], s=8 + 4 * np.sqrt(d), c=np.log1p(d), cmap="viridis", zorder=3)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG}/visualizacao_hubs.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
