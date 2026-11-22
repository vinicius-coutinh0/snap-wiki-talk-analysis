"""Parte III, Apendice C: robustez a remocao aleatoria e a ataques.

Grafo: maior componente nao dirigida. n = |V|, r = ceil(0,05 n).
Metricas apos remover r vertices (e as arestas incidentes):
  A tamanho da maior componente, B numero de componentes,
  C distancias (L e diametro estimados por BFS de uma amostra de origens
    na maior componente), D fracao de vertices isolados.
"""
import heapq
import math
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tratamento import carregar

SEED = 42
T = 30            # repeticoes da remocao aleatoria
ORIGENS = 20      # origens por medicao de distancia (50 nos ataques)
RES, FIG = "results", "results/figures"


def medidas(g, rng, n_orig, n_origens):
    """Metricas A a D de g. n_orig = vertices do grafo antes da remocao."""
    comp = g.connected_components()
    tam = np.array(comp.sizes())
    gig = comp.giant()
    srcs = rng.choice(gig.vcount(), min(n_origens, gig.vcount()), replace=False)
    soma, pares, ecc = 0.0, 0, 0
    for s in srcs:
        d = np.array(gig.distances(source=int(s))[0])
        d = d[d > 0]
        if len(d):
            soma += d.sum()
            pares += len(d)
            ecc = max(ecc, int(d.max()))
    return dict(A_maior_comp=int(tam.max()), S=tam.max() / n_orig,
                S_restante=tam.max() / g.vcount(), B_componentes=len(tam),
                C_L=soma / pares if pares else float("nan"), C_diametro_lb=ecc,
                D_isolados=float((np.array(g.degree()) == 0).mean()))


def remover(g, remover_idx):
    manter = np.ones(g.vcount(), dtype=bool)
    manter[list(remover_idx)] = False
    return g.induced_subgraph(np.nonzero(manter)[0].tolist())


def ordem_grau(grau, rng):
    """Vertices do maior para o menor grau; empates sorteados (semente fixa)."""
    return np.lexsort((rng.permutation(len(grau)), -np.asarray(grau)))


def ataque_adaptativo(adj, r):
    """Remove sempre o vertice de maior grau no grafo ja reduzido. Heap com
    atualizacao preguicosa; custo ~ soma dos graus dos vertices removidos."""
    grau = [len(a) for a in adj]
    heap = [(-d, v) for v, d in enumerate(grau)]
    heapq.heapify(heap)
    removido = bytearray(len(adj))
    saida = []
    while len(saida) < r:
        d, v = heapq.heappop(heap)
        if removido[v] or -d != grau[v]:
            continue
        removido[v] = 1
        saida.append(v)
        for w in adj[v]:
            if not removido[w]:
                grau[w] -= 1
                heapq.heappush(heap, (-grau[w], w))
    return saida


def main():
    rng = np.random.default_rng(SEED)
    g = carregar("undir_lcc")
    gd = carregar("dir_lcc")
    n = g.vcount()
    r = math.ceil(0.05 * n)
    print(f"n={n} r={r}", flush=True)

    base = medidas(g, rng, n, 50)
    print("base", base, flush=True)

    # ataques (uma execucao cada; a ordem e fixa, nao ha aleatoriedade alem do desempate)
    grau = g.degree()
    ataques = {
        "grau (estático)": ordem_grau(grau, rng)[:r],
        "grau (adaptativo)": np.array(ataque_adaptativo(g.get_adjlist(), r)),
        "PageRank (estático)": np.argsort(-np.array(gd.pagerank()), kind="stable")[:r],
    }
    linhas = [dict(cenario="sem remoção", **base)]
    for nome, idx in ataques.items():
        h = remover(g, idx)
        mt = medidas(h, rng, n, 50)
        linhas.append(dict(cenario=nome, **mt))
        print(nome, mt, flush=True)
        del h
    pd.DataFrame(linhas).to_csv(f"{RES}/parte3_robustez_ataques.csv", index=False)

    # remocao aleatoria
    rep = []
    for i in range(T):
        idx = rng.choice(n, r, replace=False)
        h = remover(g, idx)
        mt = medidas(h, rng, n, ORIGENS)
        rep.append(dict(rep=i, **mt))
        print("aleatória", i, mt, flush=True)
        del h
    dfa = pd.DataFrame(rep)
    dfa.to_csv(f"{RES}/parte3_robustez_aleatoria.csv", index=False)
    resumo(dfa, pd.DataFrame(linhas))


def resumo(dfa, dfat):
    cols = [("S", "S (maior comp./n)"), ("B_componentes", "B (nº de comp.)"),
            ("C_L", "C (caminho médio)"), ("D_isolados", "D (fração isolados)")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (c, titulo) in zip(axs, cols):
        ax.boxplot(dfa[c].values, widths=0.5)
        for (_, linha), marc in zip(dfat.iloc[1:].iterrows(), ["v", "s", "D"]):
            ax.plot(1.35, linha[c], marc, label=linha["cenario"])
        ax.axhline(dfat[c].iloc[0], color="gray", ls=":", lw=1)
        ax.set_title(titulo, fontsize=9)
        ax.set_xticks([1])
        ax.set_xticklabels(["aleatória"])
        ax.set_xlim(0.6, 1.6)
    axs[0].legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{FIG}/parte3_robustez.png", dpi=200)
    plt.close()
    tab = dfa[[c for c, _ in cols] + ["A_maior_comp", "C_diametro_lb"]].agg(["mean", "std"]).T
    tab.to_csv(f"{RES}/parte3_robustez_aleatoria_resumo.csv")
    print(tab.to_string())


if __name__ == "__main__":
    if "--resumo" in sys.argv:
        resumo(pd.read_csv(f"{RES}/parte3_robustez_aleatoria.csv"),
               pd.read_csv(f"{RES}/parte3_robustez_ataques.csv"))
    else:
        main()
