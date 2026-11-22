"""Parte III, extra: wiki-Talk contra modelos classicos com o mesmo n e grau medio.

Erdos-Renyi G(n, m), Barabasi-Albert (m = 2) e Watts-Strogatz (anel com 4
vizinhos, p = 0,1), todos com n da maior componente do wiki-Talk. Compara C, L,
distribuicao de graus e robustez (5 remocoes aleatorias e ataque estatico por
grau, de 5% dos vertices). Aqui o igraph e usado para gerar e medir.
"""
import math
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tratamento import carregar
from parte3_robustez import medidas, remover, ordem_grau

SEED = 42
REPS = 5
ORIGENS = 10
RES, FIG = "results", "results/figures"


def gerar(nome, n, m, seed):
    import random
    random.seed(seed)
    if nome == "Erdős-Rényi":
        return ig.Graph.Erdos_Renyi(n=n, m=m)
    if nome == "Barabási-Albert (m=2)":
        return ig.Graph.Barabasi(n=n, m=2, directed=False)
    return ig.Graph.Watts_Strogatz(dim=1, size=n, nei=2, p=0.1)


def analisar(nome, g, rng, n_ref):
    g.simplify()
    comp = g.connected_components()
    gig = comp.giant()
    grau = np.array(g.degree())
    base = medidas(g, rng, g.vcount(), ORIGENS)
    linha = dict(grafo=nome, n=g.vcount(), m=g.ecount(), grau_medio=grau.mean(),
                 grau_max=int(grau.max()), grau_desvio=grau.std(),
                 C_local=g.transitivity_avglocal_undirected(mode="zero"),
                 transitividade=g.transitivity_undirected(),
                 L=base["C_L"], diametro_lb=base["C_diametro_lb"],
                 fracao_gigante=gig.vcount() / g.vcount())
    r = math.ceil(0.05 * g.vcount())
    rnd = [medidas(remover(g, rng.choice(g.vcount(), r, replace=False)), rng, g.vcount(), ORIGENS)
           for _ in range(REPS)]
    linha["S_aleatoria"] = float(np.mean([x["S"] for x in rnd]))
    linha["L_aleatoria"] = float(np.mean([x["C_L"] for x in rnd]))
    atk = medidas(remover(g, ordem_grau(grau, rng)[:r]), rng, g.vcount(), ORIGENS)
    linha["S_ataque_grau"] = atk["S"]
    linha["B_ataque_grau"] = atk["B_componentes"]
    linha["L_ataque_grau"] = atk["C_L"]
    linha["D_ataque_grau"] = atk["D_isolados"]
    print(linha, flush=True)
    return linha, grau


def main():
    rng = np.random.default_rng(SEED)
    wt = carregar("undir_lcc")
    n, m = wt.vcount(), wt.ecount()
    linhas, graus = [], {}
    linha, grau = analisar("wiki-Talk", wt, rng, n)
    linhas.append(linha)
    graus["wiki-Talk"] = grau
    for i, nome in enumerate(["Erdős-Rényi", "Barabási-Albert (m=2)", "Watts-Strogatz (k=4, p=0,1)"]):
        g = gerar(nome, n, m, SEED + i)
        linha, grau = analisar(nome, g, rng, n)
        linhas.append(linha)
        graus[nome] = grau
        pd.DataFrame(linhas).to_csv(f"{RES}/parte3_modelos.csv", index=False)
        del g
    # modelo de configuracao: mesma sequencia de graus do wiki-Talk (gerado por ultimo,
    # para nao alterar a ordem dos sorteios dos modelos anteriores)
    import random
    random.seed(SEED + 9)
    # metodo "configuration" aceita lacos e arestas duplicadas, que analisar() remove com
    # simplify(); o metodo que exige grafo simples nao termina com um hub de grau 100 mil
    gc = ig.Graph.Degree_Sequence(wt.degree(), method="configuration")
    linha, _ = analisar("Configuração (mesmos graus)", gc, rng, n)
    linhas.append(linha)
    pd.DataFrame(linhas).to_csv(f"{RES}/parte3_modelos.csv", index=False)
    del gc
    plt.figure(figsize=(5.5, 3.8))
    for nome, gr in graus.items():
        k, cont = np.unique(gr[gr > 0], return_counts=True)
        y = 1 - (np.cumsum(cont) - cont) / cont.sum()   # P(K >= k) em cada grau distinto
        plt.loglog(k, y, "o-", ms=2, lw=1, label=nome)
    plt.xlabel("grau k")
    plt.ylabel("P(K ≥ k)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG}/parte3_modelos_graus.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
