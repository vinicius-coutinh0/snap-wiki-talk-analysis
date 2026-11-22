"""Parte III, descoberta: a rede e de transmissao (poucos emissores alcancam quase todos).

Hipotese das duas camadas: so uma minoria escreve (grau de saida > 0) e essa
minoria alcanca quase toda a maior componente. Teste: remover os k vertices de
maior grau de saida (ou de entrada, ou k aleatorios) do grafo dirigido e medir
que fracao do grafo restante e alcancavel, em media, a partir de um escritor
sorteado entre os que restam (grau de saida > 0 no grafo reduzido).
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tratamento import carregar
from parte3_robustez import remover, ordem_grau

SEED = 42
N_FONTES = 50
REPS_ALEATORIO = 3
RES, FIG = "results", "results/figures"


def alcance(h, rng):
    """Fracao media do grafo alcancavel (caminho dirigido) a partir de escritores."""
    saida = np.array(h.outdegree())
    escritores = np.nonzero(saida > 0)[0]
    if len(escritores) == 0:
        return 0.0, 0.0, 0
    fontes = rng.choice(escritores, min(N_FONTES, len(escritores)), replace=False)
    fr = np.array([len(h.subcomponent(int(s), mode="out")) / h.vcount() for s in fontes])
    ep = fr.std(ddof=1) / np.sqrt(len(fr)) if len(fr) > 1 else 0.0
    return float(fr.mean()), float(ep), len(escritores)


def main():
    rng = np.random.default_rng(SEED)
    g = carregar("dir_lcc")
    n = g.vcount()
    r = math.ceil(0.05 * n)
    ks = [0, 10, 100, 1000, 10000, 50000, r]
    ord_out = ordem_grau(g.outdegree(), rng)
    ord_in = ordem_grau(g.indegree(), rng)
    print(f"n={n} escritores={int((np.array(g.outdegree()) > 0).sum())} r={r}", flush=True)
    linhas = []
    for k in ks:
        for nome, ordem in [("maior grau de saída", ord_out), ("maior grau de entrada", ord_in)]:
            h = remover(g, ordem[:k]) if k else g
            a, ep, nesc = alcance(h, rng)
            linhas.append(dict(remocao=nome, k=k, alcance_medio=a, erro_padrao=ep,
                               escritores_restantes=nesc, vertices_restantes=h.vcount()))
            print(linhas[-1], flush=True)
        vals = []
        for _ in range(REPS_ALEATORIO if k else 1):
            h = remover(g, rng.choice(n, k, replace=False)) if k else g
            a, ep, nesc = alcance(h, rng)
            vals.append((a, ep, nesc))
        # erro padrao entre as repeticoes (ou o da unica execucao quando k = 0)
        ep_rep = float(np.std([v[0] for v in vals], ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 \
            else vals[0][1]
        linhas.append(dict(remocao="aleatória", k=k, alcance_medio=float(np.mean([v[0] for v in vals])),
                           erro_padrao=ep_rep,
                           escritores_restantes=int(np.mean([v[2] for v in vals])),
                           vertices_restantes=n - k))
        print(linhas[-1], flush=True)
        pd.DataFrame(linhas).to_csv(f"{RES}/parte3_descoberta.csv", index=False)

    df = pd.DataFrame(linhas)
    plt.figure(figsize=(5.5, 3.8))
    for nome, d in df.groupby("remocao"):
        d = d[d.k > 0]
        plt.errorbar(d.k, d.alcance_medio, yerr=d.erro_padrao, fmt="o-", capsize=2, label=nome)
    plt.axhline(df[df.k == 0].alcance_medio.iloc[0], color="gray", ls=":", lw=1)
    plt.xscale("log")
    plt.xlabel("vértices removidos (k)")
    plt.ylabel("fração alcançável a partir de um escritor")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG}/parte3_descoberta.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
