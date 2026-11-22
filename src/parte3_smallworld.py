"""Parte III, Apendice A: small-world com codigo proprio.

C (clusterizacao) e L (caminho medio) sao calculados aqui, sem biblioteca de
grafos: triangulos por interseccao de conjuntos e BFS de algoritmos.py.
O igraph so e usado para montar a lista de adjacencia e conferir C.
A referencia aleatoria e um Erdos-Renyi G(n, m) gerado com numpy, com o mesmo
n e o mesmo numero de arestas, medido com o mesmo codigo.
"""
import math
import time
import numpy as np
import pandas as pd
from scipy import stats
from tratamento import carregar
import algoritmos as A

SEED = 42
N_ORIGENS = 100
RES = "results"


def triangulos_por_vertice(adj):
    """Triangulos de cada vertice. Orienta cada aresta do vertice de menor
    posicao (grau, id) para o de maior e intersecta os conjuntos de saida."""
    n = len(adj)
    ordem = sorted(range(n), key=lambda v: (len(adj[v]), v))
    pos = [0] * n
    for i, v in enumerate(ordem):
        pos[v] = i
    F = [{v for v in adj[u] if pos[v] > pos[u]} for u in range(n)]
    t = [0] * n
    for u in range(n):
        Fu = F[u]
        for v in Fu:
            comum = Fu & F[v]
            if comum:
                k = len(comum)
                t[u] += k
                t[v] += k
                for w in comum:
                    t[w] += 1
    return np.array(t, dtype=np.int64)


def clusterizacao(adj):
    """Devolve (C medio local com C_v = 0 se grau < 2, transitividade, triangulos)."""
    t = triangulos_por_vertice(adj)
    d = np.array([len(a) for a in adj], dtype=np.float64)
    pares = d * (d - 1) / 2
    cv = np.zeros(len(adj))
    ok = d >= 2
    cv[ok] = t[ok] / pares[ok]
    return cv.mean(), t.sum() / pares.sum(), int(t.sum() // 3)


def caminho_medio(adj, origens):
    """Media das distancias de cada origem a todos os alcancaveis (BFS propria).
    Devolve (L, erro_padrao, diametro_lb, fracao_alcancada)."""
    medias, ecc, alc = [], 0, []
    n = len(adj)
    for s in origens:
        d = np.array(A.bfs(adj, int(s)))
        d = d[d > 0]
        medias.append(d.mean())
        ecc = max(ecc, int(d.max()))
        alc.append(len(d) / (n - 1))
    medias = np.array(medias)
    return medias.mean(), medias.std(ddof=1) / math.sqrt(len(medias)), ecc, float(np.mean(alc))


def erdos_renyi(n, m, rng):
    """G(n, m) aproximado: sorteia pares, remove loops e repetidos, fica com m."""
    u = rng.integers(0, n, int(m * 1.03))
    v = rng.integers(0, n, int(m * 1.03))
    k = u != v
    a, b = np.minimum(u[k], v[k]), np.maximum(u[k], v[k])
    chave = np.unique(a * n + b)
    chave = rng.permutation(chave)[:m]
    a, b = chave // n, chave % n
    adj = [[] for _ in range(n)]
    for x, y in zip(a.tolist(), b.tolist()):
        adj[x].append(y)
        adj[y].append(x)
    return adj


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    gu = carregar("undir_lcc")
    n, m = gu.vcount(), gu.ecount()
    adj = gu.get_adjlist()
    k_medio = 2 * m / n
    print(f"wiki-Talk (maior comp.): n={n} m={m} <k>={k_medio:.3f}", flush=True)

    C, T, tri = clusterizacao(adj)
    print(f"C local={C:.5f} transitividade={T:.5f} triangulos={tri} ({time.perf_counter()-t0:.0f}s)", flush=True)
    # conferencia (so aqui o igraph entra no calculo de C)
    ref = gu.transitivity_avglocal_undirected(mode="zero")
    print(f"conferencia igraph C local={ref:.5f}", flush=True)
    L, ep, diam, alc = caminho_medio(adj, rng.choice(n, N_ORIGENS, replace=False))
    print(f"L={L:.3f} +- {ep:.3f} diametro>={diam} ({time.perf_counter()-t0:.0f}s)", flush=True)
    linhas = [dict(grafo="wiki-Talk (maior comp.)", n=n, m=m, grau_medio=k_medio, C_local=C,
                   transitividade=T, triangulos=tri, L=L, L_ep=ep, diametro_lb=diam,
                   fracao_alcancada=alc, C_local_igraph_conferencia=ref)]
    del adj

    # referencia aleatoria: mesmo n e mesmo m
    adj_r = erdos_renyi(n, m, rng)
    mr = sum(len(a) for a in adj_r) // 2
    Cr, Tr, trir = clusterizacao(adj_r)
    visto = np.array(A.bfs(adj_r, int(rng.integers(n))))
    gigante = np.nonzero(visto >= 0)[0]
    Lr, epr, diamr, alcr = caminho_medio(adj_r, rng.choice(gigante, N_ORIGENS, replace=False))
    print(f"ER: m={mr} C={Cr:.2e} T={Tr:.2e} L={Lr:.3f} diametro>={diamr} gigante={len(gigante)/n:.3f}", flush=True)
    linhas.append(dict(grafo="Erdos-Renyi G(n,m) empirico", n=n, m=mr, grau_medio=2 * mr / n,
                       C_local=Cr, transitividade=Tr, triangulos=trir, L=Lr, L_ep=epr,
                       diametro_lb=diamr, fracao_alcancada=alcr))
    # valores analiticos
    linhas.append(dict(grafo="Erdos-Renyi analitico", n=n, m=m, grau_medio=k_medio,
                       C_local=k_medio / n, transitividade=k_medio / n,
                       L=math.log(n) / math.log(k_medio)))
    df = pd.DataFrame(linhas)
    df.to_csv(f"{RES}/parte3_smallworld.csv", index=False)

    # sigma = (C/C_rand) / (L/L_rand), com as duas definicoes de C e as duas referencias
    sig = []
    for ref_nome, i in [("ER empirico", 1), ("ER analitico", 2)]:
        for cnome, col in [("C local medio", "C_local"), ("transitividade", "transitividade")]:
            c_ratio = linhas[0][col] / linhas[i][col]
            l_ratio = linhas[0]["L"] / linhas[i]["L"]
            sig.append(dict(referencia=ref_nome, C_usado=cnome, C_sobre_Crand=c_ratio,
                            L_sobre_Lrand=l_ratio, sigma=c_ratio / l_ratio))
    sig = pd.DataFrame(sig)
    sig.to_csv(f"{RES}/parte3_smallworld_sigma.csv", index=False)
    print(df.to_string())
    print(sig.to_string())


if __name__ == "__main__":
    main()
