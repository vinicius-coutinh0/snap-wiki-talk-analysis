"""Parte II: algoritmos implementados em Python puro e medidos em subgrafos crescentes.

Uso: python src/parte2.py          (completo)
     RAPIDO=1 python src/parte2.py (teste rapido)
"""
import gc
import heapq
import math
import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tratamento import carregar
import algoritmos as A

SEED = 42
RAPIDO = bool(os.environ.get("RAPIDO"))
TAMANHOS = [500, 2000] if RAPIDO else [1000, 5000, 20000, 100000]
TAM_FW = [30, 60] if RAPIDO else [50, 100, 200, 300]
REPS = 5 if RAPIDO else 30
MAX_VIZ = 50       # limite de vizinhos por vertice na amostragem por BFS
PESO_MAX = 10
RES, FIG = "results", "results/figures"
rng = np.random.default_rng(SEED)

# valores da maior componente do grafo completo (para extrapolar)
N_FULL, M_U_FULL, M_D_FULL = 2388953, 4656682, 5018445


def amostra_bfs(g, inicio, k):
    """BFS a partir de `inicio` (o hub) ate k vertices. Cada vertice expande no
    maximo MAX_VIZ vizinhos sorteados; sem esse limite a amostra de 100k
    vertices seria so o hub e seus vizinhos (uma estrela)."""
    visto = {inicio}
    fila = [inicio]
    i = 0
    while i < len(fila) and len(visto) < k:
        u = fila[i]
        i += 1
        nb = g.neighbors(u)
        if len(nb) > MAX_VIZ:
            nb = rng.choice(nb, MAX_VIZ, replace=False).tolist()
        for v in nb:
            if v not in visto:
                visto.add(v)
                fila.append(v)
                if len(visto) >= k:
                    break
    return sorted(visto)


def montar(gu, gd, vert):
    """Estruturas das duas versoes da amostra; pesos inteiros 1..PESO_MAX fixos."""
    su, sd = gu.induced_subgraph(vert), gd.induced_subgraph(vert)
    n = su.vcount()
    arestas = []
    adj = [[] for _ in range(n)]
    wadj = [[] for _ in range(n)]
    arcos = []
    pesos = rng.integers(1, PESO_MAX + 1, su.ecount())
    for (u, v), w in zip(su.get_edgelist(), pesos.tolist()):
        arestas.append((u, v, w))
        adj[u].append(v)
        adj[v].append(u)
        wadj[u].append((v, w))
        wadj[v].append((u, w))
        arcos.append((u, v, w))
        arcos.append((v, u, w))
    adj_d = [[] for _ in range(n)]
    for u, v in sd.get_edgelist():
        adj_d[u].append(v)
    return dict(n=n, m=su.ecount(), md=sd.ecount(), su=su, sd=sd, adj=adj, wadj=wadj,
                arestas=arestas, arcos=arcos, adj_d=adj_d, pesos=pesos.tolist())


def validar(S):
    """Confere as implementacoes contra o igraph (so para validar, nao para medir)."""
    n, su, sd = S["n"], S["su"], S["sd"]
    s = int(rng.integers(n))
    ref = su.distances(source=s)[0]
    ok = {}
    ok["BFS"] = A.bfs(S["adj"], s) == [int(x) for x in ref]
    ok["DFS"] = len(A.dfs(S["adj"], s)) == n
    refw = su.distances(source=s, weights=S["pesos"])[0]
    dj = A.dijkstra(S["wadj"], s)
    ok["Dijkstra"] = [float(x) for x in dj] == [float(x) for x in refw]
    bf, _, neg = A.bellman_ford(n, S["arcos"], s)
    ok["Bellman-Ford"] = (not neg) and bf == dj
    # com peso 1, Dijkstra e Bellman-Ford coincidem com a BFS
    unit = [[(v, 1) for v in vs] for vs in S["adj"]]
    ok["Dijkstra (peso 1) = BFS"] = [int(x) for x in A.dijkstra(unit, s)] == [int(x) for x in ref]
    bf1, _, _ = A.bellman_ford(n, [(u, v, 1) for u, v, _ in S["arcos"]], s)
    ok["Bellman-Ford (peso 1) = BFS"] = [int(x) for x in bf1] == [int(x) for x in ref]
    comp, nc = A.tarjan_scc(S["adj_d"])
    ok["Tarjan"] = nc == len(sd.connected_components(mode="strong"))
    pk, uk = A.kruskal(n, S["arestas"])
    pp, up = A.prim(S["wadj"], 0)
    su.es["w"] = S["pesos"]
    pref = sum(su.spanning_tree(weights="w").es["w"])
    ok["Kruskal"] = pk == pref and uk == n - 1
    ok["Prim"] = pp == pref and up == n - 1
    pi, ui = A.prim_ingenuo(S["wadj"], 0)
    ok["Prim (ingenuo)"] = pi == pref and ui == n - 1
    impares = sum(len(a) % 2 for a in S["adj"])  # amostra e conexa
    esperado = "circuito" if impares == 0 else ("trilha" if impares == 2 else "nao")
    ok["Euleriano"] = A.euleriano_nao_dirigido(S["adj"]) == esperado
    return ok


def validar_fw(S):
    n = S["n"]
    mat = [[A.INF] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 0
    for u, v, w in S["arestas"]:
        mat[u][v] = mat[v][u] = min(mat[u][v], w)
    d = A.floyd_warshall([r[:] for r in mat])
    srcs = rng.choice(n, min(5, n), replace=False)
    return all(d[int(s)] == A.dijkstra(S["wadj"], int(s)) for s in srcs)


def medir(f, reps, args_list=None):
    """Tempo (s) de `reps` execucoes. args_list: um argumento por repeticao."""
    tempos = []
    gc.disable()
    for i in range(reps):
        a = args_list[i] if args_list is not None else ()
        t0 = time.perf_counter()
        f(*a) if isinstance(a, tuple) else f(a)
        tempos.append(time.perf_counter() - t0)
    gc.enable()
    return np.array(tempos)


def pushes(fn, wadj, s):
    """Conta as insercoes no heap de uma execucao (fora da medicao de tempo)."""
    c = [0]

    def p(h, x):
        c[0] += 1
        heapq.heappush(h, x)
    fn(wadj, s, p)
    return c[0]


def resumo(t):
    n = len(t)
    media, dp = t.mean(), t.std(ddof=1)
    crit = stats.norm.ppf(0.975) if n >= 30 else stats.t.ppf(0.975, n - 1)
    h = crit * dp / math.sqrt(n)
    return media, dp, media - h, media + h, "Z" if n >= 30 else "t"


def main():
    os.makedirs(FIG, exist_ok=True)
    gu, gd = carregar("undir_lcc"), carregar("dir_lcc")
    hub = int(np.argmax(gu.degree()))
    linhas, valid, brutos = [], [], []
    infos = {}

    def registrar(alg, S, t, extra=None):
        media, dp, lo, hi, dist = resumo(t)
        linhas.append(dict(algoritmo=alg, n=S["n"], m=S["m"], md=S["md"], reps=len(t), media_s=media,
                           desvio_s=dp, ic95_inf=lo, ic95_sup=hi, dist_ic=dist, **(extra or {})))
        brutos.extend(dict(algoritmo=alg, n=S["n"], rep=i, tempo_s=x) for i, x in enumerate(t))
        print(f"{alg:14s} n={S['n']:7d} media={media:.4f}s dp={dp:.4f}", flush=True)

    for k in TAMANHOS:
        vert = amostra_bfs(gu, hub, k)
        S = montar(gu, gd, vert)
        infos[k] = (S["n"], S["m"], S["md"])
        print(f"amostra {k}: n={S['n']} m={S['m']} md={S['md']} grau medio={2*S['m']/S['n']:.2f}", flush=True)
        for alg, ok in validar(S).items():
            valid.append(dict(n=S["n"], algoritmo=alg, confere=bool(ok)))
        src = [int(x) for x in rng.integers(0, S["n"], REPS)]
        registrar("BFS", S, medir(lambda s: A.bfs(S["adj"], s), REPS, src))
        registrar("DFS", S, medir(lambda s: A.dfs(S["adj"], s), REPS, src))
        registrar("Euleriano", S, medir(lambda: A.euleriano_nao_dirigido(S["adj"]), REPS, [()] * REPS))
        registrar("Dijkstra", S, medir(lambda s: A.dijkstra(S["wadj"], s), REPS, src),
                  dict(insercoes_heap=pushes(A.dijkstra, S["wadj"], src[0])))
        passadas = []
        def run_bf(s):
            passadas.append(A.bellman_ford(S["n"], S["arcos"], s)[1])
        registrar("Bellman-Ford", S, medir(run_bf, REPS, src),
                  dict(passadas_medias=float(np.mean(passadas))))
        registrar("Tarjan", S, medir(lambda: A.tarjan_scc(S["adj_d"]), REPS, [()] * REPS))
        registrar("Kruskal", S, medir(lambda: A.kruskal(S["n"], S["arestas"]), REPS, [()] * REPS))
        registrar("Prim (ingênuo)", S, medir(lambda s: A.prim_ingenuo(S["wadj"], s), REPS, src),
                  dict(insercoes_heap=pushes(A.prim_ingenuo, S["wadj"], src[0])))
        registrar("Prim", S, medir(lambda s: A.prim(S["wadj"], s), REPS, src),
                  dict(insercoes_heap=pushes(A.prim, S["wadj"], src[0])))

    for k in TAM_FW:
        S = montar(gu, gd, amostra_bfs(gu, hub, k))
        n = S["n"]
        base = [[A.INF] * n for _ in range(n)]
        for i in range(n):
            base[i][i] = 0
        for u, v, w in S["arestas"]:
            base[u][v] = base[v][u] = min(base[u][v], w)
        valid.append(dict(n=n, algoritmo="Floyd-Warshall", confere=validar_fw(S)))
        copias = [([r[:] for r in base],) for _ in range(REPS)]
        registrar("Floyd-Warshall", S, medir(A.floyd_warshall, REPS, copias))
        del copias

    df = pd.DataFrame(linhas)
    df.to_csv(f"{RES}/parte2_tempos.csv", index=False)
    pd.DataFrame(brutos).to_csv(f"{RES}/parte2_tempos_brutos.csv", index=False)
    pd.DataFrame(valid).to_csv(f"{RES}/parte2_validacao.csv", index=False)
    pd.DataFrame([dict(tamanho_pedido=k, n=v[0], m_nao_dir=v[1], m_dir=v[2],
                       grau_medio=2 * v[1] / v[0]) for k, v in infos.items()]
                 ).to_csv(f"{RES}/parte2_amostras.csv", index=False)
    analisar(df)


# funcao de custo teorica f(n, m, md) de cada algoritmo
# m = arestas nao dirigidas; md = arestas dirigidas (usado pelo Tarjan)
TEORIA = {
    "BFS": ("O(V+E)", lambda n, m, md: n + m),
    "DFS": ("O(V+E)", lambda n, m, md: n + m),
    "Euleriano": ("O(V+E)", lambda n, m, md: n + m),
    "Tarjan": ("O(V+E)", lambda n, m, md: n + md),
    "Dijkstra": ("O((V+E) log V)", lambda n, m, md: (n + m) * math.log2(n)),
    "Prim (ingênuo)": ("O((V+E) log V)", lambda n, m, md: (n + m) * math.log2(n)),
    "Prim": ("O((V+E) log V)", lambda n, m, md: (n + m) * math.log2(n)),
    "Kruskal": ("O(E log E)", lambda n, m, md: m * math.log2(max(m, 2))),
    "Bellman-Ford": ("O(VE)", lambda n, m, md: n * m),
    "Floyd-Warshall": ("O(V^3)", lambda n, m, md: n ** 3),
}


def analisar(df):
    """Figura, inclinacao contra o custo teorico e extrapolacao para o grafo completo.

    A inclinacao e a de log(tempo) contra log(f(n, m)): se a teoria vale, e 1.
    Isso evita misturar o efeito de n com o de m, ja que a densidade das amostras
    nao cresce de forma regular. Sao so 4 pontos por algoritmo.

    A figura mostra (tempo / f(n, m)) relativo ao menor subgrafo: uma reta em 1
    significa que a teoria vale; valores crescentes indicam custo por unidade
    maior nos grafos maiores. O Floyd-Warshall usa outros tamanhos e fica de fora."""
    algs = list(TEORIA)
    ext = []
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for alg in algs:
        d = df[df.algoritmo == alg].sort_values("n")
        n, m, md, y = d.n.values, d.m.values, d.md.values, d.media_s.values
        nome, f = TEORIA[alg]
        custo = np.array([f(a, b, c) for a, b, c in zip(n, m, md)])
        c = y[-1] / custo[-1]  # constante ajustada no maior tamanho
        incl = np.polyfit(np.log(custo), np.log(y), 1)[0]
        por_custo = y / custo
        razao = por_custo[-1] / por_custo[0]  # tempo por unidade de custo, maior / menor
        if alg not in ("Floyd-Warshall", "Bellman-Ford"):  # BF tem outra escala; ver texto
            ax.plot(n, por_custo / por_custo[0], "o-", ms=4, label=alg)
        linha = dict(algoritmo=alg, complexidade=nome, inclinacao_vs_custo=incl,
                     razao_tempo_por_custo_maior_menor=razao, pontos=len(n),
                     maior_n_medido=int(n[-1]), tempo_no_maior_s=y[-1])
        if alg == "Bellman-Ford":
            obs = d.passadas_medias.values * 2 * m  # passadas x arcos
            linha["inclinacao_vs_passadas_x_arcos"] = np.polyfit(np.log(obs), np.log(y), 1)[0]
            pred = y[-1] / obs[-1] * d.passadas_medias.values[-1] * 2 * M_U_FULL
        else:
            pred = c * f(N_FULL, M_U_FULL, M_D_FULL)
        linha["tempo_estimado_grafo_completo_s"] = pred
        ext.append(linha)
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vértices do subgrafo")
    ax.set_ylabel("(tempo / custo teórico) relativo ao menor subgrafo")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{FIG}/parte2_tempos.png", dpi=200)
    plt.close()
    pd.DataFrame(ext).to_csv(f"{RES}/parte2_extrapolacao.csv", index=False)
    print(pd.DataFrame(ext).to_string())


def completo():
    """Roda os algoritmos lineares e quase lineares de verdade na maior componente
    (2,4 milhoes de vertices), com poucas repeticoes (t de Student)."""
    reps, reps_bf = 10, 5
    r2 = np.random.default_rng(SEED + 1)
    gu, gd = carregar("undir_lcc"), carregar("dir_lcc")
    t0 = time.perf_counter()
    S = montar(gu, gd, list(range(gu.vcount())))
    print(f"estruturas montadas em {time.perf_counter() - t0:.0f}s: n={S['n']} m={S['m']} md={S['md']}", flush=True)
    n = S["n"]
    src = [int(x) for x in r2.integers(0, n, reps)]
    linhas, resp = [], {}

    def rodar(alg, f, args, r):
        t = medir(f, r, args)
        media, dp, lo, hi, dist = resumo(t)
        linhas.append(dict(algoritmo=alg, n=n, m=S["m"], md=S["md"], reps=r, media_s=media,
                           desvio_s=dp, ic95_inf=lo, ic95_sup=hi, dist_ic=dist))
        print(f"{alg:16s} media={media:.2f}s dp={dp:.2f}", flush=True)

    rodar("BFS", lambda s: A.bfs(S["adj"], s), src, reps)
    rodar("DFS", lambda s: A.dfs(S["adj"], s), src, reps)
    rodar("Euleriano", lambda: A.euleriano_nao_dirigido(S["adj"]), [()] * reps, reps)
    rodar("Dijkstra", lambda s: A.dijkstra(S["wadj"], s), src, reps)
    rodar("Tarjan", lambda: A.tarjan_scc(S["adj_d"]), [()] * reps, reps)
    rodar("Kruskal", lambda: A.kruskal(n, S["arestas"]), [()] * reps, reps)
    rodar("Prim (ingênuo)", lambda s: A.prim_ingenuo(S["wadj"], s), src, reps)
    rodar("Prim", lambda s: A.prim(S["wadj"], s), src, reps)
    pas = []
    rodar("Bellman-Ford", lambda s: pas.append(A.bellman_ford(n, S["arcos"], s)[1]), src, reps_bf)
    linhas[-1]["passadas_medias"] = float(np.mean(pas))
    linhas[-1]["passadas_max"] = int(max(pas))
    pd.DataFrame(linhas).to_csv(f"{RES}/parte2_completo.csv", index=False)

    # respostas reais e conferencia com o igraph
    comp, nc = A.tarjan_scc(S["adj_d"])
    tam = np.bincount(comp)
    impares = sum(len(a) % 2 for a in S["adj"])
    s = src[0]
    ref = [int(x) for x in gu.distances(source=s)[0]]
    gu.es["w"] = S["pesos"]
    pk, uk = A.kruskal(n, S["arestas"])
    resp = {
        "sccs_tarjan_python": nc,
        "sccs_igraph": len(gd.connected_components(mode="strong")),
        "maior_scc_tarjan": int(tam.max()),
        "maior_scc_igraph": max(gd.connected_components(mode="strong").sizes()),
        "vertices_grau_impar": impares,
        "eulerianidade": A.euleriano_nao_dirigido(S["adj"]),
        "euleriano_dirigido": A.euleriano_dirigido(S["adj_d"]),
        "bfs_confere_igraph": A.bfs(S["adj"], s) == ref,
        "dijkstra_confere_igraph": [float(x) for x in A.dijkstra(S["wadj"], s)]
        == [float(x) for x in gu.distances(source=s, weights="w")[0]],
        "kruskal_peso": pk,
        "kruskal_peso_igraph": int(sum(gu.spanning_tree(weights="w").es["w"])),
        "kruskal_arestas": uk,
        "bellman_ford_passadas_medias": float(np.mean(pas)),
    }
    pd.Series(resp).to_csv(f"{RES}/parte2_completo_respostas.csv", header=["valor"], index_label="item")
    print(pd.Series(resp).to_string())


if __name__ == "__main__":
    import sys
    if "--analise" in sys.argv:  # refaz so figura e extrapolacao a partir do CSV
        analisar(pd.read_csv(f"{RES}/parte2_tempos.csv"))
    elif "--completo" in sys.argv:
        completo()
    else:
        main()
