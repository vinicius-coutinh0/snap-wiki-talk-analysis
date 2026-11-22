"""Tratamento do grafo wiki-Talk (G85).

Passos:
1. Le a lista de arestas (ignora linhas com '#').
2. Remove auto-loops e arestas duplicadas.
3. Gera a versao nao dirigida (arestas u-v unicas).
4. Extrai a maior componente (fraca na dirigida, conexa na nao dirigida).
5. Salva tudo em data/processed e um resumo em results/tratamento.csv.
"""
import os
import numpy as np
import pandas as pd
import igraph as ig

RAW = "data/raw/WikiTalk.txt"
PROC = "data/processed"
RES = "results"


def carregar_bruto(path=RAW):
    df = pd.read_csv(path, sep="\t", comment="#", header=None,
                     names=["u", "v"], dtype=np.int64)
    return df["u"].to_numpy(), df["v"].to_numpy()


def reindexar(u, v):
    """Os ids originais tem buracos; mapeia para 0..n-1."""
    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    m = len(u)
    return ids, inv[:m], inv[m:]


def unicas(u, v, n):
    chave = np.unique(u.astype(np.int64) * n + v)
    return chave // n, chave % n


def tratar():
    u, v = carregar_bruto()
    info = {"arestas_brutas": len(u)}

    ids, u, v = reindexar(u, v)
    n = len(ids)
    info["vertices_brutos"] = n

    mask = u != v
    info["auto_loops_removidos"] = int((~mask).sum())
    u, v = u[mask], v[mask]

    u, v = unicas(u, v, n)
    info["arestas_dirigidas"] = len(u)
    info["duplicadas_removidas"] = info["arestas_brutas"] - info["auto_loops_removidos"] - len(u)

    # nao dirigida: par (min, max) unico
    a, b = np.minimum(u, v), np.maximum(u, v)
    a, b = unicas(a, b, n)
    info["arestas_nao_dirigidas"] = len(a)
    info["pares_reciprocos"] = info["arestas_dirigidas"] - len(a)

    gd = ig.Graph(n=n, edges=np.column_stack([u, v]).tolist(), directed=True)
    gu = ig.Graph(n=n, edges=np.column_stack([a, b]).tolist(), directed=False)

    # maior componente (fraca na dirigida == conexa na nao dirigida)
    comp = gu.connected_components()
    info["componentes"] = len(comp)
    maior = np.array(max(comp, key=len))
    info["vertices_maior_componente"] = len(maior)
    info["arestas_maior_componente_dir"] = gd.induced_subgraph(maior).ecount()

    gd_lcc = gd.induced_subgraph(maior)
    gu_lcc = gu.induced_subgraph(maior)
    info["arestas_maior_componente_nao_dir"] = gu_lcc.ecount()

    os.makedirs(PROC, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    for nome, g in [("dir", gd), ("undir", gu), ("dir_lcc", gd_lcc), ("undir_lcc", gu_lcc)]:
        g.write_picklez(f"{PROC}/wikitalk_{nome}.pkl.gz")
    np.save(f"{PROC}/ids_originais.npy", ids)

    pd.Series(info).to_csv(f"{RES}/tratamento.csv", header=["valor"], index_label="item")
    print(pd.Series(info).to_string())


def carregar(nome):
    """nome: dir, undir, dir_lcc ou undir_lcc."""
    return ig.Graph.Read_Picklez(f"{PROC}/wikitalk_{nome}.pkl.gz")


if __name__ == "__main__":
    tratar()
