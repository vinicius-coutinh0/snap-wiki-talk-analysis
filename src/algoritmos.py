"""Algoritmos da Parte II, implementados em Python puro (sem igraph).

Representacao: lista de adjacencia `adj[u] = [v, ...]`, lista ponderada
`wadj[u] = [(v, w), ...]` e lista de arestas `(u, v, w)`.
"""
import heapq
from collections import deque

INF = float("inf")


def bfs(adj, s):
    """Distancias (em arestas) a partir de s. O(V + E)."""
    dist = [-1] * len(adj)
    dist[s] = 0
    fila = deque([s])
    while fila:
        u = fila.popleft()
        d = dist[u] + 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = d
                fila.append(v)
    return dist


def dfs(adj, s):
    """DFS iterativa; devolve a ordem de visita. O(V + E)."""
    visitado = [False] * len(adj)
    ordem = []
    visitado[s] = True
    pilha = [(s, iter(adj[s]))]
    ordem.append(s)
    while pilha:
        u, it = pilha[-1]
        for v in it:
            if not visitado[v]:
                visitado[v] = True
                ordem.append(v)
                pilha.append((v, iter(adj[v])))
                break
        else:
            pilha.pop()
    return ordem


def euleriano_nao_dirigido(adj):
    """'circuito' (todos os graus pares), 'trilha' (2 impares) ou 'nao'. O(V + E)."""
    n = len(adj)
    impares = 0
    inicio = -1
    for u in range(n):
        if adj[u]:
            if inicio < 0:
                inicio = u
            if len(adj[u]) % 2:
                impares += 1
    if inicio < 0:
        return "circuito"
    visto = bfs(adj, inicio)
    if any(visto[u] < 0 and adj[u] for u in range(n)):
        return "nao"
    if impares == 0:
        return "circuito"
    return "trilha" if impares == 2 else "nao"


def euleriano_dirigido(adj):
    """Circuito euleriano dirigido: in = out em todo vertice e conexo (fraco). O(V + E)."""
    n = len(adj)
    entrada = [0] * n
    for u in range(n):
        for v in adj[u]:
            entrada[v] += 1
    if any(len(adj[u]) != entrada[u] for u in range(n)):
        return False
    und = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            und[u].append(v)
            und[v].append(u)
    return euleriano_nao_dirigido(und) != "nao"


def dijkstra(wadj, s, push=heapq.heappush):
    """Heap binario. O((V + E) log V). Pesos nao negativos.
    `push` pode ser trocado por uma funcao que conta as insercoes no heap."""
    dist = [INF] * len(wadj)
    dist[s] = 0
    heap = [(0, s)]
    pop = heapq.heappop
    while heap:
        d, u = pop(heap)
        if d > dist[u]:
            continue
        for v, w in wadj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                push(heap, (nd, v))
    return dist


def bellman_ford(n, arcos, s):
    """Relaxa todos os arcos ate nao haver mudanca. Pior caso O(V E).
    Devolve (dist, passadas, ciclo_negativo)."""
    dist = [INF] * n
    dist[s] = 0
    passadas = 0
    for _ in range(n - 1):
        mudou = False
        passadas += 1
        for u, v, w in arcos:
            du = dist[u]
            if du + w < dist[v]:
                dist[v] = du + w
                mudou = True
        if not mudou:
            return dist, passadas, False
    for u, v, w in arcos:
        if dist[u] + w < dist[v]:
            return dist, passadas, True
    return dist, passadas, False


def floyd_warshall(mat):
    """Matriz de distancias entre todos os pares. O(V^3). Altera `mat` e a devolve."""
    n = len(mat)
    for k in range(n):
        rk = mat[k]
        for i in range(n):
            ri = mat[i]
            dik = ri[k]
            if dik == INF:
                continue
            for j in range(n):
                t = dik + rk[j]
                if t < ri[j]:
                    ri[j] = t
    return mat


def tarjan_scc(adj):
    """Tarjan iterativo (evita estouro de recursao). Devolve lista com o id da SCC. O(V + E)."""
    n = len(adj)
    indice = [-1] * n
    baixo = [0] * n
    na_pilha = [False] * n
    comp = [-1] * n
    pilha = []
    contador = 0
    ncomp = 0
    for raiz in range(n):
        if indice[raiz] >= 0:
            continue
        trabalho = [(raiz, iter(adj[raiz]))]
        indice[raiz] = baixo[raiz] = contador
        contador += 1
        pilha.append(raiz)
        na_pilha[raiz] = True
        while trabalho:
            u, it = trabalho[-1]
            avancou = False
            for v in it:
                if indice[v] < 0:
                    indice[v] = baixo[v] = contador
                    contador += 1
                    pilha.append(v)
                    na_pilha[v] = True
                    trabalho.append((v, iter(adj[v])))
                    avancou = True
                    break
                elif na_pilha[v] and indice[v] < baixo[u]:
                    baixo[u] = indice[v]
            if avancou:
                continue
            trabalho.pop()
            if trabalho:
                p = trabalho[-1][0]
                if baixo[u] < baixo[p]:
                    baixo[p] = baixo[u]
            if baixo[u] == indice[u]:
                while True:
                    w = pilha.pop()
                    na_pilha[w] = False
                    comp[w] = ncomp
                    if w == u:
                        break
                ncomp += 1
    return comp, ncomp


def kruskal(n, arestas):
    """Ordena as arestas e usa union-find. O(E log E). Devolve (peso, n_arestas_usadas)."""
    pai = list(range(n))

    def achar(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    peso = 0
    usadas = 0
    for u, v, w in sorted(arestas, key=lambda e: e[2]):
        ru, rv = achar(u), achar(v)
        if ru != rv:
            pai[ru] = rv
            peso += w
            usadas += 1
            if usadas == n - 1:
                break
    return peso, usadas


def prim_ingenuo(wadj, s, push=heapq.heappush):
    """Prim que insere no heap toda aresta que sai da arvore (ate ~2E insercoes)."""
    n = len(wadj)
    na_arvore = [False] * n
    heap = [(0, s)]
    pop = heapq.heappop
    peso = 0
    usadas = -1
    while heap:
        w, u = pop(heap)
        if na_arvore[u]:
            continue
        na_arvore[u] = True
        peso += w
        usadas += 1
        for v, wv in wadj[u]:
            if not na_arvore[v]:
                push(heap, (wv, v))
    return peso, usadas


def prim(wadj, s, push=heapq.heappush):
    """Prim com vetor chave[v]: so insere no heap quando a chave de v melhora,
    como o Dijkstra. O((V + E) log V). Devolve (peso, n_arestas_usadas)."""
    n = len(wadj)
    na_arvore = [False] * n
    chave = [INF] * n
    chave[s] = 0
    heap = [(0, s)]
    pop = heapq.heappop
    peso = 0
    usadas = -1
    while heap:
        w, u = pop(heap)
        if na_arvore[u]:
            continue
        na_arvore[u] = True
        peso += w
        usadas += 1
        for v, wv in wadj[u]:
            if not na_arvore[v] and wv < chave[v]:
                chave[v] = wv
                push(heap, (wv, v))
    return peso, usadas
