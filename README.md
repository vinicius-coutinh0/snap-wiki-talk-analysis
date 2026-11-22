# Análise do grafo wiki-Talk (SNAP)

Trabalho prático de Teoria dos Grafos (UFBA), grafo G85 = **wiki-Talk**.
Aresta u → v: o usuário u editou a página de discussão de v ao menos uma vez.

## Como executar

```powershell
pip install -r requirements.txt
```

1. Baixe `wiki-Talk.txt.gz` em <https://snap.stanford.edu/data/wiki-Talk.html>, descompacte e salve como `data/raw/WikiTalk.txt` (o arquivo não é versionado).
2. Tratamento (gera `data/processed/` e `results/tratamento.csv`):
   ```powershell
   python src/tratamento.py
   ```
3. Parte I (gera `results/parte1_metricas.csv` e figuras em `results/figures/`):
   ```powershell
   python src/parte1.py
   ```

Os dois scripts rodam a partir da raiz do repositório. A semente aleatória é fixa (42).

## Estrutura

- `src/`: tratamento e métricas
- `data/processed/`: grafos tratados (formato pickle do igraph, `.pkl.gz`): `dir` e `undir` (completos, dirigido e não dirigido) e `dir_lcc` e `undir_lcc` (maior componente). `ids_originais.npy` guarda os ids originais de cada vértice.
- `results/`: tabelas CSV e figuras

Para carregar um grafo tratado sem rodar o tratamento:

```python
import igraph as ig
g = ig.Graph.Read_Picklez("data/processed/wikitalk_dir_lcc.pkl.gz")
```
