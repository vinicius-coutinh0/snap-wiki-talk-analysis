"""Parte III, Apendice B: a distribuicao de graus segue uma lei de potencia?

Ajuste com o pacote `powerlaw` (x_min e gamma por maxima verossimilhanca e KS)
e comparacao da lei de potencia com lognormal, exponencial e lei de potencia
truncada por razao de verossimilhanca. Graus de entrada, saida e total, so
vertices com grau positivo.
"""
import time
import warnings
import numpy as np
import pandas as pd
import powerlaw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tratamento import carregar

SEED = 42
RES, FIG = "results", "results/figures"
XMIN_MAX = 300   # o x_min e procurado entre 1 e este valor (acelera o ajuste)
CONCORRENTES = ["lognormal", "exponential", "truncated_power_law"]


def ajustar(nome, graus):
    graus = np.asarray(graus)
    graus = graus[graus > 0]
    t0 = time.perf_counter()
    fit = powerlaw.Fit(graus, discrete=True, xmin=(1, XMIN_MAX), verbose=False)
    linha = dict(grau=nome, n_positivos=len(graus), gamma=fit.alpha, gamma_erro=fit.sigma,
                 xmin=fit.xmin, n_cauda=int((graus >= fit.xmin).sum()),
                 fracao_cauda=float((graus >= fit.xmin).mean()), ks_D=fit.D)
    for alt in CONCORRENTES:
        R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
        linha[f"R_vs_{alt}"] = R   # R > 0: lei de potencia melhor
        linha[f"p_vs_{alt}"] = p
    linha["segundos"] = time.perf_counter() - t0
    print(linha, flush=True)
    return fit, linha


def main():
    gd = carregar("dir_lcc")
    series = {"entrada": gd.indegree(), "saída": gd.outdegree(),
              "total": gd.as_undirected(mode="collapse").degree()}
    fits, linhas = {}, []
    for nome, graus in series.items():
        fits[nome], linha = ajustar(nome, graus)
        linhas.append(linha)
        pd.DataFrame(linhas).to_csv(f"{RES}/parte3_potencia.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (nome, fit) in zip(axs, fits.items()):
        fit.plot_ccdf(ax=ax, color="k", linewidth=1, label="dados")
        fit.power_law.plot_ccdf(ax=ax, color="tab:red", linestyle="--", label="lei de potência")
        fit.lognormal.plot_ccdf(ax=ax, color="tab:blue", linestyle=":", label="lognormal")
        fit.truncated_power_law.plot_ccdf(ax=ax, color="tab:green", linestyle="-.", label="potência truncada")
        ax.set_title(f"Grau de {nome}" if nome != "total" else "Grau total")
        ax.set_xlabel("grau k")
        ax.set_ylabel("P(K ≥ k)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG}/parte3_potencia.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
