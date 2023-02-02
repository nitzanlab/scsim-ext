# scsim-ext
We extend the single-cell RNA-sequencing simulation framework, [scsim](https://github.com/dylkot/scsim), with additional functionalities of:
(1) enhancing RNA dropout effect, and (2) adding ambient RNA to drops.

## scsim
Simulate single-cell RNA-seq data using the [Splatter](https://github.com/Oshlack/splatter) statistical framework, which is described [here](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0) but implemented in python. In addition, simulates doublets and cells with shared gene-expression programs (I.e. activity programs). This was used to benchmark methods for gene expression program inference in single-cell RNA-seq data as described [here](https://elifesciences.org/articles/43803)

## Enhancing RNA dropout
We implement the dropout step described in [Splatter](https://github.com/Oshlack/splatter) by fitting a sigmoid curve through genes’ log mean count and their cell
fraction with zero reads, where the sigmoid is characterized by shape and midpoint parameters.
To enhance dropout, the sigmoid was shifted (decrease its midpoint) and the adjusted dropout probability was computed. This probability was then used to binomially sample the observed counts. 

## Ambient RNA
We lend [CellBender](https://www.biorxiv.org/content/10.1101/791699v1)'s model of sample RNA contamination (or ambient RNA). That is, we combine each drop's RNA with sampled RNA contamination and sample its RNA counts downstream.

Specifically, the mean expression of gene $g$ and cell $n$ (or here, the cell in drop $n$) was obtained
from scsim as $\lambda_{ng}$. For CellBender’s probabilistic model, the probability of having a cell in the
drop $y_n$ is set to 1 (because empty drops are not considered), and $\rho_n$
is set to 0, because zero
reads are exogenous to the drop as we account only for ambient RNA (pumped into the drop)
and not for barcode swapping. Thus, the CellBender model simplifies to:
$$C_{ng}\sim NB({d_n}^{cell}\chi_{ng} + {d_n}^{drop}{\chi_g}^a,\phi)$$

The two models were combined as follows:
$\lambda_{ng} = {d_n}^{cell} \chi_{ng}$ is the mean expression (similar to CellBender, scsim also uses a log-normal distribution of cell size)
$\lambda_g^{-}=avg_n \lambda_{ng}=\chi_g^a$
is the mean true count of gene 𝑔, that is, the ambient contribution of the
gene.
Given the fraction of ambient RNA, $f^{drop}$, and the library size, $d_{\mu}^{cell}$, and scale, $d_{\sigma}^{cell}$, used for sampling the cell size, $d_{n}^{cell}$
, $d_{n}^{cell}\sim LogNormal(d_{\mu}^{cell} + log(f^{drop}), d_{\sigma}^{cell})$ is sampled (that
is, the same variance is employed as used for sampling the cell content and the mean is set to $log(e^{d_{\mu}^{cell}}f^{drop})$.
$\phi$ is sampled as defined in CellBender
Thus, we sample:
$C_ng \sim NB(\lambda_{ng}+d_n^{drop}\lambda^{-}_g, \phi)$

For comparability, the reference is generated in the same way (using negative binomial sampling instead of scsim’s Poisson sampling), but without adding ambient RNA.
<!-- run_scsim.py has example code for running a simulation with a given set of parameters. It saves the results in the numpy compressed matrix format which can be loaded into a Pandas dataframe as follows:

    with np.load(filename) as f:
        result = pd.DataFrame(**f) -->



