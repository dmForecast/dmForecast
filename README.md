# dmForecast
[![arXiv](https://img.shields.io/badge/arXiv-2112.05166-<COLOR>.svg)](https://arxiv.org/abs/2112.05166)



dmForecast is a tool for forecasting uncertainties on properties of dwarf galaxy Dark Matter halos (e.g., shape of the density profile or enclosed mass) that is attainable through jeans modeling.

Using Fisher matrices, which are constructed from the derivatives of the likelihood function used in traditional MCMC mass modeling methods, we can predict how well we will be able to constraint the properties of systems we're interested in.

With this this method we can quickly (compared to full analysis on mock data for example) how well we can constrain properties (e.g. the inner-slope of a dark matter halo) based on how many stars we observe, the precision of our measurements, as well as explore degeneracies that arise from using spherical jeans modeling. 

Example usage provided in example.ipynb

With this tool you can explore how errors scale with the number of stars you observe, and the accuracy of your observations:

<img src="figures/sigma_vals_final.png" alt="scaling" width=75%/>

Explore where errors are minimized:

<img src="figures/mass_errors.png" alt="Errors on Mass" width=75%/>

Study degeneracies and the information needed to break them:

<img src="figures/contours_final.png" alt="confidence regions" width=75%/>

Although the code in this github can be used to recreate the above plots, optimizations that i made throughout the project to speed up integration have not yet been made implemented. This will involve using numba which in my experience has not worked well with classes. In the future i'll also switch to an auto-differentiator. The only reason i didn't do that here is because i couldn't figure out how to apply an auto-differentiator to the Hypergeometric function that I use to calculate the enclosed mass.

# Attribution
[![DOI](https://zenodo.org/badge/414746751.svg)](https://zenodo.org/badge/latestdoi/414746751)
