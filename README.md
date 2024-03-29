# Evolutionary Similarity Index
<!-- badges: start -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lthiberiol/evolSimIndex/blob/master/correlate_evolution.ipynb)
<!--badges: end -->

Assess evolutionary similarity between gene families

## Overview
Evolutionary Similarity Index (<img src="https://render.githubusercontent.com/render/math?math=I_{ES}">) is a metric for evolutionary similarity between gene families. It is the product of <img src="https://render.githubusercontent.com/render/math?math=I_{ES}=R^{2}*I_{BC}">

* <img src="https://render.githubusercontent.com/render/math?math=R^{2}">: is the coefficient of determination from an weighted Orthogonal Distance Regression

* <img src="https://render.githubusercontent.com/render/math?math=I_{BC}">: is <img src="https://render.githubusercontent.com/render/math?math=1-D_{BC}">, where <img src="https://render.githubusercontent.com/render/math?math=D_{BC}"> is the Bray-Curtis dissimilarity between two gene families.

<img src="https://render.githubusercontent.com/render/math?math=I_{ES}"> is less susceptible to phylogenetic noise than its tree-based counterparts, and more computationally efficient given that it bypasses tree reconstruction.


**Manuscript available on [doi.org/10.1093/gbe/evab187](https://doi.org/10.1093/gbe/evab187).**

---


## Getting started
Click the "Open in Colab" button above to get started. 

Press `⌘+F9` or `CTRL+F9` to run all cells, and fill the required parameters in the botton. Alternatively, press `SHIF+ENTER` to run each cell at a time.
Have fun!

## Docker

Available on [Docker Hub](https://hub.docker.com/r/thiberio/evolsimindex)

`docker pull thiberio/evolsimindex` to download container...

... once downloaded, run `docker run -p 8888:8888 thiberio/evolsimindex` and follow terminal instructions. Jupyter notebook will be available at `http://127.0.0.1:8888`
