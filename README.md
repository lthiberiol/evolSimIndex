# Evolitionary Similarity Index
<!-- badges: start -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lthiberiol/evolSimIndex/blob/master/correlate_evolution.ipynb)
<!--badges: end -->

Assess evolutionary similarity between gene familie

## Overview
Balrog is a prokaryotic gene finder based on a Temporal Convolutional Network. We took a data-driven approach to prokaryotic gene finding, relying on the large and diverse collection of already-sequenced genomes. By training a single, universal model of bacterial genes on protein sequences from many different species, we were able to match the sensitivity of current gene finders while reducing the overall number of gene predictions. Balrog does not need to be refit on any new genome.

Preprint avaialable on bioRxiv [not yet, chill out!](https://www.biorxiv.org/).

## Getting started
Click the "Open in Colab" button above to get started. 

Press `⌘+F9` or `CTRL+F9` to run all cells, and fill the required parameters in the botton. Alternatively, press `SHIF`hold shift or ctrl and press enter to run cells.
Double click the top of a cell to inspect the code inside and change things. Double click the right side of the cell to hide the code.
Have fun!

Because Balrog uses a complex gene model and performs alignment-based search with mmseqs2, each genome takes ~10-15 minutes to process. Feel free to open a GitHub issue if you run into problems or would like a command line version of Balrog.