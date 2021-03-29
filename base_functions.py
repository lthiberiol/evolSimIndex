#!/usr/bin/env python
# coding: utf-8

# # Evolutionary Similarity Index
# 
# Set of functions to assess evolutionary similarity between gene families using $I_{ES}$.
# 


#
# Import all dependencies
#
from scipy.odr              import Model, Data, RealData, ODR
from scipy.stats            import linregress
from scipy.optimize         import curve_fit
from scipy.spatial.distance import squareform, braycurtis
from matplotlib             import pyplot as plt
from sklearn.linear_model   import HuberRegressor
from copy                   import deepcopy
from collections            import Counter
from scipy.stats            import pearsonr

import igraph     as ig
import numpy      as np
import seaborn    as sns
import pandas     as pd

import sys
import random
import os
import subprocess
import re
import ete3
import multiprocessing
import itertools

class correlate_evolution:

    def __init__(self,
                 gene_ids        =False,
                 parse_leaf      =re.compile('^(GC[AF]_\d+(?:\.\d)?)[_|](.*)$'),
                 min_taxa_overlap=5):
        self.gene_ids        =gene_ids
        self.parse_leaf      =parse_leaf
        self.min_taxa_overlap=min_taxa_overlap

# #### Base linear model
# 
# Linear function with no intercept, forcing it to go through zero.
# 
    def line(self, x, slope):
        """Basic linear regression model
        This is the function provided to scipy's wODR (https://docs.scipy.org/doc/scipy/reference/odr.html)
        """
        return (slope * x) + 0


# #### Basic wODR function
# 
# function receiving `X` and `Y` variables, as well as their estimated weights.
# 
    def run_odr(self, x, y, x_weights, y_weights):
        """"receives pairwise distance matrices and wODR weights

        :parameter X: pairwise distance matrix from gene1
        :parameter Y: pairwise distance matrix from gene2
        :parameter x_weights: wODR weights for gene1 distances
        :parameter y_weights: wODR weights for gene2 distances

        :return ODR object (https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.html)
        """
        mod  = Model(self.line)
        data = Data(x,
                   y,
                   wd=x_weights,
                   we=y_weights
        )
        odr  = ODR(data,
                   mod,
                   beta0=[np.std(y)/np.std(x)]
                  )
        return(odr.run())

# #### Preliminary wODR weight estimation
# 
# automated weight estimation, weights are estimated as $\delta^{-1}$ for the `X variable`, and $\epsilon^{-1}$ for the `Y variable`.
#
    def estimate_weights(self, x, y, weight_estimation='gm'):
        """automated weight estimation, weights are estimated as $\delta^{-1}$ for the `X variable`,
        and $\epsilon^{-1}$ for the `Y variable`.

        $\delta$ are X axis residuals
        $\epsilon$ are Y axis residuals

        :parameter X: pairwise distance matrix from gene1
        :parameter Y: pairwise distance matrix from gene2
        :parameter weight_estimation:
            'gm' - estimate weights using geometric mean slope (default)
            'huber' - estimate weights from robust non-quadratic Huber Regression function
            'ols' - estimate weights using OLS
            """
        if weight_estimation == 'gm':
            slope = np.std(y)/np.std(x)
            x_res = abs(x - self.line(y,
                                      slope**-1))
            y_res = abs(y - self.line(x,
                                      slope))

        elif weight_estimation == 'huber':
            huber_xy  = HuberRegressor(fit_intercept=False).fit(x.reshape(-1, 1), y)
            huber_yx  = HuberRegressor(fit_intercept=False).fit(y.reshape(-1, 1), x)

            y_res     = abs(y - self.line(x,
                                     huber_xy.coef_))

            x_res     = abs(x - self.line(y,
                                     huber_yx.coef_))

        elif weight_estimation == 'ols':
            xy_params = curve_fit(self.line, x, y)
            y_res     = abs(y - self.line(x,
                                     xy_params[0]))

            yx_params = curve_fit(self.line, y, x)
            x_res     = abs(x - self.line(y,
                                     yx_params[0]))
        else:
            raise Exception('weight_estimation must be "gm", "huber", or "ols"')

        #
        # if residuals are equal do zero it drives the weight to infinity,
        #     and it is good practice not weigh things infinitely
        x_res[x_res==0] = 1e-10
        y_res[y_res==0] = 1e-10

        return(1/abs(x_res),
               1/abs(y_res))

# ### Load input data functions
# 
# 
# Independent of its source, sequences must be named as *genome*`separator`*gene* with no spaces in it. Currently accepted `separator` between genome and gene ids are:
# 
# * *\<genome>*`_`*\<gene>*
# * *\<genome>*`|`*\<gene>*
# * *\<genome>*`.`*\<gene>*
# 
# > If all assessed gene families are single copies, `separators` and *\<gene ids>* may be ignored.

#  Load data from ".mldist" file
    def load_matrix(self, file_path=None):
        """Load <.mldist> file from IQTree into pandas DataFrame

        :parameter file_path: path to <.mldist> file
        """

        dist_matrix = pd.read_csv(file_path,
                                  delim_whitespace = True,
                                  skiprows         = 1,
                                  header           = None,
                                  index_col        = 0)
        dist_matrix.columns = dist_matrix.index

        return(dist_matrix)

# Alternativelly, load pairwise distances from newick file
    def get_matrix_from_tree(self, newick_txt=None):
        """Load pairwise distance matrix from newick tree.

        :parameter newick_text: tree in newick format, not its path, but tree itself (string)
        """
        #
        # create ete3.TreeNode object from newick string
        tree = ete3.Tree(newick_txt, format=1)

        leaf_names = tree.get_leaf_names()
        #
        # add/overwrite internal branch names
        for count, node in enumerate(tree.traverse()):
            if not node.is_leaf():
                node.name = 'node_%i' % count

        #
        # create an edge list from parent node to descendants the will be used to populate network
        #     branch length will be used as edge weights
        edges = []
        for node in tree.traverse():
            #
            # if node is a terminal one (i.e. leaf) there are no descendants to proceed...
            if not node.is_leaf():
                for child in node.get_children():
                    edges.append((node.name,
                                  child.name,
                                  child.dist))

        #
        # load the Directed Acyclic Graph to iGraph, it calculates pairwise distances MUCH, MUCH FASTER than ete3
        #     despite calling it a DAG, the resulting network is undirected, otherwise it would be impossible to
        #     to retrieve distances between leaves...
        dag  = ig.Graph.TupleList(edges     =tuple(edges),
                                  directed  =False, # yeah, the name is misleading, but trees are DAGs...
                                  edge_attrs=['weight'])

        patristic_distances = np.array(dag.shortest_paths(source =leaf_names,
                                                          target =leaf_names,
                                                          weights='weight'))

        #
        # add zeros to the diagonal...
        np.fill_diagonal(patristic_distances, 0.0)

        dist_matrix = pd.DataFrame(index  =leaf_names,
                                   columns=leaf_names,
                                   data   =patristic_distances)
        return(dist_matrix)

# Match possibly co-evolving genes within a genome by looking for pairs minimzing wODR residuals
    def match_copies(self,
                     matrix1, matrix2,
                     taxa1,   taxa2,
                     force_single_copy=False):
        """Select best pairing copies between assessed gene families

        :parameter matrix1: DataFrame with distances from gene1
        :parameter matrix2: DataFrame with distances from gene2
        :parameter taxa1: taxon table from gene1 (pd.DataFrame)
        :parameter taxa2: taxon table from gene2 (pd.DataFrame)

        Return paired copies of input DataFrames"""

        #
        # create a single DataFrame matching taxa from both gene families, and
        #     remove "|<num>" identification for added copies
        all_taxon_pairs           = pd.DataFrame()
        all_taxon_pairs['taxon1'] = [re.sub('\|\d+$', '', taxon, flags=re.M)
                                     for taxon in taxa1.taxon]
        all_taxon_pairs['taxon2'] = [re.sub('\|\d+$', '', taxon, flags=re.M)
                                     for taxon in taxa2.taxon]
        all_taxon_pairs['genome'] = taxa1.genome.tolist()

        #
        # summarize distances matrices by using only its upper triangle (triu)
        triu_indices = np.triu_indices_from(matrix1, k=1)
        condensed1   = matrix1.values[triu_indices]
        condensed2   = matrix2.values[triu_indices]

        #
        # run ODR with no weights...
        model = Model(self.line)
        data  = Data(condensed1,
                     condensed2)
        odr   = ODR(data,
                    model,
                    beta0=[np.std(condensed2) / # Geometric Mean slope estimate
                           np.std(condensed1)]  #
                   )

        regression = odr.run()

        #
        # create DataFrame with all residuals from the preliminary ODR with all
        #      possible combinations of gene within the same genome
        residual_df = pd.DataFrame(columns=['x_taxon1',   'x_genome1',
                                            'x_taxon2',   'x_genome2',

                                            'y_taxon1',   'y_genome1',
                                            'y_taxon2',   'y_genome2',

                                            'x_residual', 'y_residual'],
                                   data=zip(matrix1.index[triu_indices[0]],        #x_taxon1
                                            taxa1.iloc[triu_indices[0], 1].values, #x_genome1
                                            matrix1.index[triu_indices[1]],        #x_taxon2
                                            taxa1.iloc[triu_indices[1], 1].values, #x_genome2

                                            matrix2.index[triu_indices[0]],        #y_taxon1
                                            taxa2.iloc[triu_indices[0], 1].values, #y_genome1
                                            matrix2.index[triu_indices[1]],        #y_taxon2
                                            taxa2.iloc[triu_indices[1], 1].values, #y_genome2

                                            abs(regression.delta),                 #x_residual
                                            abs(regression.eps))                   #y_residual
                                  )
        residual_df['combined_residual'] = residual_df.x_residual + residual_df.y_residual

        #
        # we won't acknowledge residuals from pairs within the same genome, as they
        #     obligatorily include false pairings they can be very misleading...
        #
        # identify within genome residuals here...
        within_genomes = ((residual_df.x_genome1 == residual_df.x_genome2) |
                          (residual_df.y_genome1 == residual_df.y_genome2))
        #
        # ... and remove them here!
        residual_df.drop(index  =residual_df.index[within_genomes],
                         inplace=True)

        #
        # traverse genomes with duplicated gene...
        #     the "for" notation is weird, but ".duplicated()" will return all
        #     duplicated genomes, all copies, and ".unique()" will filter to a
        #     single one!
        for genome in taxa1.genome[taxa1.genome.duplicated()].unique():

            #
            # all homologs from <genome> in <gene family1>
            matrix1_homologs = taxa1.loc[taxa1.genome==genome,
                                         'taxon'].values
            #
            # and all homologs from <genome> in <gene family2>
            matrix2_homologs = taxa2.loc[taxa2.genome==genome,
                                         'taxon'].values

            #
            # empy DataFrame to be filled with residuals from pairs of homologs from
            #     both gene families
            homolog_combinations = pd.DataFrame(columns=['homolog1',
                                                         'homolog2',
                                                         'residual_sum'])
            
            for homolog1, homolog2 in itertools.product(matrix1_homologs,
                                                        matrix2_homologs):
                #
                # retrieve all datapoints involving <homolog1> and <homolog2>
                tmp_df = residual_df.query('(x_taxon1 == @homolog1 | x_taxon2 == @homolog1) &'
                                           '(y_taxon1 == @homolog2 | y_taxon2 == @homolog2)')

                if not tmp_df.shape[0]:
                    continue

                #
                # remove "|<num>" sufix from taxon names to obtain original name
                homolog1 = re.sub('\|\d+$',
                                  '',
                                  homolog1,
                                  flags=re.M)
                homolog2 = re.sub('\|\d+$',
                                  '',
                                  homolog2,
                                  flags=re.M)

                # add all residuals related to each pair of possibly co-evolving genes
                #     within a single genome to a dataframe
                homolog_combinations = homolog_combinations.append(
                    pd.Series(index=['homolog1',
                                     'homolog2',
                                     'residual_sum'],
                              data =[homolog1,
                                     homolog2,
                                     tmp_df.combined_residual.sum()]),
                    ignore_index=True
                )

            #
            # sort pairs of possibly co-evolving genes based on its sum of residuals
            #     the pair with the smallest sum of residuals is the one deviating
            #     the least from the expected linear association between pairwise
            #     distances!
            homolog_combinations.sort_values('residual_sum', inplace=True)
            best_pairs = set()
            while homolog_combinations.shape[0]:
                first_row = homolog_combinations.iloc[0]
                best_pairs.add((first_row.homolog1, first_row.homolog2))
                homolog_combinations.drop(index=homolog_combinations.query(
                    '(homolog1 == @first_row.homolog1) | '
                    '(homolog2 == @first_row.homolog2)'
                ).index, 
                                          inplace=True)
#                 homolog_combinations = homolog_combinations.query('(homolog1 != @first_row.homolog1) & '
#                                                                   '(homolog2 != @first_row.homolog2)').copy()

                if force_single_copy:
                    break

            if force_single_copy:
                indices_to_drop = all_taxon_pairs.query(
                    'genome==@genome'
                )
                indices_to_drop = indices_to_drop.query(
                    '(taxon1 == @first_row.homolog1 & taxon2 != @first_row.homolog2) |'
                    '(taxon1 != @first_row.homolog1 & taxon2 == @first_row.homolog2)'
                )
                all_taxon_pairs.drop(index =indices_to_drop.index,
                                     inplace=True)
                taxa1.drop(          index  =indices_to_drop.index,
                                     inplace=True)
                taxa2.drop(          index  =indices_to_drop.index,
                                     inplace=True)

                
                indices_to_drop = all_taxon_pairs.query(
                    'genome==@genome &'
                    '(taxon1 != @first_row.homolog1 & taxon2 != @first_row.homolog2)'
                )
                all_taxon_pairs.drop(index =indices_to_drop.index,
                                     inplace=True)
                taxa1.drop(          index  =indices_to_drop.index,
                                     inplace=True)
                taxa2.drop(          index  =indices_to_drop.index,
                                     inplace=True)
                
            else:
            # drop all gene combinations where one is not each other's best pairing
                for homolog1, homolog2 in best_pairs:
                    indices_to_drop = all_taxon_pairs.query(
                        '(taxon1 == @homolog1 & taxon2 != @homolog2) |'
                        '(taxon1 != @homolog1 & taxon2 == @homolog2)'
                    ).index


                    all_taxon_pairs.drop(index =indices_to_drop,
                                        inplace=True)

                    taxa1.drop(index  =indices_to_drop,
                               inplace=True)
                    taxa2.drop(index  =indices_to_drop,
                               inplace=True)

        #
        # Debugging
        #
        if taxa1[taxa1.duplicated(subset=['genome', 'gene'])].shape[0] or \
           taxa2[taxa2.duplicated(subset=['genome', 'gene'])].shape[0]:
            raise Exception('Duplicated genomes and genes, investigate...')
            print(f'Duplicated genomes and genes, investigate...',
                  file=sys.stderr)
            
        taxa1.drop_duplicates(subset =['genome', 'gene'], 
                              inplace=True)
        taxa2.drop_duplicates(subset =['genome', 'gene'], 
                              inplace=True)
        
        if not all(taxa1.genome == taxa2.genome):
            raise Exception('**Wow, taxa order is wrong! ABORT!!!')
        
        matrix1 = matrix1.reindex(index  =taxa1.taxon,
                                  columns=taxa1.taxon,
                                  copy   =True)
        matrix2 = matrix2.reindex(index  =taxa2.taxon,
                                  columns=taxa2.taxon,
                                  copy   =True)

        return(matrix1, taxa1, matrix2, taxa2)
    
    def get_Ibc(self, input1, input2, input_type='taxa_table'):
        #add comment
        
        if input_type=='matrix':
            freq1 = Counter( input1.index.tolist() )
            freq2 = Counter( input2.index.tolist() )

        elif input_type=='taxa_table':
            freq1 = Counter( input1.genome.tolist() )
            freq2 = Counter( input2.genome.tolist() )

        freq1_input = []
        freq2_input = []
        for taxon in set(freq1.keys()).union(freq2.keys()):
            if taxon in freq1:
                freq1_input.append(freq1[taxon])
            else:
                freq1_input.append(0)
                
            if taxon in freq2:
                freq2_input.append(freq2[taxon])
            else:
                freq2_input.append(0)

        Ibc = 1 - braycurtis(freq1_input, freq2_input)
        
        return(Ibc)

# Balance distance matrices, duplicate rows/columns to reflect multiples copies in the compared gene families
    def balance_matrices(self, matrix1, matrix2, force_single_copy=False):
        """Remove taxa present in only one matrix, and sort matrices to match taxon order in both DataFrames

        :parameter matrix1: DataFrame with distances from gene1
        :parameter matrix2: DataFrame with distances from gene2

        :return sorted matrix1
        :return sorted matrix1
        :return taxon table from gene1 (pd.DataFrame)
        :return taxon table from gene2 (pd.DataFrame)
        """

        if self.gene_ids:
            Ibc = self.get_Ibc(matrix1, 
                               matrix2,
                               input_type='matrix')
            #
            # if there are no gene ids, there is not much to do, just prune genomes
            #     present in only one of the gene families.
            shared_genomes = np.intersect1d(matrix1.index,
                                            matrix2.index)

            matrix1 = matrix1.reindex(index  =shared_genomes,
                                      columns=shared_genomes,
                                      copy   =True)
            matrix2 = matrix2.reindex(index  =shared_genomes,
                                      columns=shared_genomes,
                                      copy   =True)

            return (matrix1, None,
                    matrix2, None,
                    Ibc)

        #
        # create DataFrames for taxa in each gene family, break sequence names into
        #     <genome> and <gene>, and add together with original sequence name
        tmp_taxa = []
        for index in matrix1.index:
            genome, gene = re.search(self.parse_leaf, index).groups()
            tmp_taxa.append([index, genome, gene])
        taxa1 = pd.DataFrame(columns=['taxon', 'genome', 'gene'],
                             data   =tmp_taxa)

        tmp_taxa = []
        for index in matrix2.index:
            genome, gene = re.search(self.parse_leaf, index).groups()
            tmp_taxa.append([index, genome, gene])
        taxa2 = pd.DataFrame(columns=['taxon', 'genome', 'gene'],
                             data=tmp_taxa)
        
        Ibc = self.get_Ibc(taxa1, 
                           taxa2,
                           input_type='taxa_table')

        #
        # genomes present in both gene families...
        shared_genomes = np.intersect1d(taxa1.genome.unique(),
                                        taxa2.genome.unique())
        
        if len(shared_genomes) < self.min_taxa_overlap:
            return(None, None, None, None, Ibc)
        
        # ... and remove genomes occurring into a single gene family from taxa
        #     DataFrame
        taxa1 = taxa1[taxa1.genome.isin(shared_genomes)]
        taxa2 = taxa2[taxa2.genome.isin(shared_genomes)]

        if not taxa1.genome.is_unique or not taxa2.genome.is_unique:
            #
            # get the number of copies of both gene families within each genome
            taxa1_frequency = taxa1.genome.value_counts()
            taxa2_frequency = taxa2.genome.value_counts()

            #
            # once both gene families contain the same genomes, go through genomes
            #     duplicated in one of them
            for genome in shared_genomes:
                genome1_count = taxa1_frequency[genome]
                genome2_count = taxa2_frequency[genome]

                if genome1_count > 1 or genome2_count > 1:

                    tmp_df1 = taxa1.query('genome == @genome').copy()
                    tmp_df2 = taxa2.query('genome == @genome').copy()

                    names_to_delete1 = tmp_df1.taxon.tolist()
                    names_to_delete2 = tmp_df2.taxon.tolist()

                    count          = 0
                    tmp_df1.taxon += f'|{count}'
                    tmp_df2.taxon += f'|{count}'
                    for (index1, row1), (index2, row2) in itertools.product(tmp_df1.iterrows(), 
                                                                            tmp_df2.iterrows()):
                        count += 1
                        row1.taxon = re.sub('\|\d+$', 
                                            fr'|{count}', 
                                             row1.taxon,
                                            re.M)

                        taxa1       = taxa1.append(row1, 
                                                   ignore_index=True)

                        reference_name          = re.sub('\|\d+$', 
                                                         '', 
                                                         row1.taxon)
                        matrix1[    row1.taxon] = matrix1[    reference_name]
                        matrix1.loc[row1.taxon] = matrix1.loc[reference_name]


                        row2.taxon = re.sub('\|\d+$', 
                                             f'|{count}', 
                                             row2.taxon)
                        taxa2       = taxa2.append(row2, 
                                                   ignore_index=True)

                        reference_name          = re.sub('\|\d+$', 
                                                         '', 
                                                         row2.taxon)
                        matrix2[    row2.taxon] = matrix2[    reference_name]
                        matrix2.loc[row2.taxon] = matrix2.loc[reference_name]


                    taxa1.drop(  index  =taxa1.query('taxon.isin(@names_to_delete1)').index, 
                                 inplace=True)
                    matrix1.drop(index  =names_to_delete1, 
                                 columns=names_to_delete1, 
                                 inplace=True)

                    taxa2.drop(index  =taxa2.query('taxon.isin(@names_to_delete2)').index, 
                               inplace=True)
                    matrix2.drop(index  =names_to_delete2, 
                                 columns=names_to_delete2, 
                                 inplace=True)

        #
        # sort both taxa tables according to genomes for properly matching
        taxa1.sort_values('genome', kind='mergesort', inplace=True)
        taxa2.sort_values('genome', kind='mergesort', inplace=True)

        taxa1.reset_index(drop=True, inplace=True)
        taxa2.reset_index(drop=True, inplace=True)

        #
        # match matrices index and column sorting as taxa tables
        matrix1 = matrix1.reindex(index  =taxa1.taxon,
                                  columns=taxa1.taxon,
                                  copy   =True)
        matrix2 = matrix2.reindex(index  =taxa2.taxon,
                                  columns=taxa2.taxon,
                                  copy   =True)

        #
        # once matrices were sorted to contain every possible pair between genes
        #     present in the same genome, submit it to the <match_copies> function
        if not taxa1.genome.is_unique or not taxa2.genome.is_unique:
            matrix1, taxa1, matrix2, taxa2 = self.match_copies(matrix1, matrix2, 
                                                               taxa1,   taxa2, 
                                                               force_single_copy)

        return(matrix1, taxa1,
               matrix2, taxa2,
               Ibc)

# ### Where the magic happens
    def assess_coevolution(self, matrix1, matrix2):
        """Calculate $I_ES$ between pairwise matrices.

        :parameter matrix1: DataFrame containing pairwise distances from gene1
        :parameter matrix2: DataFrame containing pairwise distances from gene2

        :return wODR coefficient of determination (R^2)
        :return Bray-Curtis Index (Ibc)
        :return Evolutionary Similarity Index (Ies), product between $R^2 * Ibc$
        """

        #
        # balance taxa and matrices from each gene families
        #   and calculate bray-curtis dissimilarity from input matrices
        #
        matrix1, taxa1, matrix2, taxa2, Ibc = self.balance_matrices(matrix1.copy(),
                                                                    matrix2.copy())
        #
        # test if gene families have the minimum overlap between each other.
        min_overlap = True
        if matrix1 is None and matrix2 is None:
            min_overlap = False
        elif not self.gene_ids and taxa1.genome.unique().shape[0] < self.min_taxa_overlap:
            min_overlap = False
        elif   self.gene_ids and               matrix1.shape[0] < self.min_taxa_overlap:
            min_overlap = False

        if not min_overlap:
#             print(f'Assessed matrices have less than {self.min_taxa_overlap} taxa overlap. '
#                    'To change this behavior adjust overlap parameter.',
#                   file=sys.stderr)
            return([None, None, None])

        #
        # generate condensed matrices
        triu_indices = np.triu_indices_from(matrix1, k=1)
        condensed1   = matrix1.values[triu_indices]
        condensed2   = matrix2.values[triu_indices]
        
        if condensed1.std() == 0 or condensed2.std() == 0:
            return(
                0,
                Ibc,
                0 * Ibc # Evolutionary Similarity Index(Ies)
            )

        
        #
        # estimate weights
        odr_weights = self.estimate_weights(condensed1, condensed2)
        #
        # run wODR with condensed matrices and estimated weights
        regression = self.run_odr(condensed1,
                                  condensed2,
                                  *odr_weights)
        #
        # calculate R^2 from wODR model.
        mean_x = np.mean(condensed1)
        mean_y = np.mean(condensed2)

        mean_pred_x = regression.xplus.mean()
        mean_pred_y = regression.y.mean()

        x_SSres = sum(regression.delta**2)
        y_SSres = sum(regression.eps  **2)
        SSres   = x_SSres + y_SSres

        x_SSreg = sum(
            (regression.xplus - mean_pred_x)**2
        )
        y_SSreg = sum(
            (regression.y     - mean_pred_y)**2
        )
        SSreg   = x_SSreg + y_SSreg

        x_SStot = sum(
            (condensed1 - mean_x)**2
        )
        y_SStot = sum(
            (condensed2 - mean_y)**2
        )
        SStot   = x_SStot + y_SStot

        r2 = 1 - SSres/SStot
    #     r2 = SSreg/SStot

        return(
    #         regression,
            r2,
            Ibc,
            r2 * Ibc # Evolutionary Similarity Index(Ies)
        )

# dist1 = run_dist_matrix('/work/clusterEvo/distance_matrices/000284/000284')
# dist2 = run_dist_matrix('/work/clusterEvo/distance_matrices/000302/000302')

# regression, r2 = assess_coevolution(dist1, dist2)

# regression.pprint()
# print(f'\nR**2 = {r2}')

## Beta: [0.49615544]
## Beta Std Error: [0.00033644]
## Beta Covariance: [[1.99398167e-06]]
## Residual Variance: 0.056766828371899364
## Inverse Condition #: 1.0
## Reason(s) for Halting:
##   Sum of squares convergence
##
## R**2 = 0.8947962483462916

