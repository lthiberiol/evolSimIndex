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
from scipy.optimize         import curve_fit
from sklearn.linear_model   import HuberRegressor
from collections            import Counter

import igraph     as ig
import numpy      as np
import pandas     as pd

import sys
import random
import os
import re
import ete3
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
                     matrix1, taxa1,
                     matrix2, taxa2,
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
        all_taxon_pairs['gene1']  = taxa1.gene
        all_taxon_pairs['gene2']  = taxa2.gene
        all_taxon_pairs['genome'] = taxa1.genome.tolist()
        all_taxon_pairs['pairs']  = all_taxon_pairs[['gene1', 'gene2']].apply(lambda x: frozenset(x), axis=1)

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

        ############################################### new code...
        #
        # create DataFrame with all residuals from the preliminary ODR with all
        #      possible combinations of gene within the same genome
        residual_df = pd.DataFrame(columns=['matrix1_gene', 
                                            'matrix2_gene', 
                                            'genome',
                                            'to_drop',
                                            'combined_residual'],
                                   data   =zip(taxa1.iloc[triu_indices[0], 0].values,
                                               taxa2.iloc[triu_indices[0], 0].values,
                                               taxa1.iloc[triu_indices[0], 1].values,
                                               taxa1.iloc[triu_indices[0], 1].values == taxa1.iloc[triu_indices[1], 1].values,
                                               abs(regression.delta)+abs(regression.eps))
                                 )

        residual_df = residual_df.append(
            pd.DataFrame(columns=['matrix1_gene', 
                                  'matrix2_gene', 
                                  'genome',
                                  'to_drop',
                                  'combined_residual'],
                         data   =zip(taxa1.iloc[triu_indices[1], 0].values,
                                     taxa2.iloc[triu_indices[1], 0].values,
                                     taxa1.iloc[triu_indices[1], 1].values,
                                     taxa1.iloc[triu_indices[0], 1].values == taxa1.iloc[triu_indices[1], 1].values,
                                     abs(regression.delta)+abs(regression.eps))
                        ),
            sort        =True,
            ignore_index=True
        )
        
        residual_df.drop(index  =residual_df.index[residual_df.to_drop], 
                         inplace=True)
                
        sum_paired_residuals = residual_df.groupby(
            ['matrix1_gene', 'matrix2_gene']
        ).agg(
            residual_sum=pd.NamedAgg(column ="combined_residual", 
                                     aggfunc=sum),
            genome      =pd.NamedAgg(column='genome', aggfunc=lambda x: x.iloc[0])
        ).reset_index()

        sum_paired_residuals.sort_values('residual_sum', 
                                         inplace=True)
        sum_paired_residuals.reset_index(inplace=True, 
                                         drop   =True)

        best_pairs = pd.DataFrame(columns=['gene1', 'gene2', 'genome'])
        for genome, indices in sum_paired_residuals.groupby('genome').groups.items():

            pairing_possibilities = sum_paired_residuals.loc[indices].copy()

            while pairing_possibilities.shape[0]:
                first_row = pairing_possibilities.iloc[0]

                best_pairs = best_pairs.append(
                    pd.Series(index=['gene1', 
                                     'gene2', 
                                     'genome'],
                              data =[first_row.matrix1_gene, 
                                     first_row.matrix2_gene, 
                                     genome]),
                    ignore_index=True
                )

                if force_single_copy:
                    break

                pairing_possibilities.drop(
                    index=pairing_possibilities.query(
                        '(matrix1_gene == @first_row.matrix1_gene) | '
                        '(matrix2_gene == @first_row.matrix2_gene)'
                    ).index, 
                    inplace=True)

        best_pairs['pairs'] = best_pairs[['gene1', 'gene2']].apply(lambda x: frozenset(x), 
                                                                   axis=1)

        all_taxon_pairs = all_taxon_pairs.query('pairs.isin(@best_pairs.pairs)').copy()
        taxa1 = taxa1.reindex(index=all_taxon_pairs.index)
        taxa2 = taxa2.reindex(index=all_taxon_pairs.index)
        
        taxa1.sort_values('genome', kind='mergesort', inplace=True)
        taxa2.sort_values('genome', kind='mergesort', inplace=True)

        taxa1.reset_index(drop=True, inplace=True)
        taxa2.reset_index(drop=True, inplace=True)

        ############################################### ...up to here

        
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

        numerator, denominator = 0, 0
        for tmp1, tmp2 in zip(freq1_input, freq2_input):
            numerator   += abs(tmp1 - tmp2)
            denominator += abs(tmp1 + tmp2)

        return( 1- numerator/denominator )

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

            ############################################### new code...
            grouped_taxa1 = taxa1.groupby('genome')
            grouped_taxa2 = taxa2.groupby('genome')
        
            taxa1['new_taxon_name'] = taxa1.taxon
            taxa2['new_taxon_name'] = taxa2.taxon

            new_taxa1 = pd.DataFrame(columns=['new_taxon_name', 
                                              'taxon', 
                                              'genome',
                                              'gene'])
            new_taxa2 = pd.DataFrame(columns=['new_taxon_name', 
                                              'taxon', 
                                              'genome', 
                                              'gene'])

            for genome in shared_genomes:
                
                if taxa1_frequency[genome] > 1 or taxa2_frequency[genome] > 1:

                    sub_taxa1 = grouped_taxa1.get_group(genome)
                    sub_taxa2 = grouped_taxa2.get_group(genome)

                    for count, ((index1, row1), 
                                (index2, row2)) in enumerate(
                        itertools.product(sub_taxa1.copy().iterrows(), 
                                          sub_taxa2.copy().iterrows()), 1):

                        row1.new_taxon_name = row1.taxon + f'|{count}'
                        row2.new_taxon_name = row2.taxon + f'|{count}'

                        new_taxa1 = new_taxa1.append(row1, 
                                                     sort=True,
                                                     ignore_index=True)
                        new_taxa2 = new_taxa2.append(row2, 
                                                     sort=True,
                                                     ignore_index=True)
            
            new_taxa1 = new_taxa1.append(
                taxa1.query('genome not in @new_taxa1.genome'),
                sort        =True,
                ignore_index=True
            )
            
            new_taxa2 = new_taxa2.append(
                taxa2.query('genome not in @new_taxa2.genome'),
                sort        =True,
                ignore_index=True
            )
            
            new_taxa1.sort_values('genome', 
                                  kind   ='mergesort', 
                                  inplace=True)
            new_taxa1.reset_index(drop   =True, 
                                  inplace=True)
            new_taxa2.sort_values('genome', 
                                  kind   ='mergesort', 
                                  inplace=True)
            new_taxa2.reset_index(drop   =True, 
                                  inplace=True)

            #
            # match matrices index and column sorting as taxa tables
            matrix1 = matrix1.reindex(index  =new_taxa1.taxon,
                                      columns=new_taxa1.taxon,
                                      copy   =True)
            matrix1.index   = new_taxa1.new_taxon_name
            matrix1.columns = new_taxa1.new_taxon_name

            matrix2 = matrix2.reindex(index  =new_taxa2.taxon,
                                      columns=new_taxa2.taxon,
                                      copy   =True)
            matrix2.index   = new_taxa2.new_taxon_name
            matrix2.columns = new_taxa2.new_taxon_name

            new_taxa1.drop(columns='taxon', inplace=True)
            new_taxa2.drop(columns='taxon', inplace=True)

            new_taxa1.rename(columns={'new_taxon_name':'taxon'}, inplace=True)
            new_taxa2.rename(columns={'new_taxon_name':'taxon'}, inplace=True)
            ############################################### ...up to here
        
        #
        # once matrices were sorted to contain every possible pair between genes
        #     present in the same genome, submit it to the <match_copies> function
        if not taxa1.genome.is_unique or not taxa2.genome.is_unique:
            matrix1, taxa1, matrix2, taxa2 = self.match_copies(matrix1, new_taxa1, 
                                                               matrix2, new_taxa2, 
                                                               force_single_copy)
            
        #
        # sort both taxa tables according to genomes for properly matching
        taxa1.sort_values('genome', kind='mergesort', inplace=True)
        taxa2.sort_values('genome', kind='mergesort', inplace=True)

        taxa1.reset_index(drop=True, inplace=True)
        taxa2.reset_index(drop=True, inplace=True)

        matrix1 = matrix1.reindex(index  =taxa1.taxon,
                                  columns=taxa1.taxon,
                                  copy   =True)
        matrix2 = matrix2.reindex(index  =taxa2.taxon,
                                  columns=taxa2.taxon,
                                  copy   =True)

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

