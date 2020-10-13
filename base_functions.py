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
from io                     import BytesIO, StringIO
from IPython.display        import Javascript, FileLink

import igraph     as ig
import numpy      as np
import seaborn    as sns
import pandas     as pd
import ipywidgets as widgets

import sys
import random
import os
import subprocess
import re
import ete3
import multiprocessing
import itertools


# #### Base linear model
# 
# Linear function with no intercept, forcing it to go through zero.
# 
def line(x, slope):
    """Basic linear regression model
    This is the function provided to scipy's wODR (https://docs.scipy.org/doc/scipy/reference/odr.html)
    """
    return (slope * x) + 0


# #### Basic wODR function
# 
# function receiving `X` and `Y` variables, as well as their estimated weights.
# 
def run_odr(x, y, x_weights, y_weights):
    """"receives pairwise distance matrices and wODR weights

    Parameters:
    X: pairwise distance matrix from gene1
    Y: pairwise distance matrix from gene2
    x_weights: wODR weights for gene1 distances
    y_weights: wODR weights for gene2 distances

    returns an ODR object (https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.html)
    """
    mod = Model(line)
    dat = Data(x, 
               y, 
               wd=x_weights, 
               we=y_weights
    )
    odr = ODR(dat, 
              mod,
              beta0=[np.std(y)/np.std(x)])
    return(odr.run())


# #### Preliminary wODR weight estimation
# 
# automated weight estimation, weights are estimated as $\delta^{-1}$ for the `X variable`, and $\epsilon^{-1}$ for the `Y variable`.
#
def estimate_weights(x, y, weight_estimation='gm'):
    """automated weight estimation, weights are estimated as $\delta^{-1}$ for the `X variable`,
    and $\epsilon^{-1}$ for the `Y variable`.

    $\delta$ are X axis residuals
    $\epsilon$ are Y axis residuals

    Parameters:
    X: pairwise distance matrix from gene1
    Y: pairwise distance matrix from gene2
    weight_estimation:
        'gm' - estimate weights using geometric mean slope (default)
        'huber' - estimate weights from robust non-quadratic Huber Regression function
        'ols' - estimate weights using OLS
        """
    if weight_estimation == 'gm':
        slope = np.std(y)/np.std(x)
        x_res = abs(x - line(y, 
                             slope))
        y_res = abs(y - line(x, 
                             slope))

    elif weight_estimation == 'huber':
        huber_xy  = HuberRegressor(fit_intercept=False).fit(x.reshape(-1, 1), y)
        huber_yx  = HuberRegressor(fit_intercept=False).fit(y.reshape(-1, 1), x)

        y_res     = abs(y - line(x, 
                                 huber_xy.coef_))

        x_res     = abs(x - line(y, 
                                 huber_yx.coef_))
        
    elif weight_estimation == 'ols':
        xy_params = curve_fit(line, x, y)
        y_res     = abs(y - line(x, 
                                 xy_params[0]))
        
        yx_params = curve_fit(line, y, x)
        x_res     = abs(x - line(y, 
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

# #### Load data from ".mldist" file
def load_matrix(file_path=None):
    """Load <.mldist> file from IQTree into pandas DataFrame

    file_path: path to <.mldist> file
    """
        
    dist_matrix = pd.read_csv(file_path,
                              delim_whitespace = True, 
                              skiprows         = 1, 
                              header           = None,
                              index_col        = 0)
    dist_matrix.columns = dist_matrix.index
    
    return(dist_matrix)


# #### Alternativelly, load pairwise distances from newick file
def get_matrix_from_tree(newick_txt=None):
    """Load pairwise distance matrix from newick tree.

    newick_text: tree in newick format, not its path, but tree itself (string)
    """
    tree = ete3.Tree(newick_txt, format=1)

    leaf_names = tree.get_leaf_names()
    for count, node in enumerate(tree.traverse()):
        if not node.is_leaf():
            node.name = 'node_%i' % count

    edges = []
    for node in tree.traverse():
        if not node.is_leaf():
            for child in node.get_children():
                edges.append((node.name,
                              child.name,
                              child.dist))

    dag  = ig.Graph.TupleList(edges     =tuple(edges), 
                              directed  =False,
                              edge_attrs=['weight']
                             )
    
    patristic_distances     = np.array(dag.shortest_paths(source=leaf_names, 
                                                          target=leaf_names, 
                                                          weights='weight'))
                                       
    np.fill_diagonal(patristic_distances, 0.0)
    
    dist_matrix = pd.DataFrame(index  =leaf_names, 
                               columns=leaf_names, 
                               data   =patristic_distances)
    return(dist_matrix)


# #### Match possibly co-evolving genes within a genome by looking for pairs minimzing wODR residuals
def match_copies(matrix1, matrix2, taxa1, taxa2):
    
    all_taxon_pairs           = pd.DataFrame()
    all_taxon_pairs['taxon1'] = [re.sub('\|\d$', '', taxon, flags=re.M)
                                for taxon in taxa1.taxon]
    all_taxon_pairs['taxon2'] = [re.sub('\|\d$', '', taxon, flags=re.M)
                                for taxon in taxa2.taxon]

    triu_indices = np.triu_indices_from(matrix1, k=1)
    condensed1   = matrix1.values[triu_indices]
    condensed2   = matrix2.values[triu_indices]

    model = Model(line)
    data  = Data(condensed1, 
                 condensed2)
    odr   = ODR(data, 
                model,
                beta0=[np.std(condensed2) /
                       np.std(condensed1)]
               )

    regression = odr.run()

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

    within_genomes = ((residual_df.x_genome1 == residual_df.x_genome2) | 
                      (residual_df.y_genome1 == residual_df.y_genome2))

    residual_df.drop(index  =residual_df.index[within_genomes], 
                     inplace=True)
    
    for genome in taxa1.genome[taxa1.genome.duplicated()].unique():
    
        matrix1_homologs = taxa1.loc[taxa1.genome==genome, 
                                     'taxon'].values
        matrix2_homologs = taxa2.loc[taxa2.genome==genome, 
                                     'taxon'].values

        homolog_combinations = pd.DataFrame(columns=['homolog1', 
                                                     'homolog2', 
                                                     'residual_sum'])
        for homolog1, homolog2 in itertools.product(matrix1_homologs,
                                                    matrix2_homologs):
            tmp_df = residual_df.query('(x_taxon1 == @homolog1 | x_taxon2 == @homolog1) &'
                                       '(y_taxon1 == @homolog2 | y_taxon2 == @homolog2)')

            if not tmp_df.shape[0]:
                continue

            #
            # remove "|<num>" sufix from taxon names to obtain original name
            homolog1 = re.sub('\|\d$', 
                              '',
                              homolog1, 
                              flags=re.M)
            homolog2 = re.sub('\|\d$',
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

        homolog_combinations.sort_values('residual_sum', inplace=True)
        best_pairs = set()
        while homolog_combinations.shape[0]:
            first_row = homolog_combinations.iloc[0]
            best_pairs.add((first_row.homolog1, first_row.homolog2))
            homolog_combinations = homolog_combinations.query(f'(homolog1 != "{first_row.homolog1}") & '
                                                              f'(homolog2 != "{first_row.homolog2}")').copy()
            
        # drop all gene combinations where one is not each other's best pairing
        for homolog1, homolog2 in best_pairs:
            indices_to_drop = all_taxon_pairs.query(
                '(taxon1 == @homolog1 & taxon2 != @homolog2) |'
                '(taxon1 != @homolog1 & taxon2 == @homolog2)'
            ).index

            all_taxon_pairs.drop(index=indices_to_drop, 
                                inplace=True)

            taxa1.drop(index  =indices_to_drop, 
                       inplace=True)
            taxa2.drop(index  =indices_to_drop, 
                       inplace=True)
    
    matrix1 = matrix1.reindex(index  =taxa1.taxon, 
                              columns=taxa1.taxon, 
                              copy   =True)
    matrix2 = matrix2.reindex(index  =taxa2.taxon, 
                              columns=taxa2.taxon, 
                              copy   =True)
    
    return(matrix1, taxa1, matrix2, taxa2)


# #### Balance distance matrices, duplicate rows/columns to reflect multiples copies in the compared gene families

# In[9]:


def balance_matrices(matrix1, matrix2):

    if gene_ids.value:
        shared_genomes = np.intersect1d(matrix1.index,
                                        matrix2.index)

        matrix1 = matrix1.reindex(index  =shared_genomes,
                                  columns=shared_genomes,
                                  copy   =True)
        matrix2 = matrix2.reindex(index  =shared_genomes,
                                  columns=shared_genomes,
                                  copy   =True)

        return (matrix1, None,
                matrix2, None)
    
    tmp_taxa = []
    for index in matrix1.index:
        genome, gene = re.search(parse_leaf, index).groups()
        tmp_taxa.append([index, genome, gene])
    taxa1 = pd.DataFrame(columns=['taxon', 'genome', 'gene'],
                         data   =tmp_taxa)

    tmp_taxa = []
    for index in matrix2.index:
        genome, gene = re.search(parse_leaf, index).groups()
        tmp_taxa.append([index, genome, gene])
    taxa2 = pd.DataFrame(columns=['taxon', 'genome', 'gene'],
                         data=tmp_taxa)

    shared_genomes = np.intersect1d(taxa1.genome.unique(), 
                                    taxa2.genome.unique())

    taxa1 = taxa1[taxa1.genome.isin(shared_genomes)]
    taxa2 = taxa2[taxa2.genome.isin(shared_genomes)]

    if not taxa1.genome.is_unique or not taxa2.genome.is_unique:
    
        taxa1_frequency = taxa1.genome.value_counts() 
        taxa2_frequency = taxa2.genome.value_counts() 

        for genome in shared_genomes:
            genome1_count = taxa1_frequency[genome]
            genome2_count = taxa2_frequency[genome]

            if genome1_count > 1:
                #
                # one of the matrices must be traversed in the inversed order to make sure an 
                #     all VS all combination is obtained. That is the reason of the "iloc[::-1]"
                #     during the querying
                tmp_df = taxa2.iloc[::-1].query('genome == @genome').copy()
                for _ in range(genome1_count - 1):
                    for index, row in tmp_df.iterrows():
                        tmp_row = row.copy()
                        tmp_row.taxon += f'|{_}'
                        taxa2      = taxa2.append(tmp_row, ignore_index=True)

                        reference_name = re.sub('\|\d+$', '', tmp_row.taxon, flags=re.M)
                        matrix2[    tmp_row.taxon] = matrix2[    reference_name]
                        matrix2.loc[tmp_row.taxon] = matrix2.loc[reference_name]


            if genome2_count > 1:
                #
                # as we queried the other matrix in the reverse order, we traverse this one regularly
                tmp_df = taxa1.query('genome == @genome').copy()
                for _ in range(genome2_count - 1):
                    for index, row in tmp_df.iterrows():
                        tmp_row = row.copy()
                        tmp_row.taxon += f'|{_}'
                        taxa1 = taxa1.append(tmp_row, ignore_index=True)

                        reference_name = re.sub('\|\d+$', '', tmp_row.taxon, flags=re.M)
                        matrix1[    tmp_row.taxon] = matrix1[    reference_name]
                        matrix1.loc[tmp_row.taxon] = matrix1.loc[reference_name]

    #
    # sort both taxa tables according to genomes for properly matching
    taxa1.sort_values('genome', inplace=True)
    taxa2.sort_values('genome', inplace=True)

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
    
    if not taxa1.genome.is_unique or not taxa2.genome.is_unique:
        matrix1, taxa1, matrix2, taxa2 = match_copies(matrix1, matrix2, taxa1, taxa2)
    
    return(matrix1, taxa1, 
           matrix2, taxa2)


# ### Where the magic happens

# In[10]:


def assess_coevolution(matrix1, matrix2):

    matrix1, taxa1, matrix2, taxa2 = balance_matrices(matrix1.copy(), 
                                                      matrix2.copy())

    min_overlap = True
    if not gene_ids.value and taxa1.genome.unique().shape[0] < min_taxa_overlap.value:
        min_overlap = False
    elif   gene_ids.value and               matrix1.shape[0] < min_taxa_overlap.value:
        min_overlap = False

    if not min_overlap:
        print(f'Assessed matrices have less than {min_taxa_overlap.value} taxa overlap. '
               'To change this behavior adjust overlap parameter.',
              file=sys.stderr)
        return([None, None])

    condensed1 = squareform(matrix1.values, checks=False)
    condensed2 = squareform(matrix2.values, checks=False)
    
    odr_weights = estimate_weights(condensed1, condensed2)
    
    regression = run_odr(condensed1, 
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
    

    #
    # bray-curtis step
    #    
    if gene_ids.value:
        taxa1 = matrix1.index.tolist()
        taxa2 = matrix2.index.tolist()
    else:
        taxa1 = taxa1.genome.tolist()
        taxa2 = taxa2.genome.tolist()

    taxon_intersect = set(taxa1).intersection(taxa2)
    freq1 = Counter(taxa1)
    freq2 = Counter(taxa2)

    freq1_input = []
    freq2_input = []
    for taxon in set(taxa1).union(taxa2):
        if taxon in freq1:
            freq1_input.append(freq1[taxon])
        else:
            freq1_input.append(0)
        if taxon in freq2:
            freq2_input.append(freq2[taxon])
        else:
            freq2_input.append(0)    
            
    Ibc = 1 - braycurtis(freq1_input, freq2_input)

    return(
#         regression, 
        r2,
        Ibc,
        r2 * Ibc # Evolutionary Similarity Index(Ies)
    )


# In[11]:


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


# ## Crappy notebook interface, but still an interface!

# In[12]:


evol_dist_source = widgets.Dropdown(
    options    =[('',                       0       ), 
                 ('FASTA files',            'fasta' ), 
                 ('IQTree ".mldist" files', 'matrix'), 
                 ('newick files',           'tree'  ),
#                  ('example',                'example')
                ],
    disabled   =False,
    indent     =False,
    value      =0,
    layout     ={'width':'auto'}
)

must_align = widgets.Checkbox(
    value   =False,  
    disabled=True,
    indent  =False,
    description='Provided FASTAS are not yet aligned',
    layout     ={'width':'auto'}
)

gene_ids = widgets.Checkbox(
    value   =False,  
    disabled=False,
    indent  =False,
    description='Sequences are identified by genome only '
    '(all sequences from the same genome have the same name)',
    layout     ={'width':'auto'}
)

min_taxa_overlap = widgets.IntText(value      =5, 
                                   indent     =False,
                                   disabled   =False)

genome_gene_sep = widgets.Dropdown(
    options    =[('',                                     0  ), 
                 ('<genome>_<gene>', '_'), 
                 ('<genome>|<gene>', '|'), 
                 ('<genome>.<gene>', '.')],
    disabled   =False,
    indent     =True,
    value      =0,
    layout     ={'width':'auto'}
)

input_files = widgets.FileUpload(
    accept  ='',   # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=True, # True to accept multiple files upload else False
    disabled=True
)

def toggle_align_widgets(dropdown_source):
    input_files.disabled = not dropdown_source.new
    
    if dropdown_source.new == 'fasta':
        must_align.disabled = False
#     elif dropdown_source.new == 'example':
#         genome_gene_sep.value = '_'
#         input_files.disabled  = True
    else:
        must_align.disabled = True
        must_align.value    = False
    
def toggle_genome_gene_sep(checkbox):
    genome_gene_sep.disabled = checkbox.new
    if checkbox.new:
        genome_gene_sep.value = 0
        
def clear_uploads(*args):
    input_files.value.clear()
    input_files._counter = 0
    input_files.disabled = False
    
    evol_dist_source.value = 0
    genome_gene_sep.value  = 0
    
    example.flag = False
        
clear_button = widgets.Button(description='Clear upload',
                              button_style='warning',
                              tooltip     ='Click to clear uploaded files')
clear_button.on_click(clear_uploads)

evol_dist_source.observe(toggle_align_widgets, names='value')
gene_ids.observe(toggle_genome_gene_sep,       names='value')


#
# load example
#
example = widgets.Button(description='Load example',
                         button_style='success',
                         tooltip     ='Load example parameters')
example.flag = False    
def load_example(*args):
    evol_dist_source.disabled = False
    evol_dist_source.value    = 'matrix'
    
    genome_gene_sep.value     = '_'
    
    input_files.disabled      = True
    
    example.flag = True
    
example.on_click(load_example)

#
# submit data
#
def run_all(ev):
    display(Javascript('IPython.notebook.execute_cells_below()'))

submit = widgets.Button(description ="Submit",
                        button_style='success',
                        tooltip     ='Click here to continue with provided data')
submit.on_click(run_all)

#
# threads
#
num_threads = widgets.IntSlider(min=1, 
                                max=multiprocessing.cpu_count())

#
# download through colab
#
if in_colab:
    def download(*args):
        files.download('Ies.csv')

    download_csv = widgets.Button(description='Download Ies.csv',
                          button_style='success')
    download_csv.on_click(download)


# In[13]:


display(widgets.HBox([widgets.Label('Source of pairwise distances (there is an "example" options): '), 
                      evol_dist_source]),
        must_align,
        gene_ids,
        widgets.HBox([widgets.Label('Genome and gene ids are separated by which character: '),
                      genome_gene_sep]),
        
        widgets.HBox([widgets.Label('Minimum taxa containing both assessed gene families: '),
                      min_taxa_overlap]),
        
        widgets.HBox([widgets.Label('Number of threads to use: '),
                      num_threads]),
        
#         input_files,
        widgets.HBox([input_files, example]),
        clear_button,
        submit
       )


# If `Submit` button above is not working, in the **Toolbar** click in `Cell`->`Run All Bellow`

# #### Parsing provided data and parameters

# In[14]:


# @title Breaking "run all" if no data was uploadded

if not input_files._counter > 1 and not example.flag:
    raise ValueError('You must upload at least two files!')


# In[ ]:


# @title Loading data...

dist_matrices = []
group_names   = []

if evol_dist_source.value == 'tree':
    for file_name, file_itself in input_files.value.items():
        dist_matrices.append( get_matrix_from_tree(file_itself['content'].decode('utf-8')) )
        group_names.append( file_name )
        
elif evol_dist_source.value == 'matrix':
    if not input_files._counter and example.flag:
        for file_itself in ['000284', '000302', '000304', '000321', '000528', 
#                             '000574', '000575', '000595', '000602', '000607',
#                             '000611', '000617', '000620', '000621', '000625',
                            '000632', '000645', '000647', '000657', '000663']:
            dist_matrices.append(load_matrix(
                f'https://raw.githubusercontent.com/lthiberiol/evolSimIndex/master/tests/{file_itself}.mldist'
            ))
            group_names.append( file_itself )
    else:
        for file_name, file_itself in input_files.value.items():
            dist_matrices.append( load_matrix(BytesIO(file_itself['content'])) )
            group_names.append( file_name )
        
if   genome_gene_sep.value == '_':
    parse_leaf = re.compile('^(GC[AF]_\d+(?:\.\d)?)[_|](.*)$')
elif genome_gene_sep.value == '|':
    parse_leaf = re.compile('^(\S+?)\|(\S+)$')
elif genome_gene_sep.value == '.':
    parse_leaf = re.compile('^(\d+?)\.(.*)$')
    
if min_taxa_overlap.value < 2:
    min_taxa_overlap.value = 2


# ## Assessing evolutionary similarity between gene families!

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint(f\'Assessing Ies between {len(group_names)} genes\')\nprint(f\'\\t**using {num_threads.value} threads\\n\')\n\nmatrix_combinations     = itertools.combinations(dist_matrices, 2)\ngroup_name_combinations = itertools.combinations(group_names,   2)\ngroup_name_combinations = np.array( list(group_name_combinations) )\n\npool    = multiprocessing.Pool(processes=num_threads.value)\nresults = pool.starmap(assess_coevolution, matrix_combinations)\npool.close()\npool.join()\n\ncoevol_df = pd.DataFrame(columns=[\'R_squared\', \'Ibc\', \'Ies\'], \n                         data   =results)\n\ncoevol_df[\'gene1\'] = group_name_combinations[:, 0]\ncoevol_df[\'gene2\'] = group_name_combinations[:, 1]\n\ncoevol_df.to_csv(\'Ies.csv\')\n\nif not in_colab:\n    local_file = FileLink(\'Ies.csv\', result_html_prefix="Click here to download: ")\n    display(local_file)\nelse:\n    display(download_csv)')

