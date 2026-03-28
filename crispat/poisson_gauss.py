import os
import sys, getopt
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from torch.distributions import constraints
from scipy import sparse, stats, special
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import multiprocessing
from dask.distributed import Client, LocalCluster
import dask.bag as db
from functools import partial

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.distributions.mixture import MaskedMixture
from pyro.distributions.torch_distribution import TorchDistributionMixin 
    
    
class cont_Poisson(dist.TorchDistribution):
    arg_constraints = {'rate': constraints.nonnegative}
    def __init__(self, rate, validate_args=None):
        self.rate = rate
        super().__init__(batch_shape=rate.shape, event_shape=torch.Size([]), 
                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(cont_Poisson, _instance)
        new.rate = self.rate.expand(batch_shape)
        super(cont_Poisson, new).__init__(batch_shape=batch_shape, event_shape=torch.Size([]))
        return super(cont_Poisson, new).expand(batch_shape, _instance=new)

    def sample(self, sample_shape=torch.Size()):
        poisson = dist.Poisson(self.rate)
        return poisson.sample(sample_shape)
    
    def log_prob(self, value):
        prob = poisson_prob(value, self.rate)
        return torch.log(prob)
    

@config_enumerate
def model(data):
    '''
    Poisson-Gaussian Mixture Model 
    '''
    # Global variables
    weights = pyro.sample("weights", dist.Dirichlet(torch.tensor([0.9, 0.1])))
    loc = pyro.sample("mu", dist.Normal(3.0, 2.0))
    scale = pyro.sample("scale", dist.LogNormal(2.0, 1.0))
    lam = pyro.sample("lam", dist.LogNormal(0.0, 1.0))

    with pyro.plate("data", len(data)):
        # Local variables
        assignment = pyro.sample("assignment", dist.Categorical(weights)) 
        assignment = assignment == 1 #boolean mask needed for MaskedMixture

        poisson = cont_Poisson(lam)
        normal = dist.Normal(loc, scale)
        pyro.sample("obs", MaskedMixture(assignment, poisson, normal, validate_args = False), obs=data)
        
    
def initialize(seed, optim, elbo, data):
    '''
    Initialization for SVI 
    
    Args:
        seed: (str) seed that is used in pyro
        optim: pyro optimizer
        elbo: pyro loss function
        data: (tensor) observed transformed gRNA counts
        conditioned_model: (pyro model) conditioned pyro mixture model
    
    Returns:
        Initial loss
    '''
    global global_guide, svi
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    global_guide = AutoDelta(
        poutine.block(model, expose=["weights", "mu", "scale", "lam"])
    )
    svi = SVI(model, global_guide, optim, loss=elbo)
    return svi.loss(model, global_guide, data)


def plot_loss(losses, gRNA, output_dir):
    '''
    Saves a plot of the loss over the SVI steps
    
    Args:
        losses (list): loss over the SVI steps
        gRNA (str): name of the gRNA used for the plot title
        output_dir (str): name of the output directory
    
    Returns:
        None
    '''
    plt.figure(figsize=(8, 3), dpi=300).set_facecolor("white")
    plt.plot(losses)
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title("Convergence of SVI for " + gRNA)
    plt.savefig(output_dir+"loss_plots/loss_"+gRNA+".png", bbox_inches="tight")
    plt.close()
    
    
def plot_fitted_model(data, weights, mu, scale, lam, threshold, gRNA, output_dir):
    '''
    Saves a plot of the data histogram and the fitted mixture model
   
    Args:
        data (tensor): observed transformed gRNA counts
        weights (np array): estimated proportion for Poisson and Normal component
        mu (float): MAP estimate for the mean of the Normal distribution
        scale (float): MAP estimate for the scale of the Normal distribution
        lam (float): MAP estimate for the lambda of the Poisson distribution
        threshold (int): threshold for assigning it to the Normal component
        gRNA (str): name of the gRNA used for the plot title
        output_dir (str): name of the output directory
    
    Returns:
        None
    '''
    X = np.arange(0, max(data)+2, 0.1)
    Y1 = weights[0] * torch.tensor([poisson_prob(k, torch.tensor(lam)) for k in X])
    Y2 = weights[1] * stats.norm.pdf(X, mu, scale)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    
    # Use adaptive binwidth or bins based on data range to avoid errors with small ranges
    data_range = max(data) - min(data)
    if data_range > 1.0:
        sns.histplot(data, binwidth=0.5, color='grey', stat="proportion", ax=ax)
    elif data_range > 0:
        # For small ranges, use a fixed number of bins instead of binwidth
        n_bins = max(10, int(len(data) / 5))
        sns.histplot(data, bins=n_bins, color='grey', stat="proportion", ax=ax)
    else:
        # If all values are identical, just plot a single bar
        sns.histplot(data, bins=1, color='grey', stat="proportion", ax=ax)
    
    ax.plot(X, Y1, "r-", label = "Poisson")
    ax.plot(X, Y2, "b-", label = "Normal")
    ax.plot(X, Y1 + Y2, "k--", label = "Mixture model")
    ax.axvline(threshold, c = "green", label = "Threshold")
    plt.legend()
    plt.title("Poisson-Gaussian mixture model for " + gRNA)
    plt.ylabel("Probability Density")
    plt.xlabel("Log2 " + gRNA + " UMI counts")
    plt.savefig(output_dir + "fitted_model_plots/fitted_model_" + gRNA + ".png", bbox_inches="tight")
    plt.close()
    
    
def poisson_prob(k, lam):
    '''
    Calculates the probability P(X = k) of a Poisson distribution with parameter lam
    
    Args:
        k (float): value for which to calculate the probability
        lam (tensor): lambda parameter of the Poisson distribution
    
    Returns:
        List of probabilities
    '''
    # gamma(k + 1) = k! is used to get the probability also for non-integer values
    prob = (lam ** k) * torch.exp(-lam) / special.gamma(k + 1)
    return prob
    
    
def prob_normal_component(X, weights, mu, scale, lam):
    '''
    Calculates the probability for belonging to the Gaussian component given observations
    
    Args:
        X (list): list of values for which the probability is calculated
        weights (np array): estimated proportion for Poisson and Normal component
        mu (float): MAP estimate for the mean of the Normal distribution
        scale (float): MAP estimate for the scale of the Normal distribution
        lam (float): MAP estimate for the lambda of the Poisson distribution
    
    Returns:
        List of probabilities
    '''
    nominator = torch.tensor(stats.norm.pdf(X, mu, scale) * weights[1])
    denominator = nominator + torch.tensor([poisson_prob(k, torch.tensor(lam)) for k in X]) * weights[0]
    prob = nominator / denominator
    return prob
    

def fit_PGMM(gRNA, adata_crispr, output_dir, seed, n_iter):
    '''
    Fits Poisson-Gaussian mixture model for log2 of non-zero UMI counts of one gRNA 
    
    Args:
        gRNA (str): name of the gRNA
        adata_crispr (AnnData): anndata object with UMI counts of CRISPR Guide Capture
        output_dir (str): directory in which the resulting plots will be saved
        seed (int): seed used for pyro
        n_iter (int): number of steps for training the model
        
    Returns:
        List of cells perturbed with the specified gRNA, as well as the inferred threshold
    '''
    # Set optimizer and elbo parameters
    optim = pyro.optim.Adam({"lr": 0.01, "betas": [0.8, 0.99]})
    elbo = TraceEnum_ELBO(num_particles = 1, max_plate_nesting=1)
    
    # Data used to fit the model: log2 transformation of non-zero UMI counts for a given gRNA 
    selected_guide = adata_crispr[:,[gRNA]].X
    data = selected_guide.toarray() 
    data = torch.tensor(np.log2(data[data != 0])).float()

    # Only fit model for gRNAs with non-zero counts in at least 2 cells and with a maximum count of at least 2
    if len(data) < 2:   
        print(gRNA + " has only " + str(len(data)) + " cells with non-zero counts, so no model is fitted for that gRNA")
        return([], 0, 0, 0)
    if max(data) < np.log2(2):
        print("Max UMI count for " + gRNA + " is " + str(max(data).item()) + ", so no model is fitted for that gRNA")
        return([], 0, 0, 0)
    
    # Choose the best among 10 random initializations.
    loss, seed = min((initialize(seed, optim, elbo, data), seed) for seed in range(10))

    # Initialization of SVI
    initialize(seed, optim, elbo, data)

    # Train the model n_steps steps with early stopping when loss doesn't change at least 0.001 for 50 steps
    losses = []
    min_loss = 1.e8
    last_step = 0

    for step in range(n_iter):
        loss = svi.step(data)
        if loss < min_loss - 0.001:
            min_loss = loss
            last_step = step
        losses.append(loss)
        if (step - last_step) > 50:
            break
        
    # MAP estimates of the model
    map_estimates = global_guide(data)
    weights = map_estimates["weights"].data.numpy()
    mu = map_estimates["mu"].item()
    scale = map_estimates["scale"].item()
    lam = map_estimates["lam"].item()
    estimates = pd.DataFrame({'gRNA': [gRNA], 
                              'weight_Poisson': [weights[0]], 
                              'weight_Normal': [weights[1]], 
                              'lambda': [lam], 
                              'mu': [mu], 
                              'scale': [scale]})

    # create plot of the loss
    plot_loss(losses, gRNA, output_dir)
    
    # threshold for which probability is higher to belong to the normal component
    X = np.arange(1, selected_guide.toarray().max()+1, 1)
    log_X = np.log2(X)
    df = pd.DataFrame({'t': X, 'prob_normal_component': prob_normal_component(log_X, weights, mu, scale, lam)})
    threshold = df.loc[(df.prob_normal_component > 0.5), 't'].min()
    
    # create plot of the mixture distribution
    try:
        plot_fitted_model(data, weights, mu, scale, lam, np.log2(threshold), gRNA, output_dir)
    except Exception as e:
        print(f"Error plotting fitted model for {gRNA}: {e}")
        print(data)
        print(weights)
        print(mu)
        print(scale)
        print(lam)
        print(threshold)
        raise RuntimeError("Error plotting fitted model.")
        
    
    # get cells with gRNA counts above the threshold
    perturbed_cells = adata_crispr.obs_names[selected_guide.toarray().reshape(-1) >= threshold].tolist()
    return(perturbed_cells, threshold, losses[-1], estimates)


def parallel_assignment(gRNA, adata_crispr, output_dir, seed, n_iter):
    '''
    Wrapper function for fit_PGMM to enable parallel processing
    
    Args:
        gRNA (str): name of the gRNA
        adata_crispr (AnnData): anndata object with UMI counts of CRISPR Guide Capture
        output_dir (str): directory in which the resulting plots will be saved
        seed (int): seed used for pyro
        n_iter (int): number of steps for training the model
    
    Returns:
        Tuple of (gRNA name, perturbed cells, threshold, loss, estimates)
    '''
    perturbed_cells, threshold, loss, map_estimates = fit_PGMM(gRNA, adata_crispr, output_dir, seed, n_iter)
    return (gRNA, perturbed_cells, threshold, loss, map_estimates)


def ga_poisson_gauss(input_file, output_dir, start_gRNA = 0, step = None, n_iter = 500, n_counts = None, UMI_threshold = 0,
                     parallelize = True, n_processes = None, mem_limit = '10GB'):
    '''
    Guide assignment in which a Poisson-Gaussian mixture model is fitted to the non-zero log-transformed UMI counts
    
    Args:
        input_file (str): path to the stored anndata object with the gRNA counts
        output_dir (str): directory in which to store the resulting assignment
        start_gRNA (int, optional): index of the start gRNA when parallelizing assignment for gRNA sets
        step (int, optional): number of gRNAs for which the assignment is done (if set to None, assignment for all gRNAs in the data)
        n_iter (int, optional): number of steps for training the model
        n_counts (int, optional): subsample the gRNA counts per cell to a total of n_counts. If None (default), the UMI count matrix is used without any downsampling.
        UMI_threshold (int, optional): Additional UMI threshold for assigned cells which is applied after creating the initial assignment to remove cells with fewer UMI counts than this threshold (default: no additional UMI threshold)
        parallelize (bool, optional): whether to parallelize the computation over the gRNAs (default = True)
        n_processes (int, optional): specifies number of processes to use for parallelization if parallelize = True. If set to None (default), all available CPUs will be used (if this number is not higher than the number of gRNAs).
        mem_limit (str, optional): set memory limit for the dask cluster (default: 10GB)
        
    Returns:
        None
    '''   
    print('Guide assignment with Poisson-Gaussian mixture model as in Replogle et al.')
    # If output_dir doesn't exist, the output folders are created
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + "fitted_model_plots/")
        os.makedirs(output_dir + "loss_plots/")
        print("The output directory " + output_dir +  " was created") 

    # Load gRNA counts data
    print('Load gRNA counts')
    adata_crispr = sc.read_h5ad(input_file)
    gRNA_list = adata_crispr.var_names.tolist()
    
    if step != None:
        end_gRNA = start_gRNA + step - 1
        if end_gRNA >= len(gRNA_list):
            end_gRNA = len(gRNA_list) - 1
        gRNA_list = gRNA_list[start_gRNA:(end_gRNA + 1)]
        
    # Downsampling of gRNA counts per cell to a maximum of n_counts per cell
    if n_counts != None:     
        sc.pp.downsample_counts(adata_crispr, counts_per_cell = n_counts)
    
    # Fit Poisson-Gaussian Mixture Model (PGMM) for each gRNA
    perturbations = pd.DataFrame()
    thresholds = pd.DataFrame()
    losses = pd.DataFrame()
    estimates = pd.DataFrame()
    
    print('Fit Poisson-Gaussian Mixture Model for each gRNA')
    
    if parallelize:
        if n_processes == None:
            n_processes = os.cpu_count() # if not specified all available CPUs will be used
        if n_processes > len(gRNA_list):
            n_processes = len(gRNA_list)
        print(str(n_processes) + ' parallel processes used')
        
        # Set multiprocessing start method to 'fork' for compatibility with workflow systems
        # This avoids the FileNotFoundError when running from stdin in environments like Cromwell
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'fork':
                multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            # Start method already set, continue with existing method
            pass
        
        # start a local Dask cluster
        cluster = LocalCluster(n_workers = n_processes, threads_per_worker = 1, memory_limit = mem_limit)
        client = Client(cluster, heartbeat_interval='5s', timeout='30s')
        
        # Scatter the large adata object once to all workers to avoid repeated serialization
        scattered_adata = client.scatter(adata_crispr, broadcast=True)
        
        # run in parallel over gRNAs with progress tracking
        # Use tqdm to track progress
        with tqdm(total=len(gRNA_list), desc="Processing gRNAs") as pbar:
            # Process in chunks and update progress as each completes
            results = []
            for gRNA in gRNA_list:
                # Pass scattered_adata directly as argument so Dask can resolve the Future
                future = client.submit(parallel_assignment, gRNA, scattered_adata, output_dir, 2024, n_iter)
                future.add_done_callback(lambda _: pbar.update())
                results.append(future)
            
            # Gather all results
            from dask.distributed import as_completed
            results = [f.result() for f in results]
        
        client.close()
        cluster.close()
        
        # combine the results per gRNA
        for gRNA, perturbed_cells, threshold, loss, map_estimates in results:
            if len(perturbed_cells) != 0:
                # get UMI_counts of assigned cells
                UMI_counts = adata_crispr[perturbed_cells, [gRNA]].X.toarray().reshape(-1)
                df = pd.DataFrame({'cell': perturbed_cells, 'gRNA': gRNA, 'UMI_counts': UMI_counts})
                perturbations = pd.concat([perturbations, df], ignore_index = True)
                thresholds = pd.concat([thresholds, pd.DataFrame({'gRNA': [gRNA], 'threshold': [threshold]})])
                losses = pd.concat([losses, pd.DataFrame({'gRNA': [gRNA], 'loss': [loss]})])
                estimates = pd.concat([estimates, map_estimates])
    
    # Fit Poisson-Gaussian Mixture Model without parallelization
    else:
        for gRNA in tqdm(gRNA_list):
            time.sleep(0.01)
            perturbed_cells, threshold, loss, map_estimates = fit_PGMM(gRNA, adata_crispr, output_dir, 2024, n_iter)
            if len(perturbed_cells) != 0:
                # get UMI_counts of assigned cells
                UMI_counts = adata_crispr[perturbed_cells, [gRNA]].X.toarray().reshape(-1)
                df = pd.DataFrame({'cell': perturbed_cells, 'gRNA': gRNA, 'UMI_counts': UMI_counts})
                perturbations = pd.concat([perturbations, df], ignore_index = True)
                thresholds = pd.concat([thresholds, pd.DataFrame({'gRNA': [gRNA], 'threshold': [threshold]})])
                losses = pd.concat([losses, pd.DataFrame({'gRNA': [gRNA], 'loss': [loss]})])
                estimates = pd.concat([estimates, map_estimates])
    
    # Optional filtering to assigned cells that have at least 'UMI_threshold' counts
    if perturbations.shape[0] != 0:
        perturbations = perturbations[perturbations['UMI_counts'] >= UMI_threshold]
    
    # Save data frames with the results
    if step == None:
        perturbations.to_csv(output_dir + 'assignments.csv', index = False)
        thresholds.to_csv(output_dir + 'gRNA_thresholds.csv', index = False)
        losses.to_csv(output_dir + 'gRNA_losses.csv', index = False)
        estimates.to_csv(output_dir + 'gRNA_estimates.csv', index = False)
    else:
        perturbations.to_csv(output_dir + 'assignments_'+str(start_gRNA)+'-'+str(end_gRNA)+'.csv', index = False)
        thresholds.to_csv(output_dir + 'gRNA_thresholds_'+str(start_gRNA)+'-'+str(end_gRNA)+'.csv', index = False)
        losses.to_csv(output_dir + 'gRNA_losses_'+str(start_gRNA)+'-'+str(end_gRNA)+'.csv', index = False)
        estimates.to_csv(output_dir + 'gRNA_estimates_'+str(start_gRNA)+'-'+str(end_gRNA)+'.csv', index = False)
    
    print('Done: outputs are saved in ' + output_dir)
    
    
if __name__ == "__main__":
    pass
