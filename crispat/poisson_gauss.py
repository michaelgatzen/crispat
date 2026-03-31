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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for containerized/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import time
import multiprocessing
# Set multiprocessing start method early for better compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set
from functools import partial
from datetime import datetime
import threading

# Disable torch's internal parallelism to avoid conflicts with multiprocessing
import torch
torch.set_num_threads(1)

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


def process_gRNA_chunk(chunk_data):
    '''
    Process a chunk of gRNAs in parallel
    
    Args:
        chunk_data (tuple): (chunk_index, gRNA_chunk, adata_crispr, output_dir, seed, n_iter, progress_counters)
    
    Returns:
        List of tuples (gRNA name, perturbed cells, threshold, loss, estimates) for each gRNA in chunk
    '''
    chunk_idx, gRNA_chunk, adata_crispr, output_dir, seed, n_iter, progress_counters = chunk_data
    results = []
    for gRNA in gRNA_chunk:
        perturbed_cells, threshold, loss, map_estimates = fit_PGMM(gRNA, adata_crispr, output_dir, seed, n_iter)
        results.append((gRNA, perturbed_cells, threshold, loss, map_estimates))
        # Update progress counter for this chunk
        if progress_counters is not None:
            progress_counters[chunk_idx] += 1
    return results


def print_progress_report(start_time, total_gRNAs, progress_counters, chunk_sizes, stop_event, report_interval, mode="parallel"):
    '''
    Periodically print progress report every 30 seconds
    
    Args:
        start_time (datetime): when processing started
        total_gRNAs (int): total number of gRNAs to process
        progress_counters (list): shared list of progress counters for each process/chunk
        chunk_sizes (list): list of chunk sizes (number of gRNAs per chunk)
        stop_event (threading.Event): event to signal when to stop printing
        report_interval (int): interval in seconds between progress reports
        mode (str): "parallel" or "sequential"
    '''
    while not stop_event.wait(report_interval):  # Wait for report_interval seconds or until stop_event is set
        try:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            
            # Count total processed with timeout protection
            try:
                # Use list() to copy values quickly, avoiding potential blocking on Manager proxy
                counter_snapshot = list(progress_counters)
                total_processed = sum(counter_snapshot)
            except Exception as e:
                print(f"Warning: Could not read progress counters: {e}", flush=True)
                total_processed = 0
            
            # Calculate estimated completion
            if total_processed > 0:
                rate = total_processed / elapsed  # gRNAs per second
                remaining = total_gRNAs - total_processed
                eta_seconds = remaining / rate
                estimated_end = current_time + pd.Timedelta(seconds=eta_seconds)
                remaining_str = str(pd.Timedelta(seconds=int(eta_seconds)))
            else:
                estimated_end = "N/A"
                remaining_str = "N/A"
            
            # Format elapsed time
            elapsed_str = str(pd.Timedelta(seconds=int(elapsed)))
            
            # Print report with flush=True to ensure output in batch/containerized environments
            print(f"\n{'='*70}", flush=True)
            print(f"Progress Report - {current_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"Elapsed time: {elapsed_str}", flush=True)
            print(f"Estimated remaining: {remaining_str}", flush=True)
            print(f"Estimated completion: {estimated_end if isinstance(estimated_end, str) else estimated_end.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"Total progress: {total_processed}/{total_gRNAs} gRNAs ({100*total_processed/total_gRNAs:.1f}%)", flush=True)
            
            if mode == "parallel" and len(counter_snapshot) > 1:
                print(f"Per-process counts:", flush=True)
                for i, (count, total) in enumerate(zip(counter_snapshot, chunk_sizes)):
                    print(f"  Process {i+1:>3}: {count:>5}/{total:>5} gRNAs completed ({100*count/total if total > 0 else 0:.1f}%)", flush=True)
            
            print(f"{'='*70}\n", flush=True)
        except Exception as e:
            # If progress reporting fails, print error but keep thread alive
            print(f"Error in progress reporting: {e}", flush=True)
            print(f"Progress monitoring will continue...", flush=True)


def ga_poisson_gauss(input_file, output_dir, start_gRNA = 0, step = None, n_iter = 500, n_counts = None, UMI_threshold = 0,
                     parallelize = True, n_processes = None, report_interval_seconds = 30):
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
        n_processes (int, optional): specifies number of processes to use for parallelization if parallelize = True. If set to None (default), all available CPUs will be used.
        report_interval_seconds (int, optional): interval in seconds between progress reports (default = 30)
        
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
        print(f'{n_processes} parallel processes used')
        
        # Split gRNAs into exactly n_processes chunks
        # Distribute gRNAs as evenly as possible across chunks
        chunk_size = len(gRNA_list) // n_processes
        remainder = len(gRNA_list) % n_processes
        
        gRNA_chunks = []
        start = 0
        for i in range(n_processes):
            # First 'remainder' chunks get one extra gRNA
            end = start + chunk_size + (1 if i < remainder else 0)
            gRNA_chunks.append(gRNA_list[start:end])
            start = end
        
        # Create shared progress counters
        manager = multiprocessing.Manager()
        progress_counters = manager.list([0] * len(gRNA_chunks))
        chunk_sizes = [len(chunk) for chunk in gRNA_chunks]
        
        # Prepare chunk data with progress tracking
        chunk_data_list = [
            (i, chunk, adata_crispr, output_dir, 2024, n_iter, progress_counters)
            for i, chunk in enumerate(gRNA_chunks)
        ]
        
        # Start progress monitoring in background thread
        start_time = datetime.now()
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=print_progress_report,
            args=(start_time, len(gRNA_list), progress_counters, chunk_sizes, stop_event, report_interval_seconds, "parallel")
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Process chunks in parallel using multiprocessing.Pool
        print(f'Processing {len(gRNA_list)} gRNAs in {len(gRNA_chunks)} chunks')
        print(f'Progress will be reported every {report_interval_seconds} seconds...\n')
        
        with multiprocessing.Pool(processes=n_processes) as pool:
            chunk_results = pool.map(process_gRNA_chunk, chunk_data_list)
        
        # Stop progress monitoring
        stop_event.set()
        monitor_thread.join(timeout=1)
        
        # Print final report
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nCompleted processing {len(gRNA_list)} gRNAs in {pd.Timedelta(seconds=int(elapsed))}")
        
        # Flatten results from all chunks
        results = [item for chunk in chunk_results for item in chunk]
        
        # Combine the results per gRNA
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
        # Create progress counter for sequential mode
        manager = multiprocessing.Manager()
        progress_counters = manager.list([0])
        chunk_sizes = [len(gRNA_list)]
        
        # Start progress monitoring in background thread
        start_time = datetime.now()
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=print_progress_report,
            args=(start_time, len(gRNA_list), progress_counters, chunk_sizes, stop_event, report_interval_seconds, "sequential")
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print(f'Processing {len(gRNA_list)} gRNAs sequentially')
        print(f'Progress will be reported every {report_interval_seconds} seconds...\n')
        
        for gRNA in gRNA_list:
            perturbed_cells, threshold, loss, map_estimates = fit_PGMM(gRNA, adata_crispr, output_dir, 2024, n_iter)
            progress_counters[0] += 1
            
            if len(perturbed_cells) != 0:
                # get UMI_counts of assigned cells
                UMI_counts = adata_crispr[perturbed_cells, [gRNA]].X.toarray().reshape(-1)
                df = pd.DataFrame({'cell': perturbed_cells, 'gRNA': gRNA, 'UMI_counts': UMI_counts})
                perturbations = pd.concat([perturbations, df], ignore_index = True)
                thresholds = pd.concat([thresholds, pd.DataFrame({'gRNA': [gRNA], 'threshold': [threshold]})])
                losses = pd.concat([losses, pd.DataFrame({'gRNA': [gRNA], 'loss': [loss]})])
                estimates = pd.concat([estimates, map_estimates])
        
        # Stop progress monitoring
        stop_event.set()
        monitor_thread.join(timeout=1)
        
        # Print final report
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nCompleted processing {len(gRNA_list)} gRNAs in {pd.Timedelta(seconds=int(elapsed))}")
    
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
