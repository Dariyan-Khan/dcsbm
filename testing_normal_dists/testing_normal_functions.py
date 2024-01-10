import numpy as np
from scipy.stats import beta
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import tqdm

np.random.seed(45)


def find_delta_inv(μ_1, μ_2, exp_rho):
    μ_1_outer = np.outer(μ_1, μ_1)
    μ_2_outer = np.outer(μ_2, μ_2)

    Δ = exp_rho**2 * (1 / 2) * (μ_1_outer + μ_2_outer)

    Δ_inv = np.linalg.inv(Δ) 
    return Δ_inv

def exp_X1_inner_func(x, ρ, μ):
    return (np.dot(x, ρ*μ) - (np.dot(x, ρ*μ)**2)) * np.outer(ρ*μ, ρ*μ)

def covariance_estimate(x, μ_1, μ_2, prior, exp_rho, N_ρ=1000, N_t=1000):
    ρ_samples_1 = np.array([prior() for _ in range(N_ρ)])
    ρ_samples_2 = np.array([prior() for _ in range(N_ρ)])
    μ_1_integral_estimate = (1 / N_ρ) * sum(exp_X1_inner_func(x, ρ, μ_1) for ρ in ρ_samples_1)
    μ_2_integral_estimate = (1 / N_ρ) * sum(exp_X1_inner_func(x, ρ, μ_2) for ρ in ρ_samples_2)
    exp_X1_func_estimate = 0.5 * (μ_1_integral_estimate + μ_2_integral_estimate)
    Δ_inv = find_delta_inv(μ_1, μ_2, exp_rho)
    return (Δ_inv @ exp_X1_func_estimate @ Δ_inv) / N_t

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def simulate_adj_mat(prior, μ_1, μ_2):
    μ_mat = np.stack((μ_1, μ_2), axis=1)
    N_t = 1000
    bern_params = [(prior(), np.random.randint(0,2)) for _ in range(N_t)]
    adj_mat = np.zeros((N_t, N_t))

    for i in range(N_t):
        ρ_i, μ_i = bern_params[i][0], μ_mat[:, bern_params[i][1]]
        for j in range(i):
            ρ_j, μ_j = bern_params[j][0], μ_mat[:, bern_params[j][1]]

            adj_mat[i,j] = np.random.binomial(1, ρ_i * ρ_j * np.dot(μ_i, μ_j))

            adj_mat[j,i] = adj_mat[i,j]
        
        adj_mat[i,i] = 1
    
    assert check_symmetric(adj_mat)

    return adj_mat, bern_params

def spectral_emb(μ_1, μ_2, prior, N_t=1000):
    μ_mat = np.stack((μ_1, μ_2), axis=1)
    adj_mat, bern_params = simulate_adj_mat(prior, μ_1, μ_2)
    eigvals, eigvecs = np.linalg.eig(adj_mat)
    sorted_indexes = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[sorted_indexes]
    eigvecs = eigvecs[:,sorted_indexes]
    embedding_dim = len(μ_1)
    eigvecs_trunc = eigvecs[:,:2]
    eigvals_trunc = np.diag(np.sqrt(np.abs(eigvals[:2])))
    spectral_emb = eigvecs_trunc @ eigvals_trunc
    true_means = np.zeros((N_t, 2))

    for i in range(N_t):
        ρ_i, μ_i = bern_params[i][0], μ_mat[:, bern_params[i][1]]
        true_means[i, :] =  ρ_i * μ_i

    best_orthog_mat = orthogonal_procrustes(spectral_emb, true_means)
    spectral_emb = spectral_emb @ best_orthog_mat[0]
    return spectral_emb

def clt_sample(prior, μ_1, μ_2, exp_rho, N_t=1000):
    group_1_samples = []
    ρ_1_samples = []

    group_mean = μ_1

    for _ in tqdm.tqdm(range(N_t)):
        # sample the ρ's
        ρ = prior()
        group_1_samples.append(np.random.multivariate_normal(ρ*group_mean,        
                            covariance_estimate(ρ*group_mean, μ_1, 
                                                μ_2, prior, exp_rho)))

        ρ_1_samples.append(ρ)

    group_1_samples = np.array(group_1_samples)
    ρ_1_samples = np.array(ρ_1_samples)
        
    # group 2
    group_2_samples = []
    ρ_2_samples = []

    group_mean = μ_2
    for _ in tqdm.tqdm(range(N_t)):
        # sample the ρ's
        ρ = prior()
        group_2_samples.append(np.random.multivariate_normal(ρ*group_mean,
                            covariance_estimate(ρ*group_mean, μ_1, 
                                                μ_2, prior, exp_rho)))
        ρ_2_samples.append(ρ)

    group_2_samples = np.array(group_2_samples)
    ρ_2_samples = np.array(ρ_2_samples)
    all_hat_samples = np.concatenate((group_1_samples, group_2_samples), axis=0)
    all_hat_ρ = np.concatenate((ρ_1_samples,ρ_2_samples), axis=0)

    true_hat_means = np.zeros((2*N_t, 2))

    for i in tqdm.tqdm(range(N_t)):
        true_hat_means[i, :] =  all_hat_ρ[i] * μ_1 # ρ_1_samples

    for i in tqdm.tqdm(range(N_t,2*N_t)):
        true_hat_means[i, :] =  all_hat_ρ[i] * μ_2
    
    best_orthog_mat_hat = orthogonal_procrustes(all_hat_samples, true_hat_means)
    all_hat_samples = all_hat_samples @ best_orthog_mat_hat[0]
    return all_hat_samples

def exp_matrix_A_func(x, ρ, μ):
    return (np.dot(x, ρ*μ)) * np.outer(ρ*μ, ρ*μ)

def exp_matrix_B_func(x, ρ, μ):
    return (np.dot(x, ρ*μ)**2) * np.outer(ρ*μ, ρ*μ)

def mvn_assump_samples(matrix_func_A, matrix_func_B, μ_1, μ_2, prior, exp_rho, second_mom_rho, N_ρ=1000, N_t=1000):
    ρ_samples_1 = np.array([prior() for _ in range(N_ρ)])
    ρ_samples_2 = np.array([prior() for _ in range(N_ρ)])
    μ_1_integral_estimate = lambda x, mat_func: (1 / N_ρ) * sum(mat_func(x, ρ, μ_1) for ρ in ρ_samples_1)
    μ_2_integral_estimate = lambda x, mat_func: (1 / N_ρ) * sum(mat_func(x, ρ, μ_2) for ρ in ρ_samples_2)
    exp_matrix_innner_func_estimate =  lambda x, mat_func: 0.5 * (μ_1_integral_estimate(x, mat_func) + μ_2_integral_estimate(x, mat_func))
    Δ_inv = find_delta_inv(μ_1, μ_2, exp_rho)
    cov_matrix_func_estimate = lambda x, mat_func: (Δ_inv @ exp_matrix_innner_func_estimate(x,mat_func) @ Δ_inv) / N_t

    A_dash_1 = cov_matrix_func_estimate(μ_1, matrix_func_A)
    B_dash_1 = cov_matrix_func_estimate(μ_1, matrix_func_B)

    mvn_cov_1 = (A_dash_1 * exp_rho) - (B_dash_1 * second_mom_rho)

    A_dash_2 = cov_matrix_func_estimate(μ_2, matrix_func_A)
    B_dash_2 = cov_matrix_func_estimate(μ_2, matrix_func_B)

    mvn_cov_2 = (A_dash_2 * exp_rho) - (B_dash_2 * second_mom_rho)

    mvn_1_assumption_samples = [np.random.multivariate_normal(0.5 * μ_1, mvn_cov_1) for _ in range(1000)]
    mvn_2_assumption_samples = [np.random.multivariate_normal((0.5 * μ_2), mvn_cov_2) for _ in range(1000)]

    mvn1_x = [mvn_sample[0] for mvn_sample in mvn_1_assumption_samples]
    mvn1_y = [mvn_sample[1] for mvn_sample in mvn_1_assumption_samples]

    mvn2_x = [mvn_sample[0] for mvn_sample in mvn_2_assumption_samples]
    mvn2_y = [mvn_sample[1] for mvn_sample in mvn_2_assumption_samples]

    return mvn1_x, mvn1_y, mvn2_x, mvn2_y

def mvn_assump_samples_wrapper(μ_1, μ_2, prior, exp_rho, second_mom_rho, N_ρ=1000, N_t=1000):
    return mvn_assump_samples(exp_matrix_A_func, exp_matrix_B_func, μ_1, 
                              μ_2, prior, exp_rho, second_mom_rho, N_ρ=N_ρ,
                              N_t=N_t)


def exp_X1_inner_func_assump(x, ρ, exp_rho, μ):
    return ((np.dot(x, ρ*μ) / exp_rho) - (np.dot(x, ρ*μ)**2)) * np.outer(ρ*μ, ρ*μ)

def covariance_under_assump(ρ, μ_1, μ_2, prior, exp_rho, N_ρ=1000, N_t=1000):
    ρ_samples_1 = np.array([prior() for _ in range(N_ρ)])
    ρ_samples_2 = np.array([prior() for _ in range(N_ρ)])
    μ_1_integral_estimate = (1 / N_ρ) * sum(exp_X1_inner_func_assump(μ_1, ρ, exp_rho, μ_1) for ρ in ρ_samples_1)
    μ_2_integral_estimate = (1 / N_ρ) * sum(exp_X1_inner_func_assump(μ_2, ρ, exp_rho,  μ_2) for ρ in ρ_samples_2)
    exp_X1_func_estimate_assump = 0.5 * (μ_1_integral_estimate + μ_2_integral_estimate)  # CHECK THIS LINE
    Δ_inv = find_delta_inv(μ_1, μ_2, exp_rho)
    covariance_estimate_assump = ρ**2 * (Δ_inv @ exp_X1_func_estimate_assump @ Δ_inv) / N_t
    return covariance_estimate_assump

def samples_under_assump(μ_1, μ_2, prior, exp_rho, N_ρ=1000, N_t=1000):
    # group 1
    group_1_samples_assump = []
    ρ_1_samples_assump = []

    group_mean = μ_1
    for i in tqdm.tqdm(range(N_t)):
        # sample the ρ's
        ρ = prior()
        group_1_samples_assump.append(np.random.multivariate_normal(ρ*group_mean,
                            covariance_under_assump(ρ, μ_1, μ_2, prior,
                                                    exp_rho)))

        ρ_1_samples_assump.append(ρ)

    group_1_samples_assump = np.array(group_1_samples_assump)
    ρ_1_samples_assump = np.array(ρ_1_samples_assump)
    
    # group 2

    group_2_samples_assump = []
    ρ_2_samples_assump = []

    group_mean = μ_2
    for i in tqdm.tqdm(range(N_t)):
        # sample the ρ's
        ρ = prior()

        group_2_samples_assump.append(np.random.multivariate_normal(ρ*group_mean,
                            covariance_under_assump(ρ, μ_1, μ_2, prior,
                                                    exp_rho)))
        
        ρ_2_samples_assump.append(ρ)

    group_2_samples_assump = np.array(group_2_samples_assump)
    ρ_2_samples_assump = np.array(ρ_2_samples_assump)

    all_hat_samples_assump = np.concatenate((group_1_samples_assump, group_2_samples_assump), axis=0)
    all_hat_ρ_assump = np.concatenate((ρ_1_samples_assump,ρ_2_samples_assump), axis=0)

    true_hat_means_assump = np.zeros((2*N_t, 2))

    for i in tqdm.tqdm(range(N_t)):
        true_hat_means_assump[i, :] =  ρ_1_samples_assump[i] * μ_1

    for i in tqdm.tqdm(range(N_t,2*N_t)):
        true_hat_means_assump[i, :] =  all_hat_ρ_assump[i] * μ_2


    best_orthog_mat_hat_assump = orthogonal_procrustes(all_hat_samples_assump, true_hat_means_assump)
    all_hat_samples_assump = all_hat_samples_assump @ best_orthog_mat_hat_assump[0]
    return all_hat_samples_assump

def plot_spherical_data(zipped_data, μ_1, μ_2, title):
    colours = ["red", "green", "blue", "orange", "purple", "black", "grey"]
    for x, y in zipped_data:
        plt.scatter(x,y, color=colours.pop(0))

    xrange = np.linspace(-3, 3, 1000)

    plt.scatter(xrange, (μ_1[1] / μ_1[0])*xrange, color=colours.pop(0))
    plt.scatter(xrange, (μ_2[1]/μ_2[0])*xrange, color=colours.pop(0))

    plt.xlim(-0.5, 1)
    plt.ylim(-0.5, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()


















        




