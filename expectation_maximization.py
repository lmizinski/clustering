#https://github.com/ajcr/em-explanation/blob/master/em-notebook-2.ipynb
#https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/
#ipython make notebook trusted
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

#print("Hello, World!")
#print(stats.norm(50, 15).pdf([61, 84]))

np.random.seed(110) # for reproducible results

# set the parameters for red and blue distributions we will draw from
red_mean = 3
red_std = 0.8

blue_mean = 7
blue_std = 2

# draw 20 samples from each normal distribution
red = np.random.normal(red_mean, red_std, size=20)
blue = np.random.normal(blue_mean, blue_std, size=20)
both_colours = np.sort(np.concatenate((red, blue))) # array with every sample point (for later use)

plt.rcParams['figure.figsize'] = (15, 2)
plt.plot(red, np.zeros_like(red), '.', color='r', markersize=10);
plt.plot(blue, np.zeros_like(blue), '.', color='b', markersize=10);
plt.title(r'Distribution of red and blue points (known colours)', fontsize=17);
plt.yticks([]);
plt.show();

#np.mean(red)
#np.std(red)
#np.mean(blue)
#np.std(blue)

plt.rcParams['figure.figsize'] = (15, 2)
plt.plot(both_colours, np.zeros_like(both_colours), '.', color='purple', markersize=10);
plt.title(r'Distribution of red and blue points (hidden colours)', fontsize=17);
plt.yticks([]);
plt.show();

#stats.norm(50, 15).pdf([61, 84])



def weight_of_colour(colour_likelihood, total_likelihood):
    """
    Compute the weight for each colour at each data point.
    """
    return colour_likelihood / total_likelihood

def estimate_mean(data, weight):
    """
    For each data point, multiply the point by the probability it
    was drawn from the colour's distribution (its "weight").
    
    Divide by the total weight: essentially, we're finding where 
    the weight is centred among our data points.
    """
    return np.sum(data * weight) / np.sum(weight)

def estimate_std(data, weight, mean):
    """
    For each data point, multiply the point's squared difference
    from a mean value by the probability it was drawn from
    that distribution (its "weight").
    
    Divide by the total weight: essentially, we're finding where 
    the weight is centred among the values for the difference of
    each data point from the mean.
    
    This is the estimate of the variance, take the positive square
    root to find the standard deviation.
    """
    variance = np.sum(weight * (data - mean)**2) / np.sum(weight)
    return np.sqrt(variance)
    
def plot_guesses(red_mean_guess, blue_mean_guess, red_std_guess, blue_std_guess, alpha=1):
    """
    Plot bell curves for the red and blue distributions given guesses for mean and standard deviation.
    
    alpha : transparency of the plotted curve
    """
    # set figure size and plot the purple dots
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.plot(both_colours, np.zeros_like(both_colours), '.', color='purple', markersize=10)
       
    # compute the size of the x axis
    lo = np.floor(both_colours.min()) - 1
    hi = np.ceil(both_colours.max()) + 1
    x = np.linspace(lo, hi, 500)
    
    # plot the bell curves
    plt.plot(x, stats.norm(red_mean_guess, red_std_guess).pdf(x), color='r', alpha=alpha)
    plt.plot(x, stats.norm(blue_mean_guess, blue_std_guess).pdf(x), color='b', alpha=alpha)
    
    # vertical dotted lines for the mean of each colour - find the height
    # first (i.e. the probability of the mean of the colour group)
    r_height = stats.norm(red_mean_guess, red_std_guess).pdf(red_mean_guess)
    b_height = stats.norm(blue_mean_guess, blue_std_guess).pdf(blue_mean_guess)
    
    plt.vlines(red_mean_guess, 0, r_height, 'r', '--', alpha=alpha)
    plt.vlines(blue_mean_guess, 0, b_height, 'b', '--', alpha=alpha);
    
# estimates for the mean
red_mean_guess = 1.1
blue_mean_guess = 9

# estimates for the standard deviation
red_std_guess = 2
blue_std_guess = 1.7

N_ITER = 20 # number of iterations of EM

alphas = np.linspace(0.1, 1, N_ITER) # transparency of curves to plot for each iteration

for i in range(N_ITER):
    
    ## Expectation step
    ## ----------------

    likelihood_of_red = stats.norm(red_mean_guess, red_std_guess).pdf(both_colours)
    likelihood_of_blue = stats.norm(blue_mean_guess, blue_std_guess).pdf(both_colours)

    red_weight = weight_of_colour(likelihood_of_red, likelihood_of_red+likelihood_of_blue)
    blue_weight = weight_of_colour(likelihood_of_blue, likelihood_of_red+likelihood_of_blue)

    ## Maximisation step
    ## -----------------
    
    # N.B. it should not ultimately matter if compute the new standard deviation guess
    # before or after the new mean guess
    
    red_std_guess = estimate_std(both_colours, red_weight, red_mean_guess)
    blue_std_guess = estimate_std(both_colours, blue_weight, blue_mean_guess)

    red_mean_guess = estimate_mean(both_colours, red_weight)
    blue_mean_guess = estimate_mean(both_colours, blue_weight)

    plot_guesses(red_mean_guess, blue_mean_guess, red_std_guess, blue_std_guess, alpha=alphas[i])

def printmd(string):
    display(Markdown(string));

plt.show();



md = """
|            | True Mean      | Estimated Mean | True Std.      | Estimated Std. | 
| :--------- |:--------------:| :------------: |:-------------: |:-------------: |
| Red        | {true_r_m:.5f} | {est_r_m:.5f}  | {true_r_s:.5f} | {est_r_s:.5f}  | 
| Blue       | {true_b_m:.5f} | {est_b_m:.5f}  | {true_b_s:.5f} | {est_b_s:.5f}  |
"""

printmd(
	md.format(
		true_r_m=np.mean(red),
		true_b_m=np.mean(blue),
		
		est_r_m=red_mean_guess,
		est_b_m=blue_mean_guess,
		
		true_r_s=np.std(red),
		true_b_s=np.std(blue),
		
		est_r_s=red_std_guess,
		est_b_s=blue_std_guess,
	)
)

display(HTML('<h1>Hello, world!</h1>'));
