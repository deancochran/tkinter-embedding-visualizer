import pandas as pd
from scipy import stats
import itertools as it

def perform_2sided_ks_tests(data: pd.DataFrame, distribution_attr: str, sensitive_attr:str):
    """
    Description:
    perform_ks_tests returns a set of all permutations of the pairs of sensitive attributes,
    and their 2 sample KS statistics of the specified distribution in question

    For a 2-sided KS test, the null hypothesis is that the two distributions are identical, 
    F(x)=G(x) for all x; the alternative is that they are not identical.

    If the p-val < .05 reject the null hypothesis

    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr -  this is a float valued column in the data argument
    sensitive_attr - this is a categorical string valued column in the data argument

    Returns:
    dict() - {(attr_a, attr_b): (k_stat, p_val),...} for all combinations of categories in sensitive attribute column

    """
    results = {}
    sensitive_attr_categories = data[sensitive_attr].unique()
    assert len(sensitive_attr_categories) > 1

    for combination in it.combinations(sensitive_attr_categories, 2):
        (attr_a, attr_b) = combination
        attr_a_data = data[data[sensitive_attr]==attr_a][distribution_attr]
        attr_b_data = data[data[sensitive_attr]==attr_b][distribution_attr]
        results[combination] = stats.kstest(attr_a_data, attr_b_data)

    return results

def get_wasserstein_dist(data, distribution_attr, sensitive_attr, attr_a, attr_b):
    """Description:
    get_wasserstein_dist computes the distributional distance between two samples. 

    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr -  this is a float valued column in the data argument
    sensitive_attr - this is a categorical string valued column in the data argument
    attr_a = string name of the first sample (a sensitive attribute label)
    attr_b = string name of the second sample (a sensitive attribute label)

    Returns:
    distance: Int
    """
    attr_a_data = data[data[sensitive_attr]==attr_a][distribution_attr]
    attr_b_data = data[data[sensitive_attr]==attr_b][distribution_attr]
    distance = stats.wasserstein_distance(attr_a_data, attr_b_data)

    return distance


def analyze_2sided_ks_results(data: pd.DataFrame, distribution_attr: str, sensitive_attr:str):
    """
    Description:
    analyze_2sided_ks_results uses the output from perform_2sided_ks_tests to identify pairs of sensitive attributes,
    of the indicated distribution_attr, that are statistically different. 
    (looks for pairs whose p-val < .05 == reject the null hypothesis)

    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr -  this is a float valued column in the data argument
    sensitive_attr - this is a categorical string valued column in the data argument

    Returns:
    dict() - {(attr_a, attr_b): (k_stat, p_val),...} for all combinations of categories in sensitive attribute column

    """
    output={}
    results = perform_2sided_ks_tests(data, distribution_attr, sensitive_attr)
    
    for (attr_a, attr_b), (k_stat, p_val) in results.items():
        if p_val<.05:
            output[(attr_a, attr_b)]=get_wasserstein_dist(data, distribution_attr, sensitive_attr, attr_a, attr_b)

    return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

def analyze_sensitive_attr_distributions(data: pd.DataFrame, distribution_attr_list: list, sensitive_attr_list: list):
    """
    Description:
    analyze_sensitive_attr_distributions is a function that utilizes analyze_2sided_ks_results and perform_2sided_ks_tests
    to extend the opportunities of findings by aggregation all statistically different pairings of sensitive attributes, 
    for all the possible distributions that are given 
    
    Arguments:
    data - pandas Dataframe obj containing both the distribution_attr and the sensitive_attr in the column names
    distribution_attr_list - list of column names
    sensitive_attr_list - list of column names

    Returns:
    
    """
    results={}
    for (distr_attr, sens_attr) in it.product(distribution_attr_list,sensitive_attr_list): 
        output=analyze_2sided_ks_results(data,distr_attr,sens_attr)
        if len(output) > 0:
            results[(distr_attr,sens_attr)]=output
    return results