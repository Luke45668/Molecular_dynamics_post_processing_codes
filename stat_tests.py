

def chi_stat_test_uniform(sig,deg_f,f_obs):
    import scipy.stats
    from scipy.stats import chi2
    
    # comparing to uniform
    chi_stat=scipy.stats.chisquare(f_obs).statistic
    p_value=scipy.stats.chisquare(f_obs).pvalue
    chi_table=chi2.ppf(1-sig, deg_f)
    if chi_stat < chi_table: # not sure on this one 
            print("Test Passed, chi stat was ",chi_stat)
            print("Passing bound",chi_table)
            print("P value",p_value)
    else:
            print("Test failed, chi stat was ",chi_stat)
            print("Passing bound",chi_table)
            print("P value",p_value)
    return chi_stat,p_value

def chi_stat_test_custom(sig,deg_f,f_obs,f_exp):
    import scipy.stats
    from scipy.stats import chi2

         # comparing  to custom expected dist 
    chi_stat=scipy.stats.chisquare(f_obs,f_exp).statistic
    p_value=scipy.stats.chisquare(f_obs,f_exp).pvalue
    chi_table=chi2.ppf(1-sig, deg_f)
    if chi_stat < chi_table: # not sure on this one 
            print("Test Passed, chi stat was ",chi_stat)
            print("Passing bound",chi_table)
            print("P value",p_value)
    else:
            print("Test failed, chi stat was ",chi_stat)
            print("Passing bound",chi_table)
            print("P value",p_value)
    
    return chi_stat,p_value