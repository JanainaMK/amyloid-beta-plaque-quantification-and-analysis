def get_significance_level(p_val):
    significance_level = ""
    if p_val < 0.001:
        significance_level += "*"
    if p_val < 0.01:
        significance_level += "*"
    if p_val < 0.05:
        significance_level += "*"
    return significance_level
