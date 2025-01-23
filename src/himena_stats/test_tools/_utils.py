def pvalue_to_asterisks(pval: float) -> str:
    if pval > 0.05:
        return "n.s."
    elif pval > 0.01:
        return "*"
    elif pval > 0.001:
        return "**"
    elif pval > 0.0001:
        return "***"
    else:
        return "****"
