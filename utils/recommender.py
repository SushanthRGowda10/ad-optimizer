def generate_recommendation(ctr, budget):
    if ctr < 2:
        return "⚠ Low CTR: Try changing platform and increase budget"
    elif ctr < 5:
        return "⚡ Moderate CTR: Improve targeting strategy"
    else:
        return "✅ High CTR: Scale this campaign"