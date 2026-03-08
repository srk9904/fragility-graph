def central_utility(data):
    """
    This is a 'hub' function. 
    Many other functions depend on it, so it should have HIGH FRAGILITY.
    """
    return data * 1.05

def calculation_engine(val):
    # Depends on central_utility
    result = central_utility(val)
    if result > 100:
        return result * 0.9
    return result

def data_processor(items):
    # Depends on central_utility
    return [central_utility(i) for i in items]

def final_report(val):
    # High-level function that depends on the engine
    report_val = calculation_engine(val)
    return f"Final Value: {report_val}"

# Note: Save this file to see the extension in action!
