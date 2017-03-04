#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        @Return: a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    # Get the errors of predictions to net_worth
    errors = abs(predictions - net_worths)

    sorted_errors = sorted(errors)

    # Get the lowest 90% of errors, cut off highest 10% of errors
    percent_of_errors = sorted_errors[ : int(len(sorted_errors) * 0.9)]

    # Store all age, net_worth, error values in list if error is in lower 90%
    for i in range(len(errors)):
        if errors[i] <= percent_of_errors[-1]:
                cleaned_data.append((ages[i], net_worths[i], errors[i]))
    
    return cleaned_data

