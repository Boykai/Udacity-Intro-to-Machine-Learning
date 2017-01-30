#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    sorted_predictions = sorted(predictions)
    
    percent_of_predictions = sorted_predictions[ : int(len(sorted_predictions) * 0.9)]
    
    for i in range(len(predictions)):
        if i > len(percent_of_predictions):
            break
        else:
            if predictions[i] in percent_of_predictions:
                cleaned_data.append((ages[i], net_worths[i], predictions[i]))
    
    return cleaned_data

