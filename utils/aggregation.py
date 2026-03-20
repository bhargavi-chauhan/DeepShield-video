import numpy as np

def aggregate_predictions(predictions):
    # predictions = list of probabilities
    avg = np.mean(predictions)
    
    if avg > 0.5:
        return "FAKE", avg
    else:
        return "REAL", avg


def temporal_variation(frames):
    diffs = []
    
    for i in range(len(frames)-1):
        diff = np.mean(abs(frames[i] - frames[i+1]))
        diffs.append(diff)
    
    return np.mean(diffs)