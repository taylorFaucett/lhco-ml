import numpy as np



def oversample(X, y):
    targets = list(set(y))
    sizes = []
    for target in targets:
        sizes.append(len(y[y==target]))
    biggest = max(sizes)
    for target in targets:
        if len(y[y==target]) < biggest:
            X_reduced = X[y==target]
            X_replacement = np.resize(X_reduced, ((biggest, X_reduced.shape[-2], X_reduced.shape[-1])))
            y_replacement = np.resize(y[y==target], biggest)
            X = np.vstack((X, X_replacement)) 
            y = np.hstack((y, y_replacement))
    return X, y

def balance(X, y):
    targets = list(set(y))
    sizes = []
    for target in targets:
        sizes.append(len(y[y==target]))
    smallest = min(sizes)
    for target in targets:
        if target == 0:
            X_new = X[y==target][:smallest]
            y_new = y[y==target][:smallest]
        else:
            X_new = np.vstack((X_new, X[y==target][:smallest]))
            y_new = np.hstack((y_new, y[y==target][:smallest]))
    return X_new, y_new
            
