import pandas as pd

def convert_wtl_errors(wtl_errors=None):
    if wtl_errors is None:
        wtl_errors = cPickle.load(open("data/wtl-errors.pkl"))
    print(wtl_errors.keys())
    counts = wtl_errors['counts']
    corrections = wtl_errors['corrections']
    contexts = wtl_errors['contexts']
    rows = []
    for error in corrections.keys():
        correction = corrections[error]
        count = counts[error]
        context = contexts[error][0]
        rows.append((count,error,correction,context))
    return pd.DataFrame(data=rows, columns=['Count', 'Error', 'Correction', 'Context'])
