import progressbar

def build_progressbar(items):
    if isinstance(items, int):
        maxval = items
    else:
        maxval = len(items)
    return progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=maxval).start()
