import progressbar

def build_progressbar(items):
    if isinstance(items, int):
        max_value = items
    else:
        max_value = len(items)
    return progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        max_value=max_value).start()
