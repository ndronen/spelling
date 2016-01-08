import progressbar

def build_progressbar(items):
    return progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(items)).start()
