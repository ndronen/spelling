# spelling

Evaluate the spelling corrections provided by the implementations in
spelling.dictionaries on the corpus of errors in 'data/aspell.dat':

    import spelling.mitton
    dfs = spelling.mitton.run('data/aspell.dat')
    spelling.mitton.evaluate(dfs)
