# spelling

Evaluate the spelling corrections provided by the implementations in
spelling.dictionaries on the corpus of errors in 'data/aspell.dat':

    import spelling.mitton
    dfs = spelling.mitton.build_mitton_datasets('data/aspell.dat')
    spelling.mitton.evaluate_ranks(dfs)
    evaluation = spelling.mitton.evaluate_ranks(dfs, ranks=[1])
    print(evaluation.sort_values('Accuracy'))
