import joblib
import spelling.train
import cPickle

if __name__ == '__main__':
    results = spelling.train.run_leave_out_one_dataset(n_jobs=20, verbose=True)
    cPickle.dump(results, open('results.pkl', 'w'))
