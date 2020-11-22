#import scipy.io

import efe
from efe.exp_generators import *
import efe.tools as tools

if __name__ =="__main__":

	#Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
	wn18 = build_data(name = 'wn18',path = tools.cur_path + '/datasets/')


	#SGD hyper-parameters:
	params = Parameters(learning_rate = 0.5,
						max_iter = 1000,
						batch_size = int(len(wn18.train.values) / 100),  #Make 100 batches
						neg_ratio = 1, 
						valid_scores_every = 50,
						learning_rate_policy = 'adagrad',
						contiguous_sampling = False )

	
	all_params = { "Rescal_Logistic_Model" : params } ; emb_size = 100; lmbda =0.1
	tools.logger.info( "Learning rate: " + str(params.learning_rate))
	tools.logger.info( "Max iter: " + str(params.max_iter))
	tools.logger.info( "Generated negatives ratio: " + str(params.neg_ratio))
	tools.logger.info( "Batch size: " + str(params.batch_size))


	#Then call a local grid search, here only with one value of rank and regularization
	wn18.grid_search_on_all_models(all_params, embedding_size_grid = [emb_size], lmbda_grid = [lmbda], nb_runs = 1)

	#Print best averaged metrics:
	wn18.print_best_MRR_and_hits()

	#Print best averaged metrics per relation:
	wn18.print_best_MRR_and_hits_per_rel()
