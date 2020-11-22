"""
Define all model classes following the definition of Abstract_Model.
"""
import downhill
import theano
import theano.tensor as TT


data_type = 'float32'
#Single precision:
theano.config.floatX = data_type
theano.config.mode = 'FAST_RUN' #  'Mode', 'ProfileMode'(deprecated), 'DebugMode', 'FAST_RUN', 'FAST_COMPILE'
theano.config.exception_verbosity = 'high'

from .tools import *
from .batching import * 
from .evaluation import *



class Abstract_Model(object):

	def __init__(self):
		self.name = self.__class__.__name__
		
		#Symbolic expressions for the prediction function (and compiled one too), the loss, the regularization, and the loss to
		#optimize (loss + lmbda * regul)
		#To be defined by the child classes:
		self.pred_func = None
		self.pred_func_compiled = None

		self.loss_func = None
		self.regul_func = None
		self.loss_to_opt = None
		
		#Symbolic variables for training values
		self.ys = TT.vector('ys')
		self.rows = TT.lvector('rows')
		self.cols = TT.lvector('cols')
		self.tubes = TT.lvector('tubes') 


		#Current values for which the loss is currently compiled
		#3 dimensions:
		self.n = 0 #Number of subject entities
		self.m = 0 #Number of relations
		self.l = 0 #Number of object entities
		#and rank:
		self.k = 0
		#and corresponding number of parameters (i.e. n*k + m*k + l*k for CP_Model)
		self.nb_params = 0



	def set_dims(self,train_triples, hparams):
		self.n = max(train_triples.indexes[:,0]) +1
		self.m = max(train_triples.indexes[:,1]) +1
		self.l = max(train_triples.indexes[:,2]) +1
		self.k = hparams.embedding_size


	def get_pred_symb_vars(self):
		"""
		Returns the default pred parameters. Made to be overriden by child classes if need of additional input variables
		"""
		pred_inputs=[self.rows, self.cols, self.tubes]

		return pred_inputs

	def get_pred_args(self,test_idxs):
		"""
		Returns the default pred symbolic variables. Made to be overriden by child classes if need of additional input variables
		"""

		return [test_idxs[:,0], test_idxs[:,1], test_idxs[:,2]]


	def get_loss_args_and_symb_vars(self, train_triples, valid_triples, hparams):
		"""
		Returns the default loss parameters and corresponding symbolic variables. Made to be overriden by child classes if need of additional input variables
		"""

		train = Batch_Loader(train_triples, n_entities = max(self.n,self.l), batch_size = hparams.batch_size, neg_ratio = hparams.neg_ratio, contiguous_sampling = hparams.contiguous_sampling ) 
		inputs=[self.ys, self.rows, self.cols, self.tubes]
		if valid_triples is not None:
			valid = Batch_Loader(valid_triples, n_entities = max(self.n,self.l), batch_size = hparams.batch_size, neg_ratio = hparams.neg_ratio, contiguous_sampling = hparams.contiguous_sampling ) 
			#valid = [valid_triples.values[:], valid_triples.indexes[:,0], valid_triples.indexes[:,1], valid_triples.indexes[:,2]]
		else:
			valid = None

		return train, inputs, valid

	def allocate_params(self):
		nb_params=0
		#Call child class getter of initial values of model parameters:
		params = self.get_init_params()
		#And allocate them as theano shared variables
		for name, val in params.items():
			setattr(self, name, theano.shared(val, name = name) )
			nb_params += val.size
		self.nb_params = nb_params

	def reinit_params(self):
		nb_params=0
		#Call child class getter of initial values of model parameters:
		params = self.get_init_params()
		#And set their values
		for name, val in params.items():
			getattr(self, name).set_value(val, borrow  = True)
			nb_params += val.size
		self.nb_params = nb_params




	def setup_params_for_train(self,train_triples, valid_triples, hparams, redefine_loss = False):
		"""
		Calls teh define_loss function that gives the model loss to minimize
		"""


		#Check if need to redefine the loss function or not:
		if redefine_loss or self.loss_to_opt is None:

			#Initialize parameters (child class overriding):
			self.allocate_params()

			#Defining the model (child class overriding):
			self.define_loss()

			#Compile the prediction functions:
			self.pred_func_compiled = theano.function(self.get_pred_symb_vars(), self.pred_func)

		else:
			#Just reinit the params
			self.reinit_params()

		#Combine loss and regularization to get what downhill will optimize:
		#Changing the scalar value lmbda in the function actually doesn't make theano to recompile everything, it's as fast as not changing anything
		self.loss_to_opt = self.loss + hparams.lmbda * self.regul_func


	def fit(self, train_triples, valid_triples, hparams, n=0,m=0,l=0, scorer = None):

		#Set input_dimensions:
		if n == 0: #No given dimensions, can be useful for transparent predicton of entities/rels not seen in train
			self.set_dims(train_triples, hparams)
		else:
			self.n, self.m, self.l, self.k = n, m, l, hparams.embedding_size

		#Define the downhill loss corresponding to the input dimensions
		self.setup_params_for_train(train_triples, valid_triples, hparams)
		
		#get the loss inputs:
		train_vals, train_symbs, valid_vals = self.get_loss_args_and_symb_vars(train_triples, valid_triples, hparams)

		opt = downhill.build(hparams.learning_rate_policy, loss=self.loss_to_opt, inputs=train_symbs, monitor_gradients=True)

		train_vals = downhill.Dataset(train_vals, name = 'train')


		#Main SGD loop
		it = 0
		best_valid_mrr = -1
		best_valid_ap = -1
		for tm, vm in opt.iterate(train_vals, None,
				max_updates=hparams.max_iter,
				validate_every=9999999, 				#I take care of the valiation, with validation metrics instead of loss
				patience=9999999,						#Number of tolerated imporvements of validation loss that are inferior to min_improvement
				max_gradient_norm=1,          			# Prevent gradient explosion!
				learning_rate=hparams.learning_rate):


			if it % hparams.valid_scores_every == 0 and scorer is not None:
				if valid_triples is not None:
					logger.info("Validation metrics:")
					res = scorer.compute_scores(self, self.name, hparams, valid_triples)
					cv_res = CV_Results()
					cv_res.add_res(res, self.name, hparams.embedding_size, hparams.lmbda, self.nb_params)


					if scorer.compute_ranking_scores:
						metrics = cv_res.print_MRR_and_hits()
						
						#Early stopping on filtered MRR
						if best_valid_mrr >= metrics[self.name][2]:
							logger.info("Validation filtered MRR decreased, stopping here.")
							break
						else:
							best_valid_mrr = metrics[self.name][2]
					else:
						logger.info("Validation AP: " + str(res.ap))
						#Early stopping on Average Precision
						if best_valid_ap >= res.ap:
							logger.info("Validation AP decreased, stopping here.")
							break
						else:
							best_valid_ap = res.ap

			it += 1
			if it >= hparams.max_iter: #Avoid downhill resetting the parameters when max_iter is reached
				break




	def predict(self, test_idxs):
		"""
		test_idxs is a 2D numpy array of size [a,3], containing the indexes of the test triples.
		Returns a vector of size a.
		"""

		return self.pred_func_compiled( *self.get_pred_args(test_idxs) )


################ Abstract Functions: ('must' be defined by child classes)


	def get_init_params(self,n,m,l,k):
		"""
		Abstract. Returns a dict of the initial values of shared variables, indexes by the class attribute name (string).
		"""
		pass



	def define_loss(self):
		"""
		Abstract. Define the loss of the model in the child class for a given input (theano shared variables
		must have the correct dimensions at loss definition). Implies an initialization of the parameters
		(shared variables), and the initialization of three functions (prediction, loss and regularization) 
		as symbolic theano expressions: self.pred_func, self.loss_func and self.regul_func (mandatory).
		"""
		pass





#################################################################################################
# Definition of models (child classes):
#################################################################################################

class HTransRESCAL_Model(Abstract_Model):
	"""
	Rescal model
	"""

	def __init__(self):
		super(HTransRESCAL_Model, self).__init__()
		
		self.name = self.__class__.__name__
		
		self.r = None
		self.e = None
		self.R = None

	def get_init_params(self):

		params = { 'r' : randn(self.m, self.k),
				   'R' : randn(self.m, self.k, self.k),
				   'e' : randn(max(self.n,self.l),self.k)  }
		return params


	def define_loss(self):

		self.pred_func = TT.sum( TT.sum((self.e[self.rows,:,None] + self.r[self.cols,:,None]) * self.R[self.cols,:,:], 1) *  (self.e[self.tubes,:] - self.r[self.cols,:]), 1)

		self.loss = TT.sqr(self.ys - self.pred_func).mean()

		self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
						+ TT.sqr(self.R[self.cols,:,:]).mean() \
						+ TT.sqr(self.e[self.tubes,:]).mean() 



class HTransRESCAL_Logistic_Model(HTransRESCAL_Model):
	"""
	Rescal model with logistic loss
	"""

	def __init__(self):
		super(HTransRESCAL_Logistic_Model, self).__init__()
		
		self.name = self.__class__.__name__


	def define_loss(self):

		self.pred_func = TT.nnet.sigmoid( TT.sum( TT.sum((self.e[self.rows,:,None] + self.r[self.cols,:,None]) * self.R[self.cols,:,:] , 1) *  (self.e[self.tubes,:] - self.r[self.cols,:]), 1) )

		self.loss = TT.nnet.softplus( - self.ys * TT.sum( TT.sum((self.e[self.rows,:,None] + self.r[self.cols,:,None]) * self.R[self.cols,:,:], 1) *  (self.e[self.tubes,:] - self.r[self.cols,:]), 1)).mean()

		self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
						+ TT.sqr(self.R[self.cols,:,:]).mean() \
						+ TT.sqr(self.e[self.tubes,:]).mean() \
						+ TT.sqr(self.r[self.cols,:]).mean()

class TTransRESCAL_Model(Abstract_Model):
	"""
	Rescal model
	"""

	def __init__(self):
		super(TTransRESCAL_Model, self).__init__()
		
		self.name = self.__class__.__name__
		
		self.r = None
		self.e = None
		self.R = None

	def get_init_params(self):

		params = { 'r' : randn(self.m, self.k),
				   'R' : randn(self.m, self.k, self.k),
				   'e' : randn(max(self.n,self.l),self.k)  }
		return params


	def define_loss(self):

		self.pred_func = TT.sum( TT.sum((self.e[self.rows,:,None] + self.r[self.cols,:,None]) * self.R[self.cols,:,:], 1) *  (self.e[self.tubes,:] - self.r[self.cols,:]), 1)

		self.loss = TT.sqr(self.ys - self.pred_func).mean()

		self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
						+ TT.sqr(self.R[self.cols,:,:]).mean() \
						+ TT.sqr(self.e[self.tubes,:]).mean() 



class TTransRESCAL_Logistic_Model(TTransRESCAL_Model):
	"""
	Rescal model with logistic loss
	"""

	def __init__(self):
		super(TTransRESCAL_Logistic_Model, self).__init__()
		
		self.name = self.__class__.__name__


	def define_loss(self):

		self.pred_func = TT.nnet.sigmoid( TT.sum( TT.sum((self.e[self.rows,:,None] - self.r[self.cols,:,None]) * self.R[self.cols,:,:] , 1) *  (self.e[self.tubes,:] + self.r[self.cols,:]), 1) )

		self.loss = TT.nnet.softplus( - self.ys * TT.sum( TT.sum((self.e[self.rows,:,None] - self.r[self.cols,:,None]) * self.R[self.cols,:,:], 1) *  (self.e[self.tubes,:] + self.r[self.cols,:]), 1)).mean()

		self.regul_func = 0.1 * TT.sqr(self.e[self.rows,:]).mean() \
						+ 0.5 * TT.sqr(self.R[self.cols,:,:]).mean() \
						+ 0.1 * TT.sqr(self.e[self.tubes,:]).mean()  \
						+ 0.1 * TT.sqr(self.r[self.cols,:]).mean()