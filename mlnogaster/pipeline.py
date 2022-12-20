import numpy as np
from collections.abc import Iterable
import pandas as pd
import copy
import numbers
import datetime 
import editdistance
import scipy


from tpot import TPOTClassifier,TPOTRegressor

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.base import RegressorMixin,ClassifierMixin,TransformerMixin,check_array
from sklearn.neighbors import NearestNeighbors

from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

import pygad

from sklego.base import Clusterer

from .utils import inject_repr,prevFibonacci,nextFibonacci

class GeneticModelSelector(RegressorMixin,ClassifierMixin):
    
  def __init__(self,init_n_population: int=100, init_p_mutation=0.9,n_generations:int=5,task: str='classification',
              init_p_crossover=0.1,metric='accuracy',n_jobs=-1,adaptive_mutation: bool=False,max_n_population: int=10000,
              random_state=None,verbosity_lvl: int=1, approx_time_limit: int= 0,warm_start: bool=False,light=False):
            
    assert task in ['classification','regression'], 'task must either be classification or regression'

    self.task=task

    self.max_n_population=max_n_population

    self.approx_time_limit=None if approx_time_limit==0 else approx_time_limit

    if not self.approx_time_limit:

      assert n_generations, 'if no time limit, n_generations must be defined'
      
    assert init_p_crossover+init_p_mutation==1, 'rates must sum to 1'
    
    self.init_p_crossover=init_p_crossover

    self.init_p_mutation=init_p_mutation

    self.n_jobs=n_jobs

    self.adaptive_mutation=adaptive_mutation

    self.random_state=random_state

    self.verbosity_lvl=verbosity_lvl

    self.cv= ShuffleSplit(n_splits=1, random_state=self.random_state)

    self.init_n_population=init_n_population

    self.n_generations=n_generations

    self.fitted=False

    self.minimize=None

    self.configuration='TPOT light' if light else None

    self.warm_start=warm_start

    self.metric='accuracy' if self.task=='classification' else 'neg_mean_squared_error'

    self.selector=self.initialize_model()
    
  def initialize_model(self):

    if self.task=='classification':
      self.selector=TPOTClassifier(generations=1, population_size=self.init_n_population, verbosity=self.verbosity_lvl,n_jobs=-1, 
                                  random_state=self.random_state,cv=self.cv,scoring=self.metric,warm_start=self.warm_start,
                                mutation_rate=self.init_p_mutation,offspring_size=None,use_dask=True,config_dict=self.configuration)
     
    else:
      self.selector=TPOTRegressor(generations=1, population_size=self.init_n_population, verbosity=self.verbosity_lvl,config_dict=self.configuration,
                                random_state=self.random_state,cv=self.cv,scoring=self.metric,use_dask=True,n_jobs=-1,
                                warm_start=self.warm_start,mutation_rate=self.init_p_mutation,offspring_size=None)

    return self.selector
  
  def fit(self,x_train,y_train,x_test,y_test):

    x_train=check_array(x_train)
    y_train=check_array(y_train)
    x_test=check_array(x_test)
    y_test=check_array(y_test)
    
    self.fitted=False
      
    prev_score=0

    prev_pop=1

    if not self.approx_time_limit:

      n_gens=self.n_generations-1

    else:

      difference=0

      t1=datetime.datetime.now()

    self.n_population=self.init_n_population

    self.p_mutation=self.init_p_mutation

    if self.adaptive_mutation:

      self.stds=[]
    
    assert self.selector.warm_start==False

    self.selector.fit(x_train,y_train)
  
    condition=True

    while condition:

      print(self.n_population)
      
      self.selector.warm_start=True

      self.selector.offspring_size=prevFibonacci(self.n_population)

      self.selector.population_size=self.n_population

      self.selector.mutation_rate=self.p_mutation
      
      self.selector.crossover_rate=self.p_crossover=1-self.p_mutation

      self.selector.fit(x_train,y_train)

      score=self.selector.score(x_test,y_test)

      score=-score if self.minimize else score

      if self.approx_time_limit: 

        t2=datetime.datetime.now()

        difference=t2-t1
      
        condition=difference.seconds/60<=self.approx_time_limit

      else:

        n_gens-=1

        condition=n_gens>0

      if self.adaptive_mutation:
        
        stdev=np.std([self.selector.evaluated_individuals_[name]['internal_cv_score'] for name in self.selector.evaluated_individuals_])

        self.stds.append(stdev)
      
      if score>prev_score and self.n_population<prev_pop:

        prev_pop=self.n_population

        self.n_population=prevFibonacci(self.n_population)
      
      elif score>prev_score or self.n_population<prev_pop:
        prev_score=score
        
      else:

        prev_pop=self.n_population
        self.n_population=min(nextFibonacci(self.n_population),self.max_n_population)

        if self.adaptive_mutation:
          self.p_mutation=1-stdev/max(self.stds)
      
    self.fitted=True

    self.best_model=self.selector.fitted_pipeline_

    return self
  
  def custom_metric(self,func,minimize=False):

    assert 'y' in func.__code__.co_varnames and 'y_pred' in func.__code__.co_varnames and func.__code__.co_argcount==2,'must contain 2 arguments self,y,y_pred in order'

    try: 
      result=func(np.array([1,1]),np.array([2,2]))
    
    except Exception as e:
      print(e)
     
      raise Exception('fails to pass test inputs np.array([1,1]),np.array([2,2])')

    assert isinstance(result,numbers.Number), 'function must return a number'

    my_custom_scorer = make_scorer(func, greater_is_better=not self.minimize)

    self.metric=my_custom_scorer
    self.selector.scoring= self.metric
    
  def predict(self,X,y=None):

    X=check_array(X)

    return self.selector.predict(X)

  def fit_predict(self,x_train,y_train,x_test,y_test):

    return self.fit(x_train,y_train,x_test,y_test).predict(x_test)


class GeneticFeatureEngineer(TransformerMixin):
 
  def __init__(self,n_population: int=100,n_generations: int=30,n_participants: int=100,parsimony: float=0.001,
               metric='pearson',init_p_mutation: float=0.03,n_inductees: int=100, max_n_feats: int=30,superficiality: float=0.001,
               operations: Iterable=None, feat_names: Iterable=None,percentile: int=0,fitness_sharing: bool=False,
              adaptive: bool=False,n_eras: int=5,generational_improvement: bool=False,n_jobs: int=1,generosity: float=0.001):
    
    assert np.percentile(np.array([1,2]),percentile)

    self.fitness_sharing=fitness_sharing

    self.generosity=generosity

    self.superficiality=superficiality

    self.percentile=percentile
    
    self.parsimony=parsimony

    self.n_jobs=n_jobs

    self.adaptive=adaptive
    
    self.pickled=False
  
    self.metric=metric

    self.n_population=n_population

    self.n_inductees= n_inductees
    
    assert self.n_inductees<=self.n_population, 'HOF must be less than or equal to population'

    self.n_generations=n_generations

    self.n_participants=n_participants

    assert self.n_participants<=self.n_population, 'tournament size must be less than or equal to population'

    self.n_eras=n_eras

    self.max_n_feats=max_n_feats

    assert self.max_n_feats <= self.n_eras*self.n_generations*self.n_population, 'max # of features must be less than or equal to {calc}'.format(calc=self.n_eras*self.n_generations*self.n_population)

    self.init_p_mutations(init_p_mutation)
    
    assert (self.p_crossover+self.p_mutation)==1, 'rates must sum to 1'

    self.operations=operations if operations else ['add','sub','mul','sqrt','abs','neg','inv']
    
    self.feat_names=feat_names

    self.generational_improvement=generational_improvement

    self.fitted=False

    self.warm_start=False

    self.initialize_model()
  
    self.minimize=False

    self.codex=pd.DataFrame()

    self.mapper=None

    self.history=[]

  def initialize_model(self):
    
    self.engineer=SymbolicTransformer(population_size=self.n_population,hall_of_fame=self.n_inductees,generations=self.n_generations,parsimony_coefficient=self.parsimony,
                                    tournament_size=self.n_participants,const_range=None,function_set=self.operations,p_crossover=self.p_crossover,
                                    p_hoist_mutation=self.p_hoist,p_point_mutation=self.p_point,p_subtree_mutation=self.p_sub,n_jobs=self.n_jobs,n_components=self.n_inductees,
                                    feature_names=self.feat_names,metric=self.metric,warm_start=False)

    return self.engineer

  def calc_shared_fitness(self,fitness):

    if hasattr(self,'programs_lst') and self.programs_lst:

      fitnesses=np.array([gp.fitness_ for gp in self.programs_lst]).reshape(-1,1)  

      neigh=NearestNeighbors(n_neighbors=10,n_jobs=-1,metric=lambda x,y: scipy.spatial.distance.euclidean(x,y))

      neigh.fit(fitnesses)

      p_fitness=np.mean(neigh.kneighbors(np.array(fitness).reshape(-1,1))[0])      

      return self.generosity*p_fitness+fitness if not self.minimize else  -self.generosity*p_fitness+fitness
      
    else:
      return fitness
  
  def map_fitness_sharing(self,func):

    def new_func(y_true,y_pred,w):

      raw_fitness=func(self,y_true,y_pred)

      return self.calc_shared_fitness(raw_fitness)

    return new_func

  def fit_update(self,X,y,era=None):

   # era=None

    if not era:
      self.engineer.warm_start= False

    else: 
      self.engineer.warm_start=True
      
    self.engineer.fit(X,y)

    new=self.engineer.transform(X)

    column_names=[str(name) for name in self.engineer]

    new=pd.DataFrame(new,columns=column_names).T.drop_duplicates().T

    self.codex=pd.concat([self.codex,new],axis=1).T.drop_duplicates().T
    
    fitnesses=np.array([fx.fitness_ for fx in self.engineer]).reshape(-1,1)
      
    programs=np.array([gp for gp in self.engineer._best_programs])

    filtered_fitnesses=[x for x in fitnesses if x<=np.percentile(fitnesses,self.percentile)] if self.minimize else [x for x in fitnesses if x>=np.percentile(fitnesses,self.percentile)] 

    self.programs_lst+= list(programs[:len(filtered_fitnesses)])

    self.fitness_lst+=filtered_fitnesses

    assert len(self.programs_lst)==len(self.fitness_lst)
    
    return self.programs_lst,fitnesses
  
  def init_p_mutations(self,p_mutation: float=0.03):

    if p_mutation:
      assert p_mutation<=1
      self.p_mutation=p_mutation
  
    self.p_hoist=self.p_mutation/3

    self.p_sub=self.p_mutation/3

    self.p_point=self.p_mutation/3

    self.p_crossover=1-self.p_mutation
  
  def encode_genes(self,program,X):

    str_program=[x.name if not isinstance(x,int) else x for x in program]

    return np.array([self.mapper[decoding] for decoding in str_program])
  

  def calc_diversity(self,X):

    unpadded_genes=np.array([self.encode_genes(x.program,X) for x in self.programs_lst])

    max_length = max([len(row) for row in unpadded_genes])
    self.encoded_genes = np.array([np.pad(row, (0, max_length-len(row))) for row in unpadded_genes])
  
    self.g_neigh.fit(self.encoded_genes)

    g_diversity=np.mean(self.g_neigh.kneighbors(self.encoded_genes)[0])

    fitnesses=np.array(self.fitness_lst).reshape(-1,1)  

    self.p_neigh.fit(fitnesses)
   
    p_diversity=np.mean(self.p_neigh.kneighbors(fitnesses)[0])

    return self.superficiality*g_diversity+p_diversity

  def fitness_importances(self,program_lst):

    fitnesses=np.array([fx.fitness_ for fx in program_lst])

    indexer=np.argsort(fitnesses)

    top_sorted_programs=list(np.array(program_lst)[indexer])

    imps=pd.DataFrame()
    
    column_names=[str(name) for name in top_sorted_programs]

    imps.index=column_names

    imps['fitnesses']=[fx.fitness_ for fx in top_sorted_programs]
    
    imps['programs']=[fx for fx in top_sorted_programs]

    imps = imps[~imps.index.duplicated(keep='first')]

    imps=imps.sort_values('fitnesses',ascending=self.minimize)[:self.max_n_feats]

    self.leaderboard=imps.drop('programs',axis=1).copy()  

    self.engineer._best_programs=list(imps['programs'].values)
  
  def define_mapper(self,X):

    self.mapper={x:x for x in range(X.shape[-1])}

    operation_str=[x.name for x in self.operations if not isinstance(x,str)]+[x for x in self.operations if isinstance(x,str)]

    counter=0

    for operation in operation_str:

      self.mapper[operation]=counter+X.shape[-1]
      counter+=1

  def fit_create_function(program):

    def func(x1):

      if not len(x1.shape)==1:

        return program.execute(x1)

      else:

        return np.array(x1)

    return make_function(function=func,arity=1,name=str(program)[:20]+str(program)[-20:])
  
  def pickleable_version(self):

    assert self.fitted, 'cannot save unfitted model'

    to_pickle=copy.copy(self)

    to_pickle.engineer=copy.copy(self.engineer)
    
    engineerer=to_pickle.engineer

    engineerer._programs='replace_programs'
    
    engineerer.metric=self.metric_name

    engineerer._metric = self.metric_name

    for program in engineerer._best_programs:
    
      program.metric=self.metric_name
    
    to_pickle.metric=self.metric_name

    to_pickle.pickled=True

    return to_pickle

  def fit(self,X,y):

    assert not self.pickled, 'saved model only for transformation'

    X=check_array(X)
    y=check_array(y)

    self.fitted=False

    self.codex=pd.DataFrame(X.copy())
 
    self.define_mapper(X)
  
    self.programs_lst=[]

    self.fitness_lst=[]

    programs_lst,fitnesses=self.fit_update(X,y)

    if self.generational_improvement:
      new_ops=[self.fit_create_function(program) for program in self.programs_lst]
    
    if self.adaptive:

      self.g_neigh=NearestNeighbors(n_neighbors=10,n_jobs=-1,metric=lambda x,y: editdistance.eval(x,y))
  
      self.p_neigh=NearestNeighbors(n_neighbors=10,n_jobs=-1,metric=lambda x,y: scipy.spatial.distance.euclidean(x,y))

      max_participants=self.n_population

      self.diversity_lst=[]
      
      diversity=self.calc_diversity(X)

      self.diversity_lst.append(diversity)
    
    self.history.append(self.engineer.run_details_)

    for era in range(self.n_eras-1):

      self.era=era
  
      self.initialize_model()
      
      programs_lst,fitnesses=self.fit_update(X,y,era)

      if self.generational_improvement:
        new_ops+=[self.fit_create_function(program) for program in self.programs_lst]

      if self.adaptive:

        diversity=self.calc_diversity(X)

        self.diversity_lst.append(diversity)

        self.p_mutation=1-self.diversity_lst[-1]/np.max(self.diversity_lst)

        self.n_participants=round(self.diversity_lst[-1]/np.max(self.diversity_lst)*max_participants)
        
        self.init_p_mutations(None)
      
        self.history.append(self.engineer.run_details_)
      
    self.fitness_importances(np.array(self.programs_lst))

    self.fitted=True

    return self

  def custom_metric(self,func,name,minimize=False):

    self.minimize=minimize
          
    assert 'y' in func.__code__.co_varnames and 'y_pred' in func.__code__.co_varnames and func.__code__.co_varnames[0]=='self' and func.__code__.co_argcount==3,'must contain 3 arguments self,y,y_pred in order'

    try: 
      result=func(self,np.array([1,1]),np.array([2,2])) 
    
    except Exception as e:
      print(e)
      raise Exception('fails to pass test inputs np.array([1,1]),np.array([2,2])')

    assert isinstance(result,numbers.Number)

    def new_func(y_true,y_pred,w):

      return func(self,y_true,y_pred)

    if self.fitness_sharing:

      new_func=self.map_fitness_sharing(func)
    
    self.engineer.metric=make_fitness(function=new_func,greater_is_better=not self.minimize)

    self.metric=self.engineer.metric

    self.metric_name=name
    
  def custom_operation(self,func,func_name):
  
    custom_func=make_function(function=func,
                              name=func_name,
                              arity=func.__code__.co_argcount)

    self.engineer.function_set+=(custom_func,)

  def transform(self,X,y=None):

    assert self.fitted, 'engineer needs to be fitted'

    X=check_array(X)

    return self.engineer.transform(X)

  def fit_transform(self,X,y):

    return self.fit(X,y).transform(X)

  def feature_importances(self):

    assert self.fitted, 'engineer needs to be fitted'

    return self.leaderboard

class GeneticFeatureSelector(TransformerMixin):

  def __init__(self,n_population: int=100,n_generations: int=30,n_tournament: int=100,n_parents: int=None,mutation_type: str='random',
               metric=None,model=None,p_crossover: float=0.7,p_mutation: float=0.3,selection_type='rws',crossover_type: str='uniform',
               min_n_feats: int=2,minimize=False,adaptive=False):
  
    self.model=model

    if isinstance(self.model,Clusterer) or isinstance(self.model.__class__.__base__,Clusterer):

      assert min_n_feats>=2, 'clustering model requires >=2 minimal features'

    assert min_n_feats>0, 'cannot select 0 features'

    self.min_n_feats=min_n_feats

    self.minimize=minimize

    self.metric=metric if metric else mean_squared_error

    self.n_population=n_population if n_population else 100

    self.n_generations=n_generations if n_generations else 20

    self.n_tournament=n_tournament if n_tournament else 100

    self.n_parents=n_parents if n_parents else 100

    self.p_crossover=p_crossover 

    self.p_mutation=p_mutation

    assert (self.p_crossover+self.p_mutation)==1, 'rates must sum to 1'

    self.mutation_type= mutation_type if mutation_type else 'inversion'

    self.adaptive=adaptive 

    self.mutation_type='adaptive' if self.adaptive else self.mutation_type

    assert self.mutation_type in ['random','scramble','inversion','adaptive']

    if self.mutation_type=='adaptive':

       assert len(self.p_mutation)==2, 'adaptive mutation requires two rates'

       for rate in self.p_mutation:

         assert rate<1, 'p_mutation cannot be greater than 1'

       assert self.p_mutation[0]>self.p_mutation[-1],'1st rate must be greater than second'
    
    self.selection_type='tournament' if not selection_type else selection_type

    assert self.selection_type in ['sss','rws','sus','rank','random','tournament']

    self.crossover_type=crossover_type
    
    assert self.crossover_type in ['single_point','scattered','two_points','uniform']

    self.fitted=False

    def selection_fitness(solution,solution_idx):

      if not self.check_min_feats(solution):
        
        return -999
        
      self.model.fit(self.x_train[:,solution.astype(bool)],self.y_train)
  
      score=self.metric(self.y_test,self.model.predict(self.x_test[:,solution.astype(bool)])) 

      return score if not self.minimize else -score
    
    self.selection_fitness=selection_fitness
  
  def check_min_feats(self,solution):

    return np.sum(solution)>=self.min_n_feats
  
  def custom_metric(self,func):
    
    lst=[0,1]

    assert func.__code__.co_argcount==2, 'only takes 2 arguments'

    assert isinstance(func(lst,lst),float) or isinstance(func(lst,lst),int), 'requires iterables and/or numeric-based arguments'
    
    self.metric=func
  
  def fit(self,x_train,y_train,x_test,y_test):
      
    assert self.model,'model necessary for selection'

    x_train=check_array(x_train)
    Y_train=check_array(x_train)
    x_test=check_array(x_test)
    y_test=check_array(y_test)

    self.fitted=False

    self.x_train,self.y_train,self.x_test,self.y_test=x_train,y_train,x_test,y_test
      
    self.ga_instance = pygad.GA(num_generations=self.n_generations,
                       sol_per_pop=self.n_population,
                       num_genes=self.x_train.shape[-1],
                       gene_space=range(2),
                       num_parents_mating=self.n_parents,
                       fitness_func=self.selection_fitness,
                       gene_type=int,
                       crossover_type=self.crossover_type,
                       allow_duplicate_genes=True,
                       save_best_solutions=True,
                       crossover_probability=self.p_crossover,
                       parent_selection_type=self.selection_type,
                       K_tournament=self.n_tournament,
                       mutation_type=self.mutation_type,
                       mutation_probability=self.p_mutation)
    
    self.ga_instance.run()

    self.solution, solution_fitness, solution_idx = self.ga_instance.best_solution()

    if not self.check_min_feats(self.solution):
      print('Did not converge to >= {min_feats} minimal features,will use all features'.format(min_feats=self.min_n_feats))
    
      self.solution=np.full(shape=self.x_train.shape[-1],fill_value=1)
    
    self.fitted=True

    self.update_model()
          
    return self
  
  def update_model(self):

    assert self.fitted, 'nothing to update to'

    self.model.fit(self.transform(self.x_train),self.y_train)

    return self.model
  
  def transform(self,X,y=None):
    
    X=check_array(X).copy()

    return X[:,self.solution.astype(bool)]
  
  def fit_transform(self,X,y,x_test=None,y_test=None):

    return self.fit(X,y,x_test,y_test).transform(X)

