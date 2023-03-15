import numpy as np
from collections.abc import Iterable
import pandas as pd
import copy
import numbers
import datetime 
import scipy
import functools

from functools import wraps

from tpot import TPOTClassifier,TPOTRegressor

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.base import RegressorMixin,ClassifierMixin,TransformerMixin,check_array
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

import pygad

from sklego.base import Clusterer

from .utils import inject_repr,prevFibonacci,nextFibonacci,JWdistance,alt_cdf

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

from sklearn.cluster import KMeans

from sklearn.cluster import KMeans

class GeneticFeatureEngineer(TransformerMixin):
     
  def __init__(self,n_population: int=100,n_generations: int=30,n_participants: int=100,parsimony: float=0.001,random_state= None,n_seasons: int=0,impostor_gene: bool=False,
               metric='pearson',init_p_mutation: float=0.03,n_inductees: int=100, max_n_feats: int=30,superficiality: float=0.001,phenology: float=0.001,impostor_penalty:float=0.001,
               operations: Iterable=None, feat_names: Iterable=None,percentile: int=0,fitness_sharing: bool=False,mortality:float=0.001,batch:bool=False,
              adaptive: bool=False,n_eras: int=5,n_jobs: int=1,generosity: float=0.001,aging: bool=False,extinction: bool=False,n_stagnation: int=1):
    
    assert np.percentile(np.array([1,2]),percentile)

    self.fitness_sharing=fitness_sharing

    self.impostor_gene=impostor_gene

    self.extinction=extinction

    self.extinction_counter=0 

    self.n_stagnation=n_stagnation
    
    self.aging=aging

    self.mortality=mortality

    self.generosity=generosity

    self.impostor_penalty=impostor_penalty

    self.superficiality=superficiality

    self.percentile=percentile
    
    self.parsimony=parsimony

    self.phenology=phenology

    self.batch= batch

    self.n_jobs=n_jobs
    
    self.random_state= np.random.randint(10000) if not random_state else random_state

    self.adaptive=adaptive
    
    self.pickled=False

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

    self.operations_copy=self.operations.copy()
    
    self.feat_names=feat_names

    self.fitted=False

    self.n_seasons=n_seasons

    assert self.n_seasons<=self.n_population,'seasons cannot exceed population size'

    self.metric= self.custom_metric(metric,metric.__code__.co_name) if callable(metric) else metric

    self.random_counter=0

    self.init_model()
  
    self.minimize=False

    self.sign=1 if not self.minimize else -1

    self.codex=pd.DataFrame()

    self.mapper=None

    self.reverse_mapper=None

    self.leaderboard=None

    self.history=[]

    self.pareto_metrics=[]
  
  def init_model(self):

    self.engineer=SymbolicTransformer(population_size=self.n_population,hall_of_fame=self.n_inductees,generations=self.n_generations,parsimony_coefficient=self.parsimony,
                                    tournament_size=self.n_participants,const_range=None,function_set=self.operations,p_crossover=self.p_crossover,
                                    p_hoist_mutation=self.p_hoist,p_point_mutation=self.p_point,p_subtree_mutation=self.p_sub,n_jobs=self.n_jobs,
                                    feature_names=self.feat_names,metric=self.metric,warm_start=False,random_state=self.random_state+self.random_counter,n_components=self.n_inductees)

    self.random_counter+=1
    return self.engineer

  def calc_shared_fitness(self, fitness):
    if not hasattr(self, 'codex_programs'):
        return fitness
    fitnesses = np.array([gp.fitness_ for gp in self.codex_programs])
    if not fitnesses.size:
        return fitness
    distances = np.sqrt(np.sum((fitnesses.reshape(-1, 1) - fitness.reshape(1, -1)) ** 2, axis=0))
    neighbors = distances.argsort()[:min(10, len(fitnesses))]
    p_fitness = np.mean(fitnesses[neighbors])
    return self.sign * (self.generosity * p_fitness) + fitness
  
  def map_fitness_sharing(self,func):
    @wraps(func)
    def new_func(y_true,y_pred,w):

      raw_fitness=func(self,y_true,y_pred)

      return self.calc_shared_fitness(raw_fitness)

    return new_func
  
  def map_birthdates(self,programs):
    for program in programs:
      program.birthdate=self.era*self.engineer.generations
  
  def age_penalty(self,age):
    if age==0:
      return 0
  
    birthdates=[program.birthdate for program in self.codex_programs]
    range_=np.max(birthdates)-np.min(birthdates)
    st_dev=np.std(birthdates)
    b=range_
    a=2*st_dev**2
    return a*2.714**-(((age-b)**2)/(a))
  
  def fit_update(self, X, y, era=None, verbose=True):
    if verbose:
        t1 = datetime.datetime.now()
    self.engineer.fit(X, y)
    if verbose:
        t2 = datetime.datetime.now()
    new = self.engineer.transform(X)
    dup_idxs = pd.DataFrame(new).T.duplicated()
    programs = np.array(self.engineer._best_programs)
    programs = programs[~dup_idxs]
    column_names = [str(name) for name in programs]
    new = pd.DataFrame(new[:, ~dup_idxs], columns=column_names)
    self.codex = pd.concat([self.codex, new], axis=1).T.drop_duplicates().T if not self.batch else new.copy()
    
    if self.n_seasons > 1:
      season_sign = 1 if (self.era % 2 == 0) else -1

      scaler=StandardScaler()
      clusterer = KMeans(n_clusters=self.n_seasons)
      
      scaled_features =np.apply_along_axis(alt_cdf,0,np.array([program.execute(X) for program in programs]).T)
      scaled_features = scaler.fit_transform(scaled_features)

      nans_idx = np.isnan(scaled_features).any(axis=0)
      infs_idx = np.isinf(scaled_features).any(axis=0)

      programs = programs[~nans_idx & ~infs_idx]
      scaled_features = scaled_features[:, ~nans_idx & ~infs_idx]

      cluster_idxs = clusterer.fit_predict(scaled_features.T)

      for idx, program in enumerate(programs):
          program.fitness_ = program.fitness() - self.sign * self.phenology * cluster_idxs[idx] * season_sign

    if self.aging:
        self.map_birthdates(programs)

        for program in programs:
            program.fitness_ = program.fitness() - self.sign * self.mortality * self.age_penalty(self.era - program.birthdate)

        for program in self.codex_programs:
            program.fitness_ = program.fitness() - self.sign * self.mortality * self.age_penalty(self.era - program.birthdate)
    
    programs=list(programs)
  
    if self.n_seasons > 1 or self.aging:
        for program in programs + self.codex_programs:
            program.fitness_ = program.fitness()

    if len(self.pareto_metrics)>0:
      sign=-1 if self.minimize else 1
     
      programs_array=np.array([program.execute(X) for program in programs]).T

      for idx,program in enumerate(programs):
        penalty = self.parsimony * program.length_ *sign
        pareto_tuple=(program.raw_fitness_,)
        for pareto_metric in self.pareto_metrics:
          pareto_tuple+=tuple([pareto_metric(self,y,programs_array[:,idx])-penalty])
        program.pareto_fitnesses=pareto_tuple
      
      programs=self.pareto_front_fitness(programs)

      for program in programs:
        penalty = self.parsimony * program.length_ *sign
        program.fitness_=program.fitness_-penalty
      
    programs=self.gene_assignments(programs)
    
    if self.extinction and self.era!=0:
      stagnation_condition=self.engineer.run_details_['gene_scores'].max()<self.history[-1]['gene_scores'].max()
    
      self.extinction_counter+= -1 if not stagnation_condition else int(stagnation_condition)
      self.gene_extinction()

    # Sort the programs list by fitness
    programs.sort(key=lambda x: x.fitness_, reverse=self.minimize)
    filtered_fitnesses = [program.fitness_ for program in programs if
                          program.fitness_ <= np.percentile([p.fitness_ for p in programs], self.percentile)] if self.minimize else [
        program.fitness_ for program in programs if
        program.fitness_ >= np.percentile([p.fitness_ for p in programs], self.percentile)]

    # Use numpy slicing to get the filtered programs
    self.codex_programs += programs[:len(filtered_fitnesses)]

    codex_strs = [str(program) for program in self.codex_programs]
    program_strs = [str(program) for program in programs]
    
    
    return self.codex_programs
  
  def init_p_mutations(self,p_mutation: float=0.03):

    if p_mutation:
      assert p_mutation<=1
      self.p_mutation=p_mutation
  
    self.p_hoist=self.p_mutation/3
    self.p_sub=self.p_mutation/3
    self.p_point=self.p_mutation/3
    self.p_crossover=1-self.p_mutation
  
  def encode_gene(self,program):

    str_program=[x.name if not isinstance(x,int) else x for x in program]

    return np.array([self.mapper[decoding] for decoding in str_program])
  
  import numpy as np

  def pareto_front_fitness(self,objs):
    n_objectives = len(objs[0].pareto_fitnesses)
    n_individuals = len(objs)

    scores = np.zeros((n_objectives, n_individuals))
    for i in range(n_individuals):
        for j in range(n_objectives):
            scores[j][i] = getattr(objs[i], "pareto_fitnesses")[j]

    crowding_distances = np.zeros(n_individuals)

    for objective in range(n_objectives):
        sorted_indices = np.argsort(scores[objective])
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf

        for i in range(1, n_individuals - 1):
            fitness_range = scores[objective, sorted_indices[i+1]] - scores[objective, sorted_indices[i-1]]
            crowding_distances[sorted_indices[i]] += fitness_range
    
    new_scores = 1 / (crowding_distances + 1)

    for i in range(n_individuals):
        setattr(objs[i], "fitness_", new_scores[i])

    return objs

  def encode_genes(self,programs):
    
    unpadded_genes=np.array(list({self.encode_gene(x.program).tostring(): self.encode_gene(x.program) for x in programs}.values()))
    max_length = max([len(row) for row in unpadded_genes])
    encoded_genes = np.array([np.pad(row, (0, max_length-len(row)),constant_values=-999) for row in unpadded_genes])
      
    return encoded_genes
  
  def gene_assignments(self, programs):
    
    encoded_genes = self.encode_genes(programs)
    operation_encodings = [self.mapper[x] for x in self.mapper if isinstance(x, str)]
    encoded_genes[~np.isin(encoded_genes, operation_encodings)] = -999
    fitnesses=np.array([gx.fitness_ for gx in programs])

    if self.impostor_gene:

        fitnesses -= self.impostor_penalty

        impostor_idx = (encoded_genes ==-1).any(axis=1)
        ref_fitness = fitnesses[impostor_idx].mean()
        mask = (fitnesses > ref_fitness) & (~impostor_idx)

        programs = np.array(programs)[mask]
        fitnesses = fitnesses[mask]
        encoded_genes = encoded_genes[mask]

        assert len(programs) != 0, 'Average impostor gene fitness too high'
          
    repeated = np.repeat(fitnesses, repeats=encoded_genes.shape[-1], axis=0).reshape(encoded_genes.shape)

    gene_scores = pd.DataFrame({'fitness': repeated.ravel(), 'gene': encoded_genes.ravel()}).groupby('gene').mean()['fitness']
    gene_counts = pd.Series(encoded_genes.reshape(-1)).value_counts()

    gene_scores.index = [self.reverse_mapper[x] for x in gene_scores.index]
    gene_counts.index = [self.reverse_mapper[x] for x in gene_counts.index]
    
    self.engineer.run_details_['gene_scores'] = gene_scores
    self.engineer.run_details_['gene_counts'] = gene_counts

    return list(programs)
  
  def gene_extinction(self):
    purge_lst=[]
    if self.extinction_counter == self.n_stagnation:

        gene_counts = self.engineer.run_details_['gene_counts']
        purge_idx = gene_counts.drop(-999).argmax()

        if self.str_operations[purge_idx] != 'impostor_operation':
            if purge_idx < len(self.operations):
                del self.operations[purge_idx]
                print(f'purging {self.str_operations[purge_idx]}')
                purge_lst.append(self.str_operations[purge_idx])
                
            else:
                # index is out of bounds, do nothing
                pass

        self.extinction_counter = 0
    self.engineer.run_details_['purged_genes']=purge_lst

  def calc_diversity(self,X):
    self.encoded_genes = self.encode_genes(self.codex_programs)
    self.g_neigh=NearestNeighbors(n_neighbors=min(5,len(self.encoded_genes)),n_jobs=-1,metric=lambda x,y: JWdistance(x,y))
    self.p_neigh=NearestNeighbors(n_neighbors=min(5,len(self.encoded_genes)),n_jobs=-1,metric=lambda x,y: scipy.spatial.distance.euclidean(x,y))
    self.g_neigh.fit(self.encoded_genes)
    g_diversity=np.mean(self.g_neigh.kneighbors(self.encoded_genes)[0])
    fitnesses=np.array([gp.fitness_ for gp in self.codex_programs]).reshape(-1,1)  
    self.p_neigh.fit(fitnesses)
    p_diversity=np.mean(self.p_neigh.kneighbors(fitnesses)[0])
    self.engineer.run_details_['phenotypic diversity']=p_diversity
    self.engineer.run_details_['genetic diversity']=g_diversity
    return self.superficiality*g_diversity+p_diversity

  def fitness_importances(self,program_lst):

    fitnesses = np.array([fx.fitness_ for fx in program_lst])
    indexer = np.argsort(fitnesses)
    imps = pd.DataFrame({'fitnesses': fitnesses[indexer]}, index=[str(program_lst[i]) for i in indexer])
    imps['programs'] = [program_lst[i] for i in indexer]
    imps = imps.drop_duplicates(keep='last')
    imps = imps.sort_values('fitnesses', ascending=self.minimize)[:self.max_n_feats]
    self.engineer._best_programs = imps['programs'].tolist()
    self.leaderboard = imps[['fitnesses']].copy()
    self.leaderboard.index = [str(program) for program in self.engineer._best_programs]
    self.operations = self.operations_copy if self.extinction else self.operations
  
  def define_mapper(self, X):
    n_features = X.shape[-1]
    self.mapper = {x: x for x in range(n_features)}
    self.mapper[-999]=-999
    self.str_operations = [x if isinstance(x,str) else x.name for x in self.operations]
    for i, operation in enumerate(self.str_operations, start=n_features):
        self.mapper[operation] = i
    self.reverse_mapper = {y: x for x, y in self.mapper.items()}
    self.reverse_mapper[-999]=-999

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
  
  def archive(self):

      self.engineer.run_details_['n_participants']=self.n_participants
      self.engineer.run_details_['p_mutation']=self.p_mutation
      self.engineer.run_details_['era']=self.era
      self.history.append(self.engineer.run_details_)

  def fit(self, X, y):
    assert not self.pickled, 'saved model only for transformation'
    X = check_array(X)
    y = check_array(y)
    self.era = 0
    self.fitted = False
    self.codex = pd.DataFrame(X.copy())
    self.define_mapper(X)
    self.codex_programs = []
    if self.impostor_gene:
        self.custom_operation(lambda x1,x2: np.random.rand(*x1.shape), 'impostor_operation')
        self.mapper['impostor_operation'] = -1
        self.reverse_mapper[-1] = 'impostor_operation'
        self.str_operations.append('impostor_operation')
        
    codex_programs = self.fit_update(X, y)
    if self.adaptive:
        max_participants = self.n_population
        self.diversity_lst = [self.calc_diversity(X)]
    self.archive()
    for era in range(self.n_eras - 1):
        self.init_model()
        self.era += 1
        programs_lst = self.fit_update(X, y, era)
        if self.adaptive:
            self.diversity_lst.append(self.calc_diversity(X))
            self.p_mutation = 1 - self.diversity_lst[-1] / np.max(self.diversity_lst)
            self.n_participants = round(self.diversity_lst[-1] / np.max(self.diversity_lst) * max_participants)
            self.init_p_mutations(None)
        self.archive()
    self.fitness_importances(np.array(self.codex_programs))
    self.fitted = True
    return self

  def custom_metric(self, func, name, minimize=False):
    self.minimize = minimize
    assert 'y' in func.__code__.co_varnames and 'y_pred' in func.__code__.co_varnames and func.__code__.co_varnames[0] == 'self' and func.__code__.co_argcount == 3, 'must contain 3 arguments self,y,y_pred in order'
    try: 
        result = func(self, np.array([1, 1]), np.array([2, 2]))
    except Exception as e:
        print(e)
        raise Exception('fails to pass test inputs np.array([1,1]),np.array([2,2])')
    assert isinstance(result, numbers.Number)

    @wraps(func)
    def new_func(y_true, y_pred, w):
        return func(self, y_true, y_pred)

    if self.fitness_sharing:
        new_func = self.map_fitness_sharing(func)

    self.metric = make_fitness(function=new_func, greater_is_better=not self.minimize)
    self.metric_name = name
    direction = 'minimize' if minimize else 'maximize'
    print('Custom metric {name} to {direction} has been equipped'.format(name=name, direction=direction))

    if hasattr(self, 'engineer'):
        self.engineer.metric = self.metric
    else:
        return self.metric
  
  def add_pareto_metric(self, func, name):
    assert 'y' in func.__code__.co_varnames and 'y_pred' in func.__code__.co_varnames and func.__code__.co_varnames[0] == 'self' and func.__code__.co_argcount == 3, 'must contain 3 arguments self,y,y_pred in order'
    try: 
        result = func(self, np.array([1, 1]), np.array([2, 2]))
    except Exception as e:
        print(e)
        raise Exception('fails to pass test inputs np.array([1,1]),np.array([2,2])')
    assert isinstance(result, numbers.Number)

    direction = 'minimize' if self.minimize else 'maximize'
    print('Metric {name} to {direction} has been equipped as pareto metric {length}'.format(name=name, direction=direction,length=len(self.pareto_metrics)+1))

    self.pareto_metrics.append(func)

  def custom_operation(self,func,func_name):
    custom_func=make_function(function=func,
                              name=func_name,
                              arity=func.__code__.co_argcount)

    self.operations_copy+=(custom_func,)
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

