from sklego.base import Clusterer

from sklearn.base import check_array,BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import scipy

import numpy as np
import copy

from pipeline import GeneticFeatureEngineer,GeneticFeatureSelector,GeneticModelSelector

class KlusterFoldValidation:
    """
    KlusterFold cross validator
    - Create folds based on provided cluster method
    :param cluster_method: Clustering method with fit_predict attribute
    """

    def __init__(self, cluster_method=None,n_splits: int=5):
        if not isinstance(cluster_method, Clusterer):
            raise ValueError(
                "The KlusterFoldValidation only works on cluster methods with .fit_predict."
            )
        assert 'n_clusters' in cluster_method.__dict__, 'requires n_clusters as attribute'
        
        self.cluster_method = cluster_method
        self.n_splits = n_splits
        self.cluster_method.n_clusters= self.n_splits
        
        assert self.cluster_method.n_clusters==self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generator to iterate over the indices
        :param X: Array to split on
        :param y: Always ignored, exists for compatibility
        :param groups: Always ignored, exists for compatibility
        """

        X = check_array(X)

        self.cluster_method.fit(X)
        clusters = self.cluster_method.predict(X)

        self.n_splits = len(np.unique(clusters))

        if self.n_splits < 2:
            raise ValueError(
                f"Clustering method resulted in {self.n_splits} cluster, too few for fold validation"
            )

        for label in np.unique(clusters):
            yield (np.where(clusters != label)[0], np.where(clusters == label)[0])

class AgglomSub(AgglomerativeClustering):

  def __init__(self,n_clusters=5, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
    super().__init__(self)

    self.n_clusters=n_clusters

  def predict(self,X,y=None):

    return self.fit_predict(X)
  
class SingleLabelPredictor(BaseEstimator):

  def __init__(self,label,X):
    super().__init__(self)
    
    self.label=label

  def fit(self,X=None,y=None):
    return self
  
  def transform(self,X=None,y=None):

    return np.full((X.shape[0],1),self.label)
  
  def predict(self,X=None,y=None):

    return np.full((X.shape[0],1),self.label)
  
  def fit_transform(self,X=None,y=None):
    
    return np.full((X.shape[0],1),self.label)
  
  def fit_predict(self,X=None,y=None):
    return np.full((X.shape[0],1),self.label)

class GeneticAutoML:
    
  def __init__(self,feat_engineer: GeneticFeatureEngineer,feat_selector: GeneticFeatureSelector,model_selector:GeneticModelSelector,
                metric=None,n_outer_folds: int=5,task: str='classification',minimize:bool=False,scaler=StandardScaler,
               ensemble_type: str='stacking',fold_clusterer=None):
    
    self.func_dispatch={'classificationstacking': self.stack,
    'classificationvoting': self.class_vote,
    'regressionstacking': self.stack,
    'regressionvoting': self.reg_vote,
    'classificationvoting_proba': self.vote_proba,
    'classificationstacking_proba': self.stack_proba}

    assert task in ['classification','regression'], 'task must either be classification or regression'

    assert ensemble_type in ['stacking','voting'],'not a valid ensemble type'

    assert n_outer_folds>1, 'external validation requires >1 outer_folds'

    self.minimize=minimize

    self.scaler=scaler
    
    self.n_outer_folds=n_outer_folds
    
    fold_clusterer= AgglomSub() if not fold_clusterer else fold_clusterer

    self.outer_cv=KlusterFoldValidation(fold_clusterer,n_splits=self.n_outer_folds)

    self.feat_engineer=feat_engineer
    
    self.feat_selector=feat_selector

    self.model_selector=model_selector
    
    self.task=task

    self.ensemble_type=ensemble_type

    self.metric=metric if metric else self.feat_selector.metric

    self.ensemble=None

  def stack(self,meta_X):

    return self.meta_model.predict(meta_X)
 
  def reg_vote(self,meta_X):
    
    return np.mean(meta_X,axis=1)
  
  def class_vote(self,meta_X):

    return scipy.stats.mode(meta_X)[0].reshape(-1,1)

  def stack_proba(self,meta_X):
    
    return self.meta_model.predict_proba(meta_X)
  
  def vote_proba(self,meta_X):

    return np.sum(np.array([pipe.predict_proba(meta_X) for pipe in self.best_individual_models].T),axis=1)

  def custom_metric(self,func):
    
    lst=[0,1]

    assert func.__code__.co_argcount==2, 'only takes 2 arguments'

    assert isinstance(func(lst,lst),float) or isinstance(func(lst,lst),int), 'requires iterables and/or numeric-based arguments'
    
    self.scorer=func
    
  def get_ensemble(self,x_train,y_train,x_test,y_test,ensemble_type=None):
  
    if ensemble_type:
      assert ensemble_type in ['stacking','voting'],'not a valid ensemble type'
      self.ensemble_type=ensemble_type

    if self.ensemble_type=='stacking':

      self.model_selector.warm_start=False

      meta_x_train=pd.concat([pd.DataFrame(pipe.predict(x_train)) for pipe in self.best_individual_models],axis=1)

      meta_x_test=pd.concat([pd.DataFrame(pipe.predict(x_test)) for pipe in self.best_individual_models],axis=1) 

      self.model_selector.initialize_model()

      self.model_selector.selector.warm_start=False   
      
      self.model_selector.fit(meta_x_train,y_train,meta_x_test,y_test)

      self.meta_model=self.model_selector.best_model
  
  def genetic_pipeline(self,x_train,y_train,x_test,y_test,cluster_error=False):    

      x_tran_train=self.feat_engineer.fit_transform(x_train,y_train)

      x_tran_test=self.feat_engineer.transform(x_test)

      self.model_selector.initialize_model()

      self.model_selector.selector.warm_start=False
      
      self.model_selector.fit(x_tran_train,y_train,x_tran_test,y_test)

      self.feat_selector.model=self.model_selector.best_model

      self.feat_selector.fit(x_tran_train,y_train,x_tran_test,y_test)

      engineer_copy=self.feat_engineer.pickleable_version()
      
      self.model_selector.best_model.fit(self.feat_selector.transform(x_tran_train),y_train)

      selector_copy=copy.deepcopy(self.feat_selector)

      pipeline=Pipeline([('engineer',engineer_copy.engineer),('selector',selector_copy),('model',selector_copy.model)])

      if not cluster_error:

        score=self.metric(y_test,pipeline.predict(x_test))

        score=-score if self.minimize else score

        return pipeline,score
      
      else:

        return pipeline
        
  def fit_pipeline(self,x_train,y_train,x_test,y_test):
    
    if len(np.unique(y_train))==len(np.unique(y_test))==1:

      pipeline=SingleLabelPredictor(np.unique(y_train)[0])

      score=np.nan
    
    elif len(np.unique(y_train))!=len(np.unique(y_test)):

      x_train,y_train,x_test,y_test=x_test,y_test,x_train,y_train

      pipeline=self.genetic_pipeline(x_train,y_train,x_test,y_test,cluster_error=True) 

      score=np.nan

    else:
        pipeline,score=self.genetic_pipeline(x_train,y_train,x_test,y_test)

    return pipeline,score
  
  def fit(self,x_train,y_train,x_test,y_test):

    x_train=check_array(x_train)
    y_train=check_array(y_train)
    x_test=check_array(x_test)
    y_test=check_array(y_test)

    x_train=x_train.copy() 

    self.outer_folds=list(self.outer_cv.split(self.scaler.fit_transform(x_train)))
    
    self.clipboard={}

    self.clipboard['outer_folds']=self.outer_folds

    outer_counter=0

    outer_container=[]

    for train_idx,test_idx in self.outer_folds:

      x_train_outer=x_train[train_idx]
      y_train_outer=y_train[train_idx]

      x_test_outer=x_train[test_idx]
      y_test_outer=y_train[test_idx]
  
      pipeline,score=self.fit_pipeline(x_train_outer,y_train_outer,x_test_outer,y_test_outer)

      self.clipboard[outer_counter]=(pipeline,score)
            
      outer_container.append(pipeline)
      
      outer_counter+=1

    self.best_individual_models=outer_container.copy()

    self.get_ensemble(x_train,y_train,x_test,y_test)

    return self 

  def predict(self,X,y=None):

    meta_X=np.array([pipe.predict(X) for pipe in self.best_individual_models]).T

    return self.func_dispatch[self.task+self.ensemble_type](meta_X)

  def predict_proba(self,X,y=None):
    
    assert self.task=='classification','predict_proba not implemented for non-classifier'

    meta_X=np.array([pipe.predict(X) for pipe in self.best_individual_models]).T

    return self.func_dispatch[self.task+self.ensemble_type+'_proba'](meta_X)
  
  def transform(self,X,y=None):

    return np.array([pipe.transform(X) for pipe in self.best_transformers]).T    
  
  def calc_imps(self,model,factor=1): #self

    if 'feature_importances_' in dir(model):

      return model.feature_importances_ *factor

    elif 'coef_' in dir(model):
      
      return model.coef_ * factor 
      
    else:

      return 'No feature importance implemented'

  def feature_importances(self):

    self.feat_imps={}

    factors=self.calc_imps(self.meta_model[-1]) if self.meta_model else np.full((len(self.best_individual_models),),1)
    
    for n,model in enumerate([pipe['model'] for pipe in self.best_individual_models]):
      
      while '__len__' in dir(model):

          model=model[-1]
          
      self.feat_imps['fold'+str(n)]=self.calc_imps(model,factors[n])

    return self.feat_imps