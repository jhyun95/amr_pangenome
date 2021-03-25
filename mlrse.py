#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:41:138 2020

@author: jhyun95

"""

import numpy as np
import numbers, itertools
from joblib import Parallel, delayed, effective_n_jobs

import sklearn.ensemble
from sklearn.ensemble.base import _partition_estimators
from sklearn.ensemble.bagging import _generate_bagging_indices
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import has_fit_parameter

MAX_INT = np.iinfo(np.int32).max

def _careful_parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, total_n_estimators, verbose):
    """
    Modified from sklearn.ensemble._parallel_build_estimators()
    
    Private function used to build a batch of estimators within a job.
    """
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_, "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators)))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)
        
        ''' UPDATED SAMPLING SECTION '''
        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state, bootstrap_features, bootstrap, n_features,
            n_samples, max_features, max_samples)
        
        while len(np.unique(y[indices])) < 2:
            # Resample until training set is not single-class
            features, indices = _generate_bagging_indices(
                random_state, bootstrap_features, bootstrap, n_features,
                n_samples, max_features, max_samples)
            
        # Don't use sample weights, to be compatible with LinearSVC
        estimator.fit((X[indices])[:, features], y[indices])

        ''' END OF MODIFIED SECTION '''
        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


class CarefulBaggingClassifier(sklearn.ensemble.BaggingClassifier):
    '''
    Modified BaggingClassifier to address two issues with bootstrapping.
    
    1) When bootstrapping, BaggingClassifier will generate sample_weights 
    rather than create slices of the input matrix. However, LinearSVC
    is unable to use sample_weights and will forgo bootstrapping entirely.
    (see https://github.com/scikit-learn/scikit-learn/issues/10873)
    CarefulBaggingClassifier creates slices of the input matrix for
    bootstrapping rather than generating sample_weights.
    
    2) Bootstrapping with small imbalanced datasets may generate cases 
    that are entirely one class. CarefulBaggingClassifier resamples
    if this is the case.
    '''
    
    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """
        Adapted from BaggingClassifier.
        
        Build a Bagging ensemble of estimators from the training
        set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.
        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
        """
        random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(X, y, ['csr', 'csc'], dtype=None, 
                         force_all_finite=False, multi_output=True)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        ''' Replaced _parallel_build_estimators() with 
            _careful_parallel_build_estimators(), '''
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_careful_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self
    

class SafeLinearSVC(sklearn.svm.LinearSVC):
    ''' 
    LinearSVC modified to bootstrap correctly with original BaggingClassifier 
    (and by extension, BalancedBaggingClassifier).
    
    When bootstrapping, BaggingClassifier will generate sample_weights 
    rather than create slices of the input matrix. However, LinearSVC
    is unable to use sample_weights and will forgo bootstrapping entirely.
    (see https://github.com/scikit-learn/scikit-learn/issues/10873)
    
    If the base estimator's fit() function does not support sample_weights, 
    it will fall back on generating slices which will work correctly 
    with LinearSVC. (see sklearn.ensemble._parallel_build_estimators).
    '''
    
    def fit(self, X, y):
        print(X.shape, y.shape, y.sum(), y.shape[0] - y.sum(), y.tolist())
        return super(SafeLinearSVC, self).fit(X, y)