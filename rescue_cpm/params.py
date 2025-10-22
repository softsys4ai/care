#!/usr/bin/env python3

from __future__ import annotations

from typing import List

from pydantic import BaseModel, field_validator, model_validator
import numpy as np
import pandas as pd

class CausalPerformanceModelParams(BaseModel):
    data: pd.DataFrame
    parameters: List[str]
    metrics: List[str]
    objectives: List[str]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('data')
    def check_data_is_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise TypeError('The "data" field must be a pandas DataFrame.')
        return v

    @model_validator(mode='before')
    def check_all_fields(cls, values):
        parameters = values.get('parameters')
        metrics = values.get('metrics')
        objectives = values.get('objectives')

        for name, value in [('parameters', parameters), ('metrics', metrics), ('objectives', objectives)]:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise TypeError(f'The "{name}" field must be a list of strings. Hint: Ensure you are passing a list where each element is a string.')

        return values

class ConditionalMutualInformationParams(BaseModel):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def check_all_fields(cls, values):
        for name, value in values.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f'The "{name}" field must be a numpy ndarray. Hint: Ensure you are passing a numpy array.')
        return values
    

class CalculateResidualsParams(BaseModel):
    data: pd.DataFrame
    target: str

    class Config:
        arbitrary_types_allowed = True

    @field_validator('data')
    def check_data_is_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise TypeError('The "data" field must be a pandas DataFrame. Hint: Ensure you are passing a DataFrame object from pandas.')
        return v

    @field_validator('target')
    def check_target_is_string(cls, v):
        if not isinstance(v, str):
            raise TypeError('The "target" field must be a string. Hint: Ensure you are passing a string value.')
        return v

class EntropyParams(BaseModel):
    z: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @field_validator('z')
    def check_is_ndarray(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError('The "z" field must be a numpy ndarray. Hint: Ensure you are passing a numpy array.')
        return v

class CalculateJointDistributionParams(BaseModel):
    data: pd.DataFrame
    x: str
    y: str
    bins: int = 10

    class Config:
        arbitrary_types_allowed = True

    @field_validator('data')
    def check_data_is_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise TypeError('The "data" field must be a pandas DataFrame. Hint: Ensure you are passing a DataFrame object from pandas.')
        return v

    @field_validator('x', 'y')
    def check_is_string(cls, v, field):
        if not isinstance(v, str):
            raise TypeError(f'The "{field.name}" field must be a string. Hint: Ensure you are passing a string value.')
        return v

    @field_validator('bins')
    def check_is_int(cls, v):
        if not isinstance(v, int):
            raise TypeError('The "bins" field must be an integer. Hint: Ensure you are passing an integer value.')
        return v

class LatentSearchParams(BaseModel):
    x_support: range
    y_support: range
    z_support: range
    p_xy: np.ndarray
    q_z_given_xy_init: np.ndarray
    beta: float
    num_iterations: int

    class Config:
        arbitrary_types_allowed = True

    @field_validator('x_support', 'y_support', 'z_support')
    def check_is_range(cls, v, field):
        if not isinstance(v, range):
            raise TypeError(f'The "{field.name}" field must be a range. Hint: Ensure you are passing a range object.')
        return v

    @field_validator('p_xy', 'q_z_given_xy_init')
    def check_is_ndarray(cls, v, field):
        if not isinstance(v, np.ndarray):
            raise TypeError(f'The "{field.name}" field must be a numpy ndarray. Hint: Ensure you are passing a numpy array.')
        return v

    @field_validator('beta')
    def check_beta(cls, v):
        if not isinstance(v, float) or v < 0:
            raise TypeError('The "beta" field must be a float greater than 0. Hint: Ensure you are passing a float value >= 0.')
        return v

    @field_validator('num_iterations')
    def check_is_int(cls, v):
        if not isinstance(v, int):
            raise TypeError('The "num_iterations" field must be an integer. Hint: Ensure you are passing an integer value.')
        if v <= 0:
            raise ValueError('The "LS_iter" field must be an integer greater than or equal to 0. Hint: Ensure you are passing an integer value greater than or equal to 0.')
        return v

class IdentifyPartiallyDirectedEdgesParams(BaseModel):
    edges: List[str]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('edges')
    def check_is_list_of_strings(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise TypeError('The "edges" field must be a list of strings. Hint: Ensure you are passing a list where each element is a string.')
        return v

    @field_validator('edges')
    def check_valid_edge_patterns(cls, v):
        valid_patterns = ['o-o', 'o->', '<->', '-->']
        for edge in v:
            if not any(pattern in edge for pattern in valid_patterns):
                raise TypeError('Each edge must contain "o-o", "o->", "-->" or "<->". Hint: Ensure your edges match the expected patterns.')
        return v
    
class ResolvePartiallyDirectedEdgesParams(BaseModel):
    partially_directed: List[str]
    data: pd.DataFrame
    T: float
    beta_list: List[float]
    theta: float
    num_iterations: int

    class Config:
        arbitrary_types_allowed = True

    @field_validator('partially_directed')
    def check_partially_directed(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise TypeError('The "partially_directed" field must be a list of strings. Hint: Ensure you are passing a list where each element is a string.')
        valid_patterns = ['o-o', 'o->']
        for edge in v:
            if not any(pattern in edge for pattern in valid_patterns):
                raise TypeError('Each edge in "partially_directed" must contain "o-o", or "o->". Hint: Ensure your edges match the expected patterns.')
        return v

    @field_validator('data')
    def check_data(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise TypeError('The "data" field must be a pandas DataFrame. Hint: Ensure you are passing a DataFrame object from pandas.')
        return v

    @field_validator('T')
    def check_T(cls, v):
        if not isinstance(v, float):
            raise TypeError('The "T" field must be a float. Hint: Ensure you are passing a float value.')
        return v

    @field_validator('beta_list')
    def check_beta_list(cls, v):
        if not isinstance(v, list):
            raise TypeError('The "beta_list" field must be a list. Hint: Ensure you are passing a list.')
        if not all(isinstance(item, float) for item in v):
            raise TypeError('The "beta_list" field must be a list of floats. Hint: Ensure you are passing a list where each element is a float.')
        if not all(item >= 0 for item in v):
            raise ValueError('The "beta_list" field must be a list of floats greater than or equal to 0. Hint: Ensure you are passing a list where each element is a float greater than or equal to 0.')
        return v

    @field_validator('theta')
    def check_theta(cls, v):
        if not isinstance(v, float):
            raise TypeError('The "theta" field must be a float. Hint: Ensure you are passing a float value.')
        return v

    @field_validator('num_iterations')
    def num_iterations(cls, v):
        if not isinstance(v, int):
            raise TypeError('The "num_iterations" field must be an integer. Hint: Ensure you are passing an integer value.')
        if v <= 0:
            raise ValueError('The "num_iterations" field must be an integer greater than or equal to 0. Hint: Ensure you are passing an integer value greater than or equal to 0.')
        return v

class FCIParams(BaseModel):
    alpha: float
    verbose: bool = False
    show_progress: bool = False

    class Config:
        arbitrary_types_allowed = True

    @field_validator('alpha')
    def check_alpha(cls, v):
        if not isinstance(v, float):
            raise TypeError('The "alpha" field must be a float. Hint: Ensure you are passing a float value.')
        return v

    @field_validator('verbose', 'show_progress')
    def check_booleans(cls, v, field):
        if not isinstance(v, bool):
            raise TypeError(f'The "{field.name}" field must be a boolean. Hint: Ensure you are passing a boolean value.')
        return v

class CausalModelParams(BaseModel):
    alpha: float
    T: float = 0.1
    beta_list: List[float]
    theta: float
    verbose: bool = False
    LS_iter: int = 10
    save_graph: bool = True

    class Config:
        arbitrary_types_allowed = True

    @field_validator('alpha', 'T', 'theta')
    def check_float(cls, v, field):
        if not isinstance(v, float):
            raise TypeError(f'The "{field.name}" field must be a float. Hint: Ensure you are passing a float value.')
        return v

    @field_validator('beta_list')
    def check_beta_list(cls, v):
        if not isinstance(v, list):
            raise TypeError('The "beta_list" field must be a list. Hint: Ensure you are passing a list.')
        if not all(isinstance(item, float) for item in v):
            raise TypeError('The "beta_list" field must be a list of floats. Hint: Ensure you are passing a list where each element is a float.')
        if not all(item >= 0 for item in v):
            raise ValueError('The "beta_list" field must be a list of floats greater than or equal to 0. Hint: Ensure you are passing a list where each element is a float greater than or equal to 0.')
        return v

    @field_validator('verbose', 'save_graph')
    def check_booleans(cls, v, field):
        if not isinstance(v, bool):
            raise TypeError(f'The "{field.name}" field must be a boolean. Hint: Ensure you are passing a boolean value.')
        return v

    @field_validator('LS_iter')
    def check_ls_iter(cls, v):
        if not isinstance(v, int):
            raise TypeError('The "LS_iter" field must be an integer. Hint: Ensure you are passing an integer value.')
        if v <= 0:
            raise ValueError('The "LS_iter" field must be an integer greater than or equal to 0. Hint: Ensure you are passing an integer value greater than or equal to 0.')
        return v
