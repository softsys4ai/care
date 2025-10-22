#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Tuple

from pydantic import ValidationError
import pydot
import pandas as pd
import numpy as np
import pydot
import networkx as nx
from networkx.drawing.nx_pydot import write_dot

from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from causallearn.utils.cit import fisherz, mv_fisherz, kci, chisq, gsq
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import (
    BackgroundKnowledge
)
from causallearn.search.FCMBased.lingam.utils import make_dot

from rescue_cpm.graph_utils import RescueGraphUtils

import logging

logger = logging.getLogger(__name__)

class CausalPerformanceModel:
    def __init__(
        self, 
        data: pd.DataFrame, 
        design_variables: List[str],
        objective_variables: List[str],
        is_design_var_independent: bool = True,
        kpi_and_constraints_variables: None | List[str] = None,
        forbidden_edges: None | list[Tuple[str, ...]]= None,
        required_edges: None | list[Tuple[str, ...]] = None,
        directLinGAM_prior_knowledge: None | np.ndarray = None,
        is_multifidelity: bool = False,
        fidelity_param_name: None | str = None,
        use_default_bk: bool = True
    ) -> None:
        r"""
        CausalPerformanceModel class for learning causal models from observational data.
        This class uses `causal-learn` library to perform causal discovery.
        
        Algorithms used: FCI and PC.

        Both alogorithms allows enforing structural constraints to the causal graph.
        Default structural constraints are:
            forbidden_edges:
                - design_variables --> design_variables
                - objective_variables, kpi_and_constraints_variables --> design_variables
            required_edges:
                - kpi_and_constraints_variables --> objective_variables
                - fidelity --> objective_variables, kpi_and_constraints_variables

        When using FCI, If the learned graph has partially directed edges, it resolves 
        those edges using information-theoretic measures and latent search. The final
        causal graph is Acyclic Directed Mixed Graph (ADMG). If that is the case,
        then FCI can not be used for causal inference as `do-why` do not support ADMG.

        When using PC, the unidirected edges are ignored and the final graph is a 
        Directed Acyclic Graph (DAG).

        Args:
            data (pd.DataFrame): The dataset.
            design_variables (List[str]): List of parameter column names.
            kpi_and_constraints_variables (None | List[str]): List of metric column names.
            objective_variables (List[str]): List of objective column names.
            is_design_var_independent (bool): Whether design variables are independent.
            use_default_bk (bool): Whether to use default background knowledge.
            directLinGAM_prior_knowledge (None | np.ndarray): Prior knowledge for directLiNGAM.
                NOTE: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Causal%20discovery%20methods%20based%20on%20constrained%20functional%20causal%20models/lingam.html
            forbidden_edges (None | list[Tuple[str, ...]]): List of tuples. 
                representing forbidden edges.
            required_edges (None | list[Tuple[str, ...]]): List of tuples.
            is_multifidelity (bool): Whether the model is multi-fidelity.
            fidelity_param_name (None | str): Name of the fidelity parameter.
            """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        if not all(isinstance(var, str) for var in design_variables):
            raise ValueError("All design variables must be strings.")

        if kpi_and_constraints_variables is not None:
            if not all(isinstance(var, str) for var in kpi_and_constraints_variables):
                raise ValueError("All metrics or constraints variables must be strings.")

        if not all(isinstance(var, str) for var in objective_variables):
            raise ValueError("All objective variables must be strings.")

        if forbidden_edges is not None:
            if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in forbidden_edges):
                raise ValueError("All forbidden edges must be tuples of (source, target).")

        if required_edges is not None:
            if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in required_edges):
                raise ValueError("All required edges must be tuples of (source, target).")

        logger.debug("Initializing CausalPerformanceModel class.")

        self.data = data
        self.design_variables = design_variables
        self.kpi_and_constraints_variables = kpi_and_constraints_variables or []
        self.objective_variables = objective_variables
        self.outcome_variables = self.objective_variables + self.kpi_and_constraints_variables
        
        self.forbidden_edges = forbidden_edges
        self.required_edges = required_edges
        self.directLinGAM_prior_knowledge = directLinGAM_prior_knowledge
        self.is_design_var_independent = is_design_var_independent

        self.use_default_bk = use_default_bk
        self.is_multifidelity = is_multifidelity

        self.colmap = {i: col for i, col in enumerate(self.data.columns)} 

        self.is_background_knowledge = False
        if self.forbidden_edges is not None or self.required_edges is not None:
            self.is_background_knowledge = True
        
        if self.is_background_knowledge:
            if self.use_default_bk:
                # raise only one of them can be used at the same time
                # mention required_edges and forbidden_edges
                raise ValueError("`use_default_bk` can't be used when `forbidden_edges` or " \
                "`required_edges` are specified.")

        if self.is_multifidelity:
            self.fidelity_param_name = fidelity_param_name
            if self.fidelity_param_name is None:
                raise ValueError("`fidelity_param_name` must be specified for " \
                "multi-fidelity models.")


    def _conditional_mutual_information(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> float:
        """
        Calculate conditional mutual information between two variables given a third variable.

        Args:
            x (np.ndarray): First variable.
            y (np.ndarray): Second variable.
            z (np.ndarray): Conditioning variable.

        Returns:
            float: Conditional mutual information score.
        """

        from .params import ConditionalMutualInformationParams
        try:
            params = ConditionalMutualInformationParams(
                x=x,
                y=y,
                z=z
            )
        except ValidationError as e:
            logger.error(f"Conditional Mutual Information parameter validation error: {e}")
            raise e
    
        logger.debug("Calculating conditional mutual information.")
        
        # Discretize the variables
        x_discrete = pd.cut(x, bins=10, labels=False)
        y_discrete = pd.cut(y, bins=10, labels=False)
        z_discrete = pd.cut(z, bins=10, labels=False)
        logger.debug("Discretized variables: x_discrete, y_discrete, z_discrete.")
        
        total_cmi = 0
        z_values = np.unique(z_discrete)
        logger.debug("Unique values in z: %s", z_values)
        
        for z_val in z_values:
            indices = np.where(z_discrete == z_val)
            x_sub = x_discrete[indices]
            y_sub = y_discrete[indices]
            if len(x_sub) > 0 and len(y_sub) > 0:
                cmi = mutual_info_score(x_sub, y_sub)
                total_cmi += len(x_sub) / len(x_discrete) * cmi
                logger.debug("CMI for z_val %d: %f", z_val, cmi)
        
        logger.debug("Total conditional mutual information: %f", total_cmi)
        return total_cmi

    def _calculate_residuals(
        self, 
        data: pd.DataFrame, 
        target: str
    ) -> np.ndarray:
        """
        Calculate residuals for a target variable using linear regression.
        The residuals are proxies for the exogenous noise variables E and E_hat.

        Args:
            data (pd.DataFrame): DataFrame containing the data.
            target (str): Target variable name.

        Returns:
            np.ndarray: Residuals of the prediction.
        """

        from .params import CalculateResidualsParams
        try:
            params = CalculateResidualsParams(
                data=data,
                target=target
            )
        except ValidationError as e:
            logger.error(f"Calculate Residuals parameter validation error: {e}")
            raise e
        logger.debug("Calculating residuals for target variable: %s", target)
        
        # Prepare the data
        X = data.drop(columns=[target])
        y = data[target]
        logger.debug("Data prepared. X shape: %s, y shape: %s", X.shape, y.shape)
        
        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)
        logger.debug("Linear regression model fitted.")
        
        # Make predictions
        predictions = model.predict(X)
        logger.debug("Predictions made.")
        
        # Calculate residuals
        residuals = y - predictions
        logger.debug("Residuals calculated: %s", residuals.values)
        
        return residuals.values

    def _entropy(self, z: np.ndarray) -> float:
        """
        Calculate the entropy of a variable.

        Steps to Calculate Entropy:
        1. Unique Values and Counts:
        - Determine the unique values in z and count the occurrences of each unique value.
        - This gives us the frequency distribution of z.
        
        2. Probability Distribution:
        - Convert the counts of each unique value into probabilities by dividing each count by the total number of elements in z.
        - This gives us the probability distribution of z.
        
        3. Entropy Calculation:
        - Using the probability distribution, compute the entropy.
        - Entropy is a measure of the uncertainty or randomness in the variable.
        - Mathematically, entropy H(z) is calculated as:
            H(z) = -sum(p_i * log(p_i))
            where p_i is the probability of the i-th unique value in z.        

        Args:
            z (np.ndarray): Variable.

        Returns:
            float: Entropy score.
        """


        from .params import EntropyParams
        try:
            params = EntropyParams(
                z=z
            )
        except ValidationError as e:
            logger.error(f"Entropy parameter validation error: {e}")
            raise e    
            
        logger.debug("Calculating entropy for the given variable.")
        unique_elements, counts = np.unique(z, return_counts=True)
        logger.debug("Unique elements: %s", unique_elements)
        logger.debug("Counts: %s", counts)
        
        probs = counts / len(z)
        logger.debug("Probabilities: %s", probs)
        
        entropy_value = -np.sum(probs * np.log(probs))
        logger.debug("Calculated entropy: %f", entropy_value)
        
        return entropy_value


    def _calculate_joint_distribution(
        self, 
        data: pd.DataFrame, 
        x: str, 
        y: str, 
        bins: int = 10
    ) -> np.ndarray:
        """
        Calculate the joint distribution p(x, y) from data.

        Args:
            data (pd.DataFrame): DataFrame containing the data.
            x (str): Column name for the first variable.
            y (str): Column name for the second variable.
            bins (int): Number of bins for the histogram.

        Returns:
            np.ndarray: Joint distribution matrix.
        """

        from .params import CalculateJointDistributionParams
        try:
            params = CalculateJointDistributionParams(
                data=data,
                x=x,
                y=y,
                bins=bins
            )
        except ValidationError as e:
            logger.error(f"Calculate Joint Distribution parameter validation error: {e}")
            raise e  
              
        joint_dist = np.histogram2d(data[x], data[y], bins=bins, density=True)[0]
        joint_dist /= joint_dist.sum()  # Normalize
        logger.debug("Calculating joint distribution p(x, y) from data.")
        return joint_dist

    def _latent_search(
        self, 
        x_support: range, 
        y_support: range, 
        z_support: range, 
        p_xy: np.ndarray, 
        q_z_given_xy_init: np.ndarray, 
        beta: float, 
        num_iterations: int
    ) -> np.ndarray:
        """
        Perform latent search to find the joint distribution q(x, y, z).

        Args:
            x_support (range): Support range for variable x.
            y_support (range): Support range for variable y.
            z_support (range): Support range for latent variable z.
            p_xy (np.ndarray): Joint distribution p(x, y).
            q_z_given_xy_init (np.ndarray): Initial distribution q(z|x, y).
            beta (float): Beta parameter for the update rule.
            num_iterations (int): Number of iterations for the latent search.

        Returns:
            np.ndarray: Joint distribution q(x, y, z).
        """

        from .params import LatentSearchParams
        try:
            params = LatentSearchParams(
                x_support=x_support,
                y_support=y_support,
                z_support=z_support,
                p_xy=p_xy,
                q_z_given_xy_init=q_z_given_xy_init,
                beta=beta,
                num_iterations=num_iterations
            )
        except ValidationError as e:
            logger.error(f"Latent Search parameter validation error: {e}")
            raise e        
        if beta <= 0:
            raise ValueError("beta must be >= 0")
        
        q_z_given_xy = q_z_given_xy_init.copy()
        q_xyz = np.zeros((len(x_support), len(y_support), len(z_support)))

        eps = np.finfo(float).eps

        logger.debug("Starting latent search with %d iterations.", num_iterations)
        for iteration in range(num_iterations):
            logger.debug("Iteration %d/%d", iteration + 1, num_iterations)
            
            # Step 4: Form the joint q_i(x, y, z) = q_i(z|x, y)p(x, y) for all x, y, z
            for x in range(len(x_support)):
                for y in range(len(y_support)):
                    for z in range(len(z_support)):
                        q_xyz[x, y, z] = q_z_given_xy[x, y, z] * p_xy[x, y]
            
            logger.debug("Calculated joint distribution q(x, y, z).")
            
            # Step 5: Calculate q_i(z|x) and q_i(z|y)
            q_z_given_x = np.zeros((len(x_support), len(z_support)))
            q_z_given_y = np.zeros((len(y_support), len(z_support)))
            q_z = np.zeros(len(z_support))
            
            for x in range(len(x_support)):
                for z in range(len(z_support)):
                    denom_x = np.sum(q_xyz[x, :, :]) + eps
                    q_z_given_x[x, z] = np.sum(q_xyz[x, :, z]) / denom_x
            
            for y in range(len(y_support)):
                for z in range(len(z_support)):
                    denom_y = np.sum(q_xyz[:, y, :]) + eps
                    q_z_given_y[y, z] = np.sum(q_xyz[:, y, z]) / denom_y
            
            for z in range(len(z_support)):
                q_z[z] = np.sum(q_xyz[:, :, z])

            logger.debug("Calculated marginal distributions q(z|x) and q(z|y).")

            # Initialize q_z_given_xy_new for update
            q_z_given_xy_new = np.zeros((len(x_support), len(y_support), len(z_support)))

            # Step 6: Update q_i+1(z|x, y) using the formula
            for x in range(len(x_support)):
                for y in range(len(y_support)):
                    for z in range(len(z_support)):
                        F_xy = np.sum(q_z_given_x[x, :] * q_z_given_y[y, :] / np.maximum(q_z[:]**(1-beta), eps))
                        if F_xy > 0:
                            q_z_given_xy_new[x, y, z] = (1 / np.clip(F_xy, eps, None)) * (q_z_given_x[x, z] * q_z_given_y[y, z] / np.maximum(q_z[z]**(1-beta), eps))
                        else:
                            q_z_given_xy_new[x, y, z] = 0
            
            q_z_given_xy = q_z_given_xy_new.copy()
            logger.debug("Updated conditional distribution q(z|x, y) for iteration %d.", iteration + 1)

        logger.debug("Latent search completed.")
        return q_xyz


    def _identify_partially_directed_edges(
        self, 
        edges: List[str]
    ) -> List[str]:
        """
        Identify partially directed edges from the list of edges.

        Args:
            edges (List[str]): List of edges.

        Returns:
            List[str]: List of partially directed edges.
        """

        from .params import IdentifyPartiallyDirectedEdgesParams
        try:
            params = IdentifyPartiallyDirectedEdgesParams(
                edges=edges
            )
        except ValidationError as e:
            logger.error(f"Identify Partially Directed Edges parameter validation error: {e}")
            raise e  
             
        logger.debug("Identifying partially directed edges.")
        partially_directed = []
        for edge in edges:
            if 'o-o' in edge or 'o->' in edge:
                partially_directed.append(edge)
                logger.debug("Identified partially directed edge: %s", edge)
            else:
                logger.debug("Edge is not partially directed: %s", edge)
        logger.debug("Total partially directed edges identified: %d", len(partially_directed))
        return partially_directed

    
    def resolve_partially_directed_edges(
        self, 
        partially_directed: List[str], 
        data: pd.DataFrame, 
        T: float,
        beta_list: List[float], 
        theta: float, 
        num_iterations: int
    ) -> List[str]:
        """
        Resolve partially directed edges using latent search.

        Args:
            partially_directed (List[str]): List of partially directed edges.
            data (pd.DataFrame): DataFrame containing the data.
            T (float): Threshold for conditional mutual information.
            beta_list (List[float]): List of beta values for the latent search.
            theta (float): Threshold for entropy.
            num_iterations (int): Number of iterations for the latent search.

        Returns:
            List[str]: List of resolved edges.
        """

        from .params import ResolvePartiallyDirectedEdgesParams
        try:
            params = ResolvePartiallyDirectedEdgesParams(
                partially_directed=partially_directed,
                data=data,
                T=T,
                beta_list=beta_list,
                theta=theta,
                num_iterations=num_iterations
            )
        except ValidationError as e:
            logger.error(f"Resolve Partially Directed Edges parameter validation error: {e}")
            raise e          
              
        resolved_edges = []
        k = 10  # Number of bins for joint distribution
        x_support = range(k)
        y_support = range(k)
        z_support = range(k)

        logger.debug("Starting to resolve partially directed edges.")

        for edge in partially_directed:
            nodes = edge.split()
            logger.debug("Processing edge: %s", edge)
            try:
                x = data[nodes[0]].values
                y = data[nodes[2]].values
            except KeyError as e:
                logger.error("Node not found in data: %s", e)
                raise ValueError(f"Node not found in data: {e}")

            p_xy = self._calculate_joint_distribution(data, nodes[0], nodes[2], bins=k)
            q_z_given_xy_init = np.random.rand(len(x_support), len(y_support), k)
            q_z_given_xy_init /= q_z_given_xy_init.sum(axis=2, keepdims=True)  # Normalize

            I_values = []
            H_values = []

            for beta in beta_list:
                # Step 4: q^(i)(x, y, z) <- LatentSearch(q^(i)_0(z|x, y), beta_i)
                q_xyz = self._latent_search(x_support, y_support, z_support, p_xy, q_z_given_xy_init, beta, num_iterations)
                z = q_xyz.argmax(axis=2).flatten()  # Get the most likely value of Z

                # Step 5: Calculate I^(i)(X; Y | Z) and H^(i)(Z)
                H_Z = self._entropy(z)
                H_values.append(H_Z)
                I_XY_given_Z = self._conditional_mutual_information(x, y, z)
                I_values.append(I_XY_given_Z)

                logger.debug("Calculated H(Z): %f and I(X; Y | Z): %f for beta: %f", H_Z, I_XY_given_Z, beta)

            # Step 6: S = {i: I^(i)(X; Y | Z) <= T}
            S = [i for i in range(len(I_values)) if I_values[i] <= T]

            # Step 7: if min(H^(i)(Z) : i in S) > theta or S = empty then
            # If S is empty, it means no latent variable Z sufficiently explains the dependency (I(X; Y | Z) > T for all Z)
            # If min(H(Z)) > theta, it indicates that Z has high entropy, suggesting Z is not a clear explanatory variable
            if not S or (S and min(H_values[i] for i in S) > theta):
                logger.debug("High entropy or no suitable latent variable found for edge: %s", edge)
                # High entropy implies Z is too noisy or not well-defined
                # Resolve the edge based on the original partially directed edge type
                if 'o->' in edge:
                    resolved_edges.append(f"{nodes[0]} --> {nodes[2]}")
                    logger.debug(f"Resolved ({edge}) to ({nodes[0]} --> {nodes[2]})")
                else:
                    # For 'o-o' edges, use noise variables E and E_hat to decide the direction
                    E = self._calculate_residuals(data, nodes[2])  # Noise independent of nodes[2]
                    E_hat = self._calculate_residuals(data, nodes[0])  # Noise impacting nodes[0]
                    H_E = self._entropy(E)
                    H_E_hat = self._entropy(E_hat)

                    logger.debug("Calculated entropy for noise variables H(E): %f and H(E_hat): %f", H_E, H_E_hat)

                    if H_E < H_E_hat:
                        resolved_edges.append(f"{nodes[0]} --> {nodes[2]}")
                        logger.debug(f"Resolved ({edge}) to ({nodes[0]} --> {nodes[2]})")
                    else:
                        resolved_edges.append(f"{nodes[2]} --> {nodes[0]}")
                        logger.debug(f"Resolved ({edge}) to ({nodes[2]} --> {nodes[0]})")
            else:
                logger.debug("Low entropy indicates a good explanatory variable for edge: %s", edge)
                # Low entropy implies Z is a good explanatory variable
                # Keep the edge bidirectional
                resolved_edges.append(f"{nodes[0]} <-> {nodes[2]}")
                logger.debug(f"Resolved ({edge}) to ({nodes[0]} <-> {nodes[2]})")
        
        logger.debug("Finished resolving edges. Total resolved edges: %d", len(resolved_edges))
        return resolved_edges


    def _add_structural_constraints(self) -> List[Tuple]:
        """
        Default backgound knowledge strategy
        Adds constraints to exclude edges that are not possible in the graph.
        This function generates a list of forbidden edges, which represent the 
        connections that are not allowed between different types of nodes in the graph.
        forbidden_edges:
            - kpi_and_constraints_variables --> design_variables
            - outcome_variables --> design_variables
        required_edges:
            - kpi_and_constraints_variables --> outcome_variables
            - fidelity --> outcome_variables

        Returns:
            list: A list of tuples, where each tuple represents a forbidden edge.
        """
        forbidden_edges = []
        required_edges = []

        # No cur_elem (design_variables) --> design_variables
        if self.is_design_var_independent:
            for opt in self.design_variables:
                for cur_elem in self.design_variables:
                    if cur_elem != opt:
                        forbidden_edges.append((cur_elem, opt))

        # No cur_elem (outcome_variables) --> design_variables
        for opt in self.design_variables:
            for cur_elem in self.outcome_variables:
                if cur_elem != opt:
                    forbidden_edges.append((cur_elem, opt))

        ## Required cur_elem (kpi_and_constraints_variables) --> objective_variables
        if self.kpi_and_constraints_variables:
            for obj in self.objective_variables:
                for cur_elem in  self.kpi_and_constraints_variables:
                    if cur_elem != obj:
                        required_edges.append((cur_elem, obj))  

        # No kpi_and_constraints_variables --> cur_elem (design_variables)
        if self.kpi_and_constraints_variables:
            for met in  self.kpi_and_constraints_variables:
                for cur_elem in self.design_variables:
                    if cur_elem != met:
                        forbidden_edges.append((met, cur_elem))   

        if self.is_multifidelity:
            # Required fidelity_param --> outcome_variables
            for obj in self.outcome_variables:
                for cur_elem in self.design_variables:
                    if cur_elem == self.fidelity_param_name:
                        required_edges.append((cur_elem, obj))  

        logger.debug("Forbidden edges generated: %s", forbidden_edges)     
        logger.debug("Required edges generated: %s", required_edges)        
        return forbidden_edges, required_edges
    

    def _fci(
        self, 
        alpha: float, 
        independence_test_method, 
        verbose: bool = False, 
        show_progress: bool = False,
        save_graph: bool = False
    ) -> pydot.Dot:
        """
        Learns the causal model using Fast Causal Inference (FCI) algorithm.

        This function uses the FCI algorithm to learn the causal structure from the data. It allows for specifying forbidden edges
        to incorporate prior knowledge into the model. The learned causal graph is saved as a PNG image, and a list of FCI edges
        is returned.

        Arguments:
            self: The instance of the class containing the data and column mappings.
            forbidden_edges (list): A list of tuples representing edges that should be forbidden in the causal graph. Each tuple contains two elements representing the nodes.
            alpha (float): The significance level for the conditional independence tests.
            verbose (bool): If True, enables verbose output during the FCI algorithm execution.
            show_progress (bool): If True, displays a progress bar during the FCI algorithm execution.
            save_graph (bool): If True, saves the learned FCI graph as a PNG image.

        Returns:
            pydot_g : Dot

        Raises:
            ValueError: If the causal model cannot be learned.
        """

        from .params import FCIParams
        try:
            params = FCIParams(
                alpha=alpha,
                verbose=verbose,
                show_progress=show_progress
            )
        except ValidationError as e:
            logger.error(f"FCI parameter validation error: {e}")
            raise e        
        
        if independence_test_method == 'fisherz':
            independence_test_method = fisherz
        elif independence_test_method == 'chisq':
            independence_test_method = chisq
        elif independence_test_method == 'gsq':
            independence_test_method = gsq
        elif independence_test_method == 'kci':
            independence_test_method = kci
        elif independence_test_method == 'mv_fisherz':
            independence_test_method = mv_fisherz
        else:
            raise ValueError(f"Unknown independence test method: {independence_test_method}")


        df_arr = np.array(self.data)
        logger.debug("Performing Fast Causal Inference (FCI).")
        G, edges = fci(
            dataset=df_arr, 
            independence_test_method=independence_test_method, 
            alpha=alpha, verbose=verbose, 
            show_progress=show_progress
        )
        nodes = G.get_nodes()
        if self.is_background_knowledge or self.use_default_bk:
            bk = BackgroundKnowledge()
            if self.use_default_bk:
                forbidden_edges, required_edges = self._add_structural_constraints()
            else:
                forbidden_edges, required_edges = self.forbidden_edges, self.required_edges

            if forbidden_edges is not None:
                for ce in forbidden_edges:
                    f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
                    s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]
                    bk.add_forbidden_by_node(nodes[f], nodes[s])
            if required_edges is not None:
                for ce in required_edges:
                    f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
                    s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]
                    bk.add_required_by_node(nodes[f], nodes[s])
        else:
            bk = None
        G, edges = fci(
                dataset=df_arr, 
                independence_test_method=independence_test_method, 
                alpha=alpha, 
                verbose=verbose, 
                show_progress=show_progress, 
                background_knowledge=bk
        )
        logger.debug("Causal model learning completed.")
        PAG = GraphUtils.to_pydot(G, edges, labels=self.data.columns)   
        if save_graph:
            PAG.write_png("FCI.png")
        return PAG
    

    def _pc(
        self, 
        alpha: float, 
        independence_test_method, 
        verbose: bool = False, 
        show_progress: bool = False
    ) -> pydot.Dot:
        """
        Learns the causal model using Fast Causal Inference (FCI) algorithm.

        This function uses the FCI algorithm to learn the causal structure from the data. It allows for specifying forbidden edges
        to incorporate prior knowledge into the model. The learned causal graph is saved as a PNG image, and a list of FCI edges
        is returned.

        Arguments:
            forbidden_edges (list): A list of tuples representing edges that should be forbidden in the causal graph. Each tuple contains two elements representing the nodes.
            alpha (float): The significance level for the conditional independence tests.
            verbose (bool): If True, enables verbose output during the FCI algorithm execution.

        Returns:
            pydot_g : Dot

        Raises:
            ValueError: If the causal model cannot be learned.
        """

        from .params import FCIParams
        try:
            params = FCIParams(
                alpha=alpha,
                verbose=verbose,
                show_progress=show_progress
            )
        except ValidationError as e:
            logger.error(f"FCI parameter validation error: {e}")
            raise e        
        
        if independence_test_method == 'fisherz':
            independence_test_method = fisherz
        elif independence_test_method == 'chisq':
            independence_test_method = chisq
        elif independence_test_method == 'gsq':
            independence_test_method = gsq
        elif independence_test_method == 'kci':
            independence_test_method = kci
        elif independence_test_method == 'mv_fisherz':
            independence_test_method = mv_fisherz
        else:
            raise ValueError(f"Unknown independence test method: {independence_test_method}")


        df_arr = np.array(self.data)
        logger.debug("Performing PC causal discovery.")
        G = pc(
                data=df_arr, 
                alpha=alpha, 
                indep_test=independence_test_method, 
                stable=True, 
                uc_rule=0, 
                uc_priority=2, 
                verbose=verbose, 
                show_progress=show_progress)
        nodes = G.G.get_nodes()
        if self.is_background_knowledge or self.use_default_bk:
            bk = BackgroundKnowledge()
            if self.use_default_bk:
                forbidden_edges, required_edges = self._add_structural_constraints()
            else:
                forbidden_edges, required_edges = self.forbidden_edges, self.required_edges
            if forbidden_edges is not None:
                for ce in forbidden_edges:
                    f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
                    s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]
                    bk.add_forbidden_by_node(nodes[f], nodes[s])
            if required_edges is not None:
                for ce in required_edges:
                    f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
                    s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]
                    bk.add_required_by_node(nodes[f], nodes[s])
        else:
            bk = None
        G = pc(
                data=df_arr, 
                alpha=alpha, 
                indep_test=independence_test_method, 
                stable=True, 
                uc_rule=0, 
                uc_priority=2, 
                verbose=verbose, 
                show_progress=show_progress, 
                background_knowledge=bk
        )
        pydot = GraphUtils.to_pydot(G.G, G.G.get_graph_edges(), labels=self.data.columns)     
        # pydot.write_png("PC.png")
        logger.debug("Causal model learning completed.")      
        return pydot


    def _prior_knowledge_matrix(
        self,
        labels: List[str],
        design_var_names: List[str], 
        objectives_var_names: List[str]
    ) -> np.ndarray:
        """
        Create prior knowledge matrix for DirectLiNGAM with domain constraints.
        
        Args:
            labels: List of variable names in order
            design_var_names: List of design variable names  
            objectives_var_names: List of objective variable names
            
        Returns:
            prior_knowledge: numpy array with constraints applied
        """
        n_vars = len(labels)
        # Initialize with no prior knowledge
        prior_knowledge = np.full((n_vars, n_vars), -1)  
        # Get indices for design and objective variables
        design_indices = [labels.index(var) for var in design_var_names]
        objective_indices = [labels.index(var) for var in objectives_var_names]
        kpi_and_constraints_indices = [labels.index(var) for var in self.kpi_and_constraints_variables]
        
        # Apply constraint: objectives not--> design variables
        for design_idx in design_indices:
            for obj_idx in objective_indices:
                prior_knowledge[obj_idx, design_idx] = 0

        # Apply constraint: kpi_and_constraints not--> design variables
        for design_idx in design_indices:
            for met_idx in kpi_and_constraints_indices:
                prior_knowledge[met_idx, design_idx] = 0

        if self.is_design_var_independent:
            # Apply constraint: design variables not--> design variables
            for i in design_indices:
                for j in design_indices:
                    if i != j:
                        prior_knowledge[i, j] = 0

        # Apply constraint: kpi_and_constraints --> objectives
        if self.kpi_and_constraints_variables:
            for met_idx in kpi_and_constraints_indices:
                for obj_idx in objective_indices:
                    prior_knowledge[met_idx, obj_idx] = 1
        
        return prior_knowledge

    def _DirectLiNGAM(
        self, 
        random_state: int | None
    ) -> pydot.Dot:
        labels = [f'{col}' for i, col in enumerate(self.data.columns)]
        if self.directLinGAM_prior_knowledge is not None:
            prior_knowledge = self.directLinGAM_prior_knowledge
        else:
            prior_knowledge = self._prior_knowledge_matrix(
                labels=labels,
                design_var_names=self.design_variables, 
                objectives_var_names=self.objective_variables
            )
         # Fit the DirectLiNGAM model with prior knowledge
        model_lingam = lingam.DirectLiNGAM(
            random_state=random_state, 
            prior_knowledge=prior_knowledge
        )
        model_lingam.fit(self.data)
        # Future me: for some reason the adjacency matrix 
        # must be transposed to get the correct direction when using
        # make_dot
        dot_graph = make_dot(
            adjacency_matrix=model_lingam.adjacency_matrix_.T, 
            labels=labels
        )
        dot_source = dot_graph.source
        # Debug only
        # dot_graph.render('lingam_graph', format='png', cleanup=True)
        return pydot.graph_from_dot_data(dot_source)[0]

    def learn(
        self, 
        alpha: float = 0.05, 
        independence_test_method='fisherz', 
        causal_discovery: str = 'PC',
        random_state_lingam: int | None = None,
        verbose: bool = False, 
        show_progress: bool = False,
        save_graph: bool = False, 
        beta_list: None | List[float] = [0.5, 0.6], 
        theta: None | float = 0.8,         
        T: None | float = 0.2, 
        latentsearch_maxiter: None | int = 10,    
        save_raw_fci_graph: bool = False     
    ) -> pydot.Dot:
        """
        Learns the causal model using the specified method (FCI or PC) and parameters.
        This function performs the following steps:
        1. Performs causal discovery using the specified method (FCI or PC).
        2. Identifies partially directed edges in the learned graph (only FCI).
        3. Resolves partially directed edges using latent search (only FCI).
        4. For PC, undirected edges are ignored.
        5. Returns the final causal graph as Dot.

        Args:
            alpha (float): The significance level for the conditional independence tests.
            T (float): Threshold for conditional mutual information.
                Determines the threshold below which I(X; Y | Z) is considered 
                low enough to indicate that Z explains the dependency between X and Y. 
                Lower values make the condition stricter.            
            beta_list (List[float]): List of beta values for the latent search.
                Used in the update rule during latent search to balance the 
                contributions of different components in the joint distribution. 
                Higher beta values place more emphasis on the observed joint distribution.
            theta (float): Threshold for entropy.
                Sets the threshold for the entropy of the latent variable Z. 
                If H(Z) exceeds this value, Z is considered too noisy or poorly defined. 
                Lower values make it stricter to accept Z as a valid explanatory variable.
            verbose (bool): If True, enables verbose output during the FCI algorithm execution.
            latentsearch_maxiter (int): Number of iterations for the latent search.
                Determines how many iterations the latent search algorithm performs. 
                More iterations allow the algorithm to refine the 
                joint distribution q(x, y, z) more thoroughly.
            save_graph (bool): Flag to save the causal graph in png format.

        Returns:
            pydot.Dot: Causal graph.
        """      
        
        if causal_discovery == 'FCI':
            fci_parms = [alpha, independence_test_method, verbose, show_progress]
            for pfci in fci_parms:
                if pfci is None:
                    raise ValueError(f"{pfci} parameters cannot be None. "
                                     "Please provide valid values."
                    )
            G = self._fci(
                alpha=alpha, 
                independence_test_method=independence_test_method, 
                verbose=verbose, 
                show_progress=show_progress,
                save_graph=save_raw_fci_graph
            )  
            edges = RescueGraphUtils.pydot_to_edges(G)
            logger.debug("FCI edges: %s", edges)
            partially_directed = self._identify_partially_directed_edges(edges)    
            if len(partially_directed) != 0:
                di_bi_edges = []
                for edge in edges:
                    if edge not in partially_directed:
                        di_bi_edges.append(edge)
                # Step 4: Resolve partially directed edges
                resolved_edges = self.resolve_partially_directed_edges(
                    partially_directed, 
                    self.data, 
                    T, 
                    beta_list, 
                    theta, 
                    latentsearch_maxiter
                )
                # Combine fully directed and resolved edges
                final_edges = di_bi_edges + resolved_edges
                edges_to_pydot = RescueGraphUtils.edges_to_pydot(final_edges)
                # Finally adding the nodes with no edges
                causal_graph_dot = RescueGraphUtils.nodes_with_no_edges(
                    edges_to_pydot, 
                    self.data.columns.to_list()
                )
            else:
                causal_graph_dot = G    

        elif causal_discovery == 'PC':
            G = self._pc(
                alpha=alpha, 
                independence_test_method=independence_test_method, 
                verbose=verbose, 
                show_progress=show_progress)
            # TODO directly covert it nx from dot
            # Future me: Conversions are necessary due to nx.nx_pydot.from_pydot error
            edges = RescueGraphUtils.pydot_to_edges(G)
            logger.debug("PC edges: %s", edges)
            edges_to_pydot = RescueGraphUtils.edges_to_pydot(edges)
            causal_graph_dot = RescueGraphUtils.nodes_with_no_edges(
                edges_to_pydot, 
                self.data.columns.to_list()
            )

        elif causal_discovery == 'DirectLiNGAM':
            causal_graph_dot = self._DirectLiNGAM(
                random_state=random_state_lingam
            )

        else:
            raise NotImplementedError("Causal discovery method not found!")
         
        logger.debug("Causal Peformance Model learning completed!")    
        # Future me: This is experimental
        # if self.is_multifidelity:
        #     self._add_fidelity_edges(
        #         causal_graph=causal_graph_nx, 
        #         fidelity_param_name=self.fidelity_param_name)
        if save_graph:
            causal_graph_dot.write_png(f"RscueLearn_{causal_discovery}.png")
        return causal_graph_dot
    

    def _add_fidelity_edges(
        self,
        causal_graph: nx.DiGraph,
        fidelity_param_name: str,
    ) -> None:
        """
        Add edges from an existing fidelity node to given outcome variables.
        This enforces the background knowledge that the fidelity parameter 
        affects the outcome variables.

        Args:
            causal_graph: The learned directed graph (nx.DiGraph).
            fidelity_param_name: Name of the fidelity node (must already exist in the graph).
        """
        if fidelity_param_name not in causal_graph.nodes:
            raise ValueError(f"Fidelity node '{fidelity_param_name}' not found in graph.")

        # Create edges and add them
        for var in self.outcome_variables:
            if not causal_graph.has_edge(fidelity_param_name, var):
                causal_graph.add_edge(
                    fidelity_param_name,
                    var,
                    arrowhead='normal',
                    arrowtail='none',
                    dir='both'
                )
 
