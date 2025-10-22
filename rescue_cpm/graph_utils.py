#!/usr/bin/env python3

from __future__ import annotations

from typing import List

import pydot
from pydantic import BaseModel, field_validator, ValidationError
import logging

# Initialize logger
logger = logging.getLogger(__name__)


class Edges_TO_PydotParams(BaseModel):
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
        valid_patterns = ['<->', '-->']
        for edge in v:
            if not any(pattern in edge for pattern in valid_patterns):
                raise TypeError(
                'Each edge must contain "-->" or "<->". Hint: Ensure your edges match the expected patterns.'
                )
        return v



class RescueGraphUtils():
    def __init__(self) -> None:
        logger.debug("Initializing RescueGraphUtils class.")
        pass

    @staticmethod
    def pydot_to_edges(pydot_graph: pydot.Dot) -> List[str]:
            """
            Convert a pydot graph to a list of edge strings in the required format.

            Args:
            pydot_graph (pydot.Dot): Pydot graph object.

            Returns:
            List[str]: List of edge strings.
            """
            logger.debug("Converting a pydot graph to a list of edges.")
            # Create a mapping from node IDs to labels
            id_to_label = {}
            for node in pydot_graph.get_nodes():
                node_id = str(node.get_name()).strip('"')
                label = node.get_attributes().get('label', node_id).strip('"')
                id_to_label[node_id] = label

            edges = []
            for edge in pydot_graph.get_edges():
                source_id = str(edge.get_source()).strip('"')
                destination_id = str(edge.get_destination()).strip('"')
                arrowhead = edge.get_attributes().get('arrowhead', 'none')
                arrowtail = edge.get_attributes().get('arrowtail', 'none')
                
                source_label = id_to_label[source_id]
                destination_label = id_to_label[destination_id]

                if arrowhead == 'odot' and arrowtail == 'odot':
                    edges.append(f"{source_label} o-o {destination_label}")
                elif arrowhead == 'normal' and arrowtail == 'none':
                    edges.append(f"{source_label} --> {destination_label}")
                elif arrowhead == 'normal' and arrowtail == 'normal':
                    edges.append(f"{source_label} <-> {destination_label}")
                elif arrowhead == 'normal' and arrowtail == 'odot':
                    edges.append(f"{source_label} o-> {destination_label}")
            logger.debug("Total edges discovered: %d", len(edges))    
            return edges

    @staticmethod
    def edges_to_pydot(edges: List[str]) -> pydot.Dot:
        """
        Convert a list of edge strings to a pydot graph object.

        Args:
        edges (List[str]): List of edge strings in the format 'source --> destination', 
            'source o-o destination', or 'source <-> destination'.

        Returns:
        pydot.Dot: Pydot graph object.
        """

        try:
            params = Edges_TO_PydotParams(edges=edges)
        except ValidationError as e:
            raise e
        
        graph = pydot.Dot(graph_type='digraph')
        nodes = set()
        for edge in edges:
            parts = edge.split()
            source = parts[0]
            destination = parts[-1]
            
            nodes.add(source)
            nodes.add(destination)
            
            if '-->' in edge:
                graph.add_edge(pydot.Edge(source, 
                                          destination, 
                                          arrowhead='normal',
                                          arrowtail='None', 
                                          dir='both'))
            elif 'o-o' in edge:
                graph.add_edge(pydot.Edge(source, 
                                          destination, 
                                          arrowhead='odot', 
                                          arrowtail='odot', 
                                          dir='both'))
            elif '<->' in edge:
                graph.add_edge(pydot.Edge(source, destination, 
                                          arrowhead='normal', 
                                          arrowtail='normal', 
                                          dir='both'))
            elif 'o->' in edge:
                graph.add_edge(pydot.Edge(source, 
                                          destination, 
                                          arrowhead='normal', 
                                          arrowtail='odot', 
                                          dir='both'))
        for node in nodes:
            graph.add_node(pydot.Node(node))
        return graph


    @staticmethod
    def nodes_with_no_edges(pydot_graph: pydot.Dot, columns: List[str]) -> pydot.Dot:
        """
        Add nodes to the pydot graph if they are not already present.

        Args:
            pydot_graph (pydot.Dot): The pydot graph to which nodes will be added.
            columns (List[str]): A list of column names representing the nodes to add.
        """
        # Get list of nodes
        nodes = pydot_graph.get_nodes()
        
        # Extract node names into a set for efficient lookup
        node_names = {node.get_name().strip('"') for node in nodes}
        
        # Add non-connected nodes to the graph
        for column in columns:
            if column not in node_names:
                pydot_graph.add_node(pydot.Node(column))
        return pydot_graph