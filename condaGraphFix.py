# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 14:43:08 2017

@author: USER
"""
import os
import sys

def fixGraph(graph):
    path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz");
    paths = ("dot", "twopi", "neato", "circo", "fdp");
    paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths};
    graph.set_graphviz_executables(paths);
                                  
    return graph;
 