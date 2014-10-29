# -*- coding: utf-8 -*-
"""
Implementation of methods for network-based KDE based on the paper

Okabe A, Satoh T & Sugihara K (2009). 'A kernel density estimation method for networks, 
its computational method and a GIS‐based tool', International Journal of Geographical 
Information Science, 23:1, 7-32

The intention is that this should be used in conjunction with the ITNStreetNet
class defined elsewhere, which implements network construction and processing.

Given such an object, and a number of points for which the positions on the network
have been found by the relevant methods, these tools can be used to perform network
KDE calculations.

Author: Toby Davies
"""

import networkx as nx
from collections import defaultdict
import bisect as bs
from network import itn
from kernels import LinearKernel


#A helper function to do network searching from within the class
def all_paths_source_targets(G, source, targets, cutoff=None, weight='length'):
    '''
    This finds all paths of length <= cutoff between a given source node and any
    node within targets. The full paths, together with their lengths, are returned
    in a dictionary indexed by target.
    '''

    paths=defaultdict(list)

    #Bail out in the trivial case
    if cutoff == 0:
        return paths

    #Set up three structures to monitor the state of the search

    #A list which monitors the current state of the path
    current_path = [source]

    #A list whih records the distance to each step on the current path
    dist = [0]

    #A stack which records the next nodes to be searched. Each item in the stack
    #is a generator over the neighbours of a searched node. At each iteration it
    #returns the node and the weight of the connecting edge.
    stack = [( (e1,atts[weight]) for e0,e1,atts in G.edges(source,data=True))]

    #The stack will empty when the source has been exhausted
    while stack:

        #Take the most recent exploration area added to the stack (stack is last-in-first-out)
        successors = stack[-1]

        #Get the next node from the generator, and its weight
        successor, edge_weight = next(successors,(None,None))

        if successor is None:

            #If the generator has been exhausted, there are no more nodes to search
            #down this strand, so the algorithm backtracks.

            #Remove the exhausted generator from the stack
            stack.pop()
            #Backtrack to the previous position on the current path
            current_path.pop()
            #Adjust the distance list to represent the new state of current_path too
            dist.pop()

        #Otherwise, check whether the candidate node is within cutoff of the source.
        #Do this by adding the candidate link weight to the length of current_path
        elif dist[-1]+edge_weight <= cutoff:

            #If condition passed, we have a viable node

            if successor in targets:

                #If the viable node is one of the targets, record the path that we
                #took to reach it, together with its length, in a tuple. Add it
                #to the relevant entry in the paths dictionary
                paths[successor].append((current_path+[successor], dist[-1]+edge_weight))

            if successor not in current_path:

                #We also want to continue the search down this path. As long as
                #it does not cause a loop (hence the check for presence in current_path)
                #we add it to the stack so that this route is explored next
                stack.append(((e1,atts[weight]) for e0,e1,atts in G.edges(successor,data=True)))
                current_path.append(successor)
                dist.append(dist[-1]+edge_weight)

    return paths


class EqualSplitKernel():

    '''
    The equal split kernel function

    NOTE: This version assumes the network contains no cycles of length <2h, as
    in the paper.

    The class is initialised by passing the following:
        - an ITNStreetNet class
        - a set of points (kernel centres) whose locations on the network have
        been found. This should be specified as a dictionary, indexed by a unique
        point_id (can be anything as long as it does not clash with FID of street
        network featues) and where every item is (closest_edge,dist_along), as
        found by the closest_segments_euclidean() method.
        - a kernel bandwidth h
        - a univariate kernel function

    As soon as the class is initialises, a new network object g_aug is built. This
    graph is exactly the same as the street network, but with a node added for every
    kernel centre. That is, the original edges are split whenever a kernel centre
    appears, with a node (of degree 2) inserted. The node label is the same as the
    point_id.

    The reason for doing this is that is makes the task of finding all kernel centres
    within h of a given network point much easier - we can use classical network
    search methods which grow trees out of the source point in question.

    The evaluation of the kernel is then fairly trivial. There are two possibilities:
        - evaluation at one of the kernel centres (likely use in SEPP)
        - evaluation at any other point

    The first case is dealt with by evaluate_point(). You take the source point
    and do a depth-limited shortest path search on g_aug. Considering only the
    target nodes that are kernel centres, you then know all kernel centres within h
    of the source, plus the distances, plus the shortest paths needed to get there.
    It is then trivial to pass these to the kernel function and adjust the result
    for the degrees of the intermediate nodes (all the extra nodes we added make no
    difference because they are degree 2).

    The second case is slightly more complicated. A temporary new node 'EVAL_POINT_NODE'
    is created in the relevant position on g_aug. That is not as easy as it could
    be, because the position is specified before we started messing around with the
    network by adding extra nodes etc. So in some cases we have to find where it
    falls relative to the new kernel centre nodes. Once that has been done, it is
    (almost) as trivial as the first case to do the evaluation.

    In the SEPP case, I imagine it would be used as follows:
        - Initialise this class with all possible points
        - Perform a points_within_bandwidth operation for each one, and store the
        results - distances, and degree sequence of intermediate nodes - in a lookup.
        - Whenever we need to evaluate at a given point, use the above lookup,
        including only those points classified as background (or whatever) at any
        point

    '''
    def __init__(self, street_net, points, h):

        '''
        Initialise the class, do some housekeeping, and build the augmented network
        '''
        self.street_net = street_net
        self.points = points
        self.h = h
        self.kernel_univ = LinearKernel(h)

        self.build_augmented_network()


    def build_augmented_network(self):

        #Start with a straight copy of the street network
        g_aug=self.street_net.g.copy()

        #Initialise a lookup - this is going to store, for each edge, information
        #about the kernel centres on that edge
        edge_points=defaultdict(list)


        for point_id in self.points:

            edge,dist_along=self.points[point_id]

            #Make a unique ID triplet for the edge - it needs all this, rather than
            #just the FID, for later operations
            edge_id=(edge[2]['orientation_neg'],edge[2]['orientation_pos'],edge[2]['fid'])

            #Add to the list of the relevant edge both the point_id and its distance
            #from each of the end-points of the edge
            edge_points[edge_id].append((point_id,dist_along))


        #Now, for each edge that has points on it, we order the points in sequence,
        #from the negative end to the positive end, then make a new series of edges
        #linking each successive point. As part of this process a new node is implicitly
        #created for each point. Finally destroy the original edge
        for edge_id in edge_points:

            points_sequence=edge_points[edge_id]

            #Order the points according to how far they are from the negative end (edge_id[0])
            points_sequence.sort(key=lambda x: x[1][edge_id[0]])

            #Join the negative end of the original edge to the first point in the sequence
            g_aug.add_edge(edge_id[0],points_sequence[0][0],length=points_sequence[0][1][edge_id[0]])

            #Join the positive end of the original edge to the last point in the sequence
            g_aug.add_edge(edge_id[1],points_sequence[-1][0],length=points_sequence[-1][1][edge_id[1]])

            #Go through every intermediate pair of points (maybe zero)
            for i in range(len(points_sequence)-1):

                dist_between=points_sequence[i+1][1][edge_id[0]]-points_sequence[i][1][edge_id[0]]

                #Add a new edge with the correct length
                g_aug.add_edge(points_sequence[i][0],points_sequence[i+1][0],length=dist_between)

            #Destroy the original edge
            g_aug.remove_edge(edge_id[0],edge_id[1],key=edge_id[2])

        self.g_aug=g_aug

        #Keep a note of which of the original edges had points on it - useful later
        self.edge_points=edge_points


    def points_within_bandwidth(self,eval_node):
        '''
        Finds paths and distances to all points within h of a given node.

        For any point within h, the algorithm finds ALL unique paths and their distances,
        so if there are cycles then multiple paths to the same point might be found
        '''

        paths = all_paths_source_targets(self.g_aug,eval_node,self.points,cutoff=self.h,weight='length')

        point_paths={k: v for k, v in paths.iteritems() if k!=eval_node}

#        #Do a standard Dijkstra shortest path
#        distance,path=nx.single_source_dijkstra(self.g_aug,eval_node,cutoff=self.h,weight='length')
#
#        #Filter this so that we only retain information about nodes which represent
#        #points. Also ignore the source node.
#        point_distance={k: v for k, v in distance.iteritems() if k in self.points and k!=eval_node}
#        point_path={k: v for k, v in path.iteritems() if k in self.points and k!=eval_node}

        return point_paths



    def add_eval_point_node(self,eval_point):
        '''
        Adds a (temporary) extra node to the network, representing some point where
        we need to evaluate the KDE. This is not intended to be used if the evaluation
        is to be done at an existing point (kernel centre).
        '''

        #First work out where it is
        edge,dist_along=eval_point
        edge_id=(edge[2]['orientation_neg'],edge[2]['orientation_pos'],edge[2]['fid'])

        #Task is different depending on whether existing points already on that edge
        if edge_id in self.edge_points:

            #If they are, we need to find where the new one should be inserted
            #So take the sequence which have already been inserted on this edge
            points_sequence=self.edge_points[edge_id]

            #Make a list for how long each is along the edge (plus end points)
            dist_sequence=[0]+[x[1][edge_id[0]] for x in points_sequence]+[edge[2]['length']]

            #Make corresponding sequence of nodes in the augmented graph
            node_sequence=[edge_id[0]]+[x[0] for x in points_sequence]+[edge_id[1]]

            #Locate the insertion point, given that evaluation point is dist_along[edge_id[0]]
            #along the original edge
            sequence_pos=bs.bisect_left(dist_sequence,dist_along[edge_id[0]])

            #Make a new pair of edges
            self.g_aug.add_edge(node_sequence[sequence_pos-1],'EVAL_POINT_NODE',length=dist_along[edge_id[0]]-dist_sequence[sequence_pos-1])
            self.g_aug.add_edge(node_sequence[sequence_pos],'EVAL_POINT_NODE',length=dist_sequence[sequence_pos]-dist_along[edge_id[0]])

            #Take note of all details of the edge to be removed, which will be needed
            #when it is eventually restored
            removed_edge=(node_sequence[sequence_pos-1],node_sequence[sequence_pos],0)
            removed_edge_atts=self.g_aug[node_sequence[sequence_pos-1]][node_sequence[sequence_pos]][0]

            #Remove the original edge
            self.g_aug.remove_edge(node_sequence[sequence_pos-1],node_sequence[sequence_pos],0)

        else:

            #If the edge does not have any points on it, the task is easy
            self.g_aug.add_edge(edge_id[0],'EVAL_POINT_NODE',length=dist_along[edge_id[0]])
            self.g_aug.add_edge(edge_id[1],'EVAL_POINT_NODE',length=dist_along[edge_id[1]])

            #As above, store edge data before removal
            removed_edge=(edge_id[0],edge_id[1],edge_id[2])
            removed_edge_atts=self.g_aug[edge_id[0]][edge_id[1]][edge_id[2]]

            self.g_aug.remove_edge(edge_id[0],edge_id[1],edge_id[2])

        #Return the details of the edge which has been removed, so it can be restored
        return removed_edge, removed_edge_atts



    def remove_eval_point_node(self,removed_edge,removed_edge_atts):
        '''
        Removes the temporary node when no longer needed - all coincident
        edges destroyed too.

        Also restores the original edge into which the temporary node was inserted
        '''

        self.g_aug.remove_node('EVAL_POINT_NODE')

        self.g_aug.add_edge(removed_edge[0],removed_edge[1],key=removed_edge[2],attr_dict=removed_edge_atts)



    def evaluate_non_point(self,eval_point):
        '''
        Evaluate the KDE at a non-kernel centre. The argument eval_point is given
        as (closest_edge,dist_along)
        '''

        #Add a temporary node for the evaluation point
        removed_edge, removed_edge_atts = self.add_eval_point_node(eval_point)

        #Get all centres within h, plus distances and paths
        point_paths = self.points_within_bandwidth('EVAL_POINT_NODE')

        #Initialise output
        total_value=0

        #Look at every kernel centre within h
        for p in point_paths:

            #Loop all unique paths to that kernel centre
            for path, path_distance in point_paths[p]:

                #Get the value of the univariate kernel
                kernel_value = self.kernel_univ.pdf(path_distance)

                #Go through every intermediate node, dividing the kernel value
                for path_node in path:

                    kernel_value /= float(nx.degree(self.g_aug,path_node)-1)

                #Add the contribution from this kernel centre, via this path, to
                #the total
                total_value += kernel_value

        #Remove the dummy node we created
        self.remove_eval_point_node(removed_edge,removed_edge_atts)

        return total_value


    def evaluate_point(self,point_id):
        '''
        Exactly the same as above, but where the evaluation is to be done at an existing
        node (kernel centre).

        As before except without the hassle of creating new nodes etc.
        '''

        point_paths = self.points_within_bandwidth(point_id)

        total_value=0

        for p in point_paths:

            for path, path_distance in point_paths[p]:

                kernel_value = self.kernel_univ.pdf(path_distance)

                for path_node in path:

                    kernel_value /= float(nx.degree(self.g_aug,path_node)-1)

                total_value += kernel_value

        return total_value


if __name__ == '__main__':

    from settings import DATA_DIR
    import os

    ITNFILE = os.path.join(DATA_DIR, 'network_data/itn_sample', 'mastermap-itn_417209_0_brixton_sample.gml')

    # A little demo

    #Just build the network as usual
    itndata = itn.read_gml(ITNFILE)
    CurrentNet = itn.ITNStreetNet()
    CurrentNet.load_from_data(itndata)

    xmin, ymin, xmax, ymax = CurrentNet.extent
    x_grid,y_grid,edge_locator=CurrentNet.bin_edges(xmin, xmax, ymin, ymax, 50)


    #Four test points - 1 and 3 on same segment, 2 on neighbouring segment, 4 long way away.
    closest_edge1,dist_along1,snap_dist=CurrentNet.closest_segments_euclidean(531190,175214,x_grid,y_grid,edge_locator)
    closest_edge2,dist_along2,snap_dist=CurrentNet.closest_segments_euclidean(531149,175185,x_grid,y_grid,edge_locator)
    closest_edge3,dist_along3,snap_dist=CurrentNet.closest_segments_euclidean(531210,175214,x_grid,y_grid,edge_locator)
    closest_edge4,dist_along4,snap_dist=CurrentNet.closest_segments_euclidean(531198,174962,x_grid,y_grid,edge_locator)


    #Add some combination of the points as the kernel centres
    points={}
    points['point1']=(closest_edge1,dist_along1)
    #points['point2']=(closest_edge2,dist_along2)
    points['point3']=(closest_edge3,dist_along3)
    #points['point4']=(closest_edge4,dist_along4)

    #Initialise the kernel
    TestKernel=EqualSplitKernel(CurrentNet,points,100)


    #Both evaluation methods
    print TestKernel.evaluate_non_point((closest_edge2,dist_along2))
    print TestKernel.evaluate_point('point3')
