# -*- coding: utf-8 -*-
# filename          : TeamFormation.py
# description  : python implementation of seveal papers in Team Formation Problem
# author            : liu-yihong
# date              : 20211223
# license           : GNU GPLV3
# py version        : 3.9.7
# ==============================================================================

"""

Check the github repo for more information:
https://github.com/liu-yihong/Python-Team-Formation
"""
from typing import List, Tuple
import numpy as np
import networkx as nx
from itertools import combinations

NUM_SKILL = 4
SKILL_PROB = 0.5
MAX_HOP = 5
MAX_CAP = NUM_SKILL
LEADERROLE_SKILL_ID = 0
BIG_M = 5

'''
Utility Class
'''


class Task(object):
    def __init__(self) -> None:
        super().__init__()
        SkillList: np.array

    def new(self, SkillList: Tuple[List, np.array]):
        NewTaskInstance = Task()
        NewTaskInstance.SkillList = np.asarray(SkillList)

        return NewTaskInstance


class Node(object):

    def __init__(self) -> None:
        super().__init__()
        ID: int
        SkillList: np.array
        AssignmentList: np.array

    def random(self, ID: int):
        NodeInstance = Node()
        NodeInstance.ID = ID
        NodeInstance.SkillList = np.random.binomial(
            1, SKILL_PROB, NUM_SKILL)
        return NodeInstance

    def new(self, ID: int, SkillList: Tuple[List, np.array], AssignmentList: Tuple[List, np.array]):
        NodeInstance = Node()

        if SkillList is None:
            SkillList = np.zeros((1, NUM_SKILL))
        if AssignmentList is None:
            AssignmentList = np.zeros((1, NUM_SKILL))

        NodeInstance.ID, NodeInstance.SkillList, NodeInstance.AssignmentList = ID, np.asarray(
            SkillList), np.asarray(AssignmentList)
        assert (NodeInstance.AssignmentList <= NodeInstance.SkillList).all(
        ), "At least one skill does not cover assignment!"
        return NodeInstance

    def updateAssignment(self, AssignmentList: Tuple[List, np.array]):
        assert (AssignmentList <= self.SkillList).all(
        ), "At least one skill does not cover assignment!"
        self.AssignmentList = np.asarray(AssignmentList)


class CapNode(Node):

    def __init__(self) -> None:
        super().__init__()
        Capacity: int

    def random(self, ID: int):
        NodeInstance = CapNode()
        NodeInstance.ID = ID
        NodeInstance.SkillList = np.random.binomial(
            1, SKILL_PROB, NUM_SKILL)
        NodeInstance.Capacity = np.random.randint(1, MAX_CAP+1)
        return NodeInstance

    def new(self, ID: int, SkillList: Tuple[List, np.array], AssignmentList: Tuple[List, np.array], Capacity: int):
        NodeInstance = CapNode()

        if SkillList is None:
            SkillList = np.zeros((1, NUM_SKILL))
        if AssignmentList is None:
            AssignmentList = np.zeros((1, NUM_SKILL))

        NodeInstance.ID, NodeInstance.Capacity, NodeInstance.SkillList, NodeInstance.AssignmentList = ID, Capacity, np.asarray(
            SkillList), np.asarray(AssignmentList)

        # check individual skill covering constraint
        assert (NodeInstance.AssignmentList <= NodeInstance.SkillList).all(
        ), "At least one skill does not cover assignment!"

        # check individual capacity constraint
        assert np.sum(
            NodeInstance.AssignmentList) <= NodeInstance.Capacity, "Assigned items exceed capacity!"

        return NodeInstance


class Team(object):

    def __init__(self) -> None:
        super().__init__()
        InnerTask: Task
        NodeList: List[Node]

    def checkCoveringConstraints(self, SpecificTask: Task, SpecificNodeList: List[Node]) -> bool:
        AssignmentCover = np.sum(
            [i.AssignmentList for i in SpecificNodeList], axis=0)
        return (AssignmentCover >= SpecificTask.SkillList).all()


class CapTeam(Team):

    def __init__(self) -> None:
        super().__init__()

    def new(self, SpecificTask: Task, SpecificNodeList: List[Node]):
        NewTeamInstance = Team()
        # check team skill covering constraint
        assert self.checkCoveringConstraints(
            SpecificTask, SpecificNodeList), "Nodes cannot cover task requirement!"
        NewTeamInstance.InnerTask, NewTeamInstance.NodeList = SpecificTask, SpecificNodeList

        return NewTeamInstance


'''
Utility Function
'''


def generateBinomialGraph(n, p, node_type: CapNode):
    # generate an undirected graph
    G = nx.fast_gnp_random_graph(n, p, 0, False)
    # generate weights
    np.random.seed(0)
    EdgesWeight = np.random.uniform(low=0.0, high=BIG_M, size=len(G.edges))
    # assign weights
    for idx, tup in enumerate(G.edges):
        G[tup[0]][tup[1]]['weight'] = EdgesWeight[idx]
    # generate Node object instances
    NodesList = [node_type().random(ID=i) for i in G.nodes]

    return G, NodesList


def MaxItem(AllNodeList: List[Node], TeamNodeIDList: List[int], FocalTask: Task, FocalNodeID: int):
    assert NUM_SKILL == len(
        FocalTask.SkillList), "Number of skills in focal task does not match %s" % (NUM_SKILL)

    G = nx.DiGraph()
    EdgesList = []

    # add left and right nodes
    G.add_nodes_from(['I' + str(i)
                      for i in range(NUM_SKILL)], bipartite='TaskRequirement')
    G.add_nodes_from(['N' + str(i) for i in TeamNodeIDList], bipartite='Node')

    # TODO: consider how to ensure the focal node is in the final team
    # the lines below ensure the focal node is certainly included in the final team
    # the focal user must be the leader, therefore must have the leader skill
    # AllNodeList[FocalNodeID].SkillList[LEADERROLE_SKILL_ID] = 1
    # # we can add only one edge between the focal node and the leader role, so the focal node is certainly included in the final team
    # EdgesList.append(
    #     ('I' + str(LEADERROLE_SKILL_ID), 'N' + str(FocalNodeID), 1)
    # )
    # # add intermediate edges
    # for NodeID in TeamNodeIDList:
    #     assert NUM_SKILL == len(
    #         AllNodeList[NodeID].SkillList), "Number of ID %s skills does not match %s" % (NodeID, NUM_SKILL)
    #     for idx, j in enumerate(AllNodeList[NodeID].SkillList):
    #         if idx == LEADERROLE_SKILL_ID:
    #             continue
    #         EdgesList.append(
    #             ('I' + str(idx), 'N' + str(NodeID), j)
    #         )

    # add intermediate edges
    for NodeID in TeamNodeIDList:
        assert NUM_SKILL == len(
            AllNodeList[NodeID].SkillList), "Number of ID %s skills does not match %s" % (NodeID, NUM_SKILL)
        for idx, j in enumerate(AllNodeList[NodeID].SkillList):
            EdgesList.append(
                ('I' + str(idx), 'N' + str(NodeID), j)
            )

    # add source node
    G.add_node('s', bipartite='Source')
    for idx, j in enumerate(FocalTask.SkillList):
        EdgesList.append(
            ('s', 'I' + str(idx), j)
        )

    # add sink node
    G.add_node('t', bipartite='Sink')
    for NodeID in TeamNodeIDList:
        EdgesList.append(
            ('N' + str(NodeID), 't', AllNodeList[NodeID].Capacity)
        )

    # add all edges
    G.add_weighted_edges_from(EdgesList, weight='capacity')

    # check max flow
    FlowValue, FlowDict = nx.maximum_flow(G, 's', 't')

    return FlowValue, FlowDict


def MaxItemNFeasibleTeam(AllNodeList: List[Node], TeamNodeIDList: List[int], FocalTask: Task, FocalNodeID: int):
    # check max flow
    FlowValue, FlowDict = MaxItem(
        AllNodeList, TeamNodeIDList, FocalTask, FocalNodeID)

    # check feasibility of capacity and assignment
    assert FlowValue >= FocalTask.SkillList.sum(), "Infeasible team!"

    # now we update skill assignment over individuals
    NodeIDWithWorkList = []
    for NodeID in TeamNodeIDList:
        SpecificAssignment = np.zeros(NUM_SKILL)
        for idx, j in enumerate(AllNodeList[NodeID].SkillList):
            if 'N' + str(NodeID) not in FlowDict['I' + str(idx)].keys():
                continue
            SpecificAssignment[idx] = FlowDict['I' +
                                               str(idx)]['N' + str(NodeID)]
        if SpecificAssignment.sum() == 0:
            continue
        NodeIDWithWorkList.append(NodeID)
        AllNodeList[NodeID].updateAssignment(SpecificAssignment)

    # TODO: consider how to ensure the focal node is in the final team
    # check the focal node is in the final team
    # assert FocalNodeID in NodeIDWithWorkList, "Focal node not in the final team!"

    # construct the new team
    FeasibleTeam = CapTeam().new(
        SpecificTask=FocalTask,
        SpecificNodeList=[AllNodeList[i] for i in NodeIDWithWorkList]
    )

    return FeasibleTeam


def AugmentGraph(G: nx.Graph, FocalNodeID: int):
    AllOtherNodes = set(G.nodes)
    AllOtherNodes.remove(FocalNodeID)

    AllPossiblePathFromFocalNode = nx.shortest_path_length(
        G=G, source=FocalNodeID, weight='weight'
    )
    AllPossiblePathFromFocalNode.pop(FocalNodeID)

    AugmentG = nx.Graph()
    AugmentList = []
    for t in AllPossiblePathFromFocalNode.keys():
        AugmentList.append(
            (FocalNodeID, t, AllPossiblePathFromFocalNode[t])
        )
        AllOtherNodes.remove(t)
    for t in AllOtherNodes:
        AugmentList.append(
            (FocalNodeID, t, BIG_M * 100)
        )
    AugmentG.add_weighted_edges_from(AugmentList, weight='weight')

    return AugmentG


# TODO: add collab cost calculation: diameter, steiner cost, bottleneck cost

# TODO Add MinMaxSol algorithms

def MinDiamSol(G: nx.Graph, AllNodeList: List[Node], FocalNodeID: int, FocalTask: Task, MAXIMUM_HOP: int):
    FeasibleTeam = None
    for h in range(MAXIMUM_HOP):
        # edge weights included in EgoG
        EgoG = nx.ego_graph(G, FocalNodeID, radius=h + 1)
        AugmentG = AugmentGraph(
            EgoG,
            FocalNodeID
        )
        # get all possible radius
        AllRadiusList = sorted(
            [(tup[2]['weight'], tup[1])
             for _, tup in enumerate(AugmentG.edges(data=True))]
        )
        # from smallest radius to the largest
        for idx, tup in enumerate(AllRadiusList):
            PotentialNodeIDList = [FocalNodeID] + [t[-1]
                                                   for t in AllRadiusList[:idx+1]]
            try:
                FeasibleTeam = MaxItemNFeasibleTeam(
                    AllNodeList=AllNodeList,
                    TeamNodeIDList=PotentialNodeIDList,
                    FocalTask=FocalTask,
                    FocalNodeID=FocalNodeID
                )
                break
            except AssertionError:
                continue
        if FeasibleTeam is not None:
            break
    assert FeasibleTeam is not None, "Cannot find feasible team!"
    return FeasibleTeam


def MinAggrSol(G: nx.Graph, AllNodeList: List[Node], FocalNodeID: int, FocalTask: Task, MAXIMUM_HOP: int):
    FeasibleTeam = None
    for h in range(MAXIMUM_HOP):
        # get h-hop neighbors
        # edge weights included in EgoG
        EgoG = nx.ego_graph(G, FocalNodeID, radius=h + 1)
        # get augment graph
        # TODO: consider weighted edges
        AugmentG = AugmentGraph(
            EgoG,
            FocalNodeID
        )
        # initialize variables
        CoverList = [FocalNodeID]
        CurrentFlowValue, _ = MaxItem(
            AllNodeList=AllNodeList,
            TeamNodeIDList=CoverList,
            FocalTask=FocalTask,
            FocalNodeID=FocalNodeID
        )
        while CurrentFlowValue < FocalTask.SkillList.sum():
            NextNodeCandidates = set(AugmentG.nodes)
            NextNodeCandidates.difference_update(CoverList)
            FinalOBJ = 0.0
            x_bar = None
            for x in NextNodeCandidates:
                NextFlowValue, _ = MaxItem(
                    AllNodeList=AllNodeList,
                    TeamNodeIDList=CoverList + [x],
                    FocalTask=FocalTask,
                    FocalNodeID=FocalNodeID
                )
                CurrentOBJ = (NextFlowValue - CurrentFlowValue) / \
                    AugmentG[FocalNodeID][x]['weight']
                if CurrentOBJ > FinalOBJ:
                    FinalOBJ, x_bar = CurrentOBJ, x
                else:
                    continue
            if x_bar is not None:
                CoverList.append(x_bar)
                CurrentFlowValue, _ = MaxItem(
                    AllNodeList=AllNodeList,
                    TeamNodeIDList=CoverList,
                    FocalTask=FocalTask,
                    FocalNodeID=FocalNodeID
                )
            else:
                # no node can increase the flow value of current cover
                break
        if CurrentFlowValue >= FocalTask.SkillList.sum():
            # TODO: here we do not consider minimum cost steiner tree
            # SteinerTree = nx.approximation.steiner_tree(G, CoverList)
            # CoverList = list(SteinerTree.nodes)
            FeasibleTeam = MaxItemNFeasibleTeam(
                AllNodeList=AllNodeList,
                TeamNodeIDList=CoverList,
                FocalTask=FocalTask,
                FocalNodeID=FocalNodeID
            )
            break
        else:
            continue

    assert FeasibleTeam is not None, "Cannot find feasible team!"
    return FeasibleTeam


def MinMaxSol(G: nx.Graph, AllNodeList: List[Node], FocalNodeID: int, FocalTask: Task, MAXIMUM_HOP: int):
    pass
