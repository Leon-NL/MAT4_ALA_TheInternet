#Auther: Leon Hulsebos
#Date: 10-6-2024
#Class: M4B
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

DAMPINGFACTOR = 0.85
TIMEOUT = 100
TEST_INTERNET = np.array([[0,      (1/2),  0,   0    ],
                          [(1/3),  0,      0,   (1/2)],
                          [(1/3),  0,      0,   (1/2)],
                          [(1/3),  (1/2),  1,   0    ]])

INTERNET = np.array([[0,    0,      0,      0,      (1/4),  0,      0,      0],
                    [0,     0,      0,      (1/4),  0,      (1/3),  (1/4),  0],
                    [0,     (1/3),  0,      (1/4),  (1/4),  (1/3),  0,      (1/3)],
                    [(1/2), (1/3),  (1/2),  0,      0,      0,      (1/4),  0],
                    [(1/2), 0,      (1/2),  0,      0,      0,      (1/4),  (1/3)],
                    [0,     (1/3),  0,      0,      (1/4),  0,      0,      (1/3)],
                    [0,     0,      0,      (1/4),  0,      0,      0,      0],
                    [0,     0,      0,      (1/4),  (1/4),  (1/3),  (1/4),  0]])

BIG_INTERNET = np.array([[0,     (1/9),  (1/7),  0,      0,      (1/6),  (1/4),  (1/8),  0,      0,      0,      (1/6),  (1/8),  0,      0,      0,      0,      (1/7)],
                        [(1/9),  0,      (1/7),  0,      0,      0,      (1/4),  (1/8),  (1/8),  0,      0,      (1/6),  0,      0,      (1/7),  0,      0,      0    ],
                        [(1/9),  0,      0,      0,      0,      0,      0,      0,      0,      (1/6),  0,      0,      (1/8),  0,      (1/7),  0,      0,      0    ],
                        [(1/9),  (1/9),  0,      0,      0,      (1/6),  0,      0,      0,      (1/6),  0,      0,      0,      (1/5),  0,      (1/7),  0,      (1/7)],
                        [0,      0,      0,      (1/7),  0,      (1/6),  0,      (1/8),  0,      (1/6),  0,      0,      0,      0,      0,      (1/7),  (1/6),  (1/7)],
                        [0,      (1/9),  (1/7),  (1/7),  0,      0,      0,      0,      0,      0,      (1/7),  0,      0,      0,      (1/7),  0,      (1/6),  (1/7)],
                        [0,      (1/9),  (1/7),  0,      0,      (1/6),  0,      (1/8),  (1/8),  0,      0,      (1/6),  (1/8),  (1/5),  0,      (1/7),  0,      (1/7)],
                        [0,      0,      0,      (1/7),  0,      0,      0,      0,      (1/8),  0,      (1/7),  0,      0,      (1/5),  0,      (1/7),  0,      0    ],
                        [(1/9),  (1/9),  0,      0,      0,      (1/6),  0,      0,      0,      (1/6),  (1/7),  0,      (1/8),  0,      0,      (1/7),  0,      0    ],
                        [0,      (1/9),  0,      0,      0,      0,      0,      (1/8),  (1/8),  0,      (1/7),  0,      0,      0,      (1/7),  0,      0,      0    ],
                        [(1/9),  0,      0,      (1/7),  (1/6),  0,      0,      0,      0,      0,      0,      0,      (1/8),  0,      0,      (1/7),  0,      (1/7)],
                        [(1/9),  (1/9),  0,      0,      (1/6),  0,      0,      (1/8),  (1/8),  0,      (1/7),  0,      (1/8),  0,      0,      0,      (1/6),  (1/7)],
                        [0,      0,      0,      (1/7),  (1/6),  0,      (1/4),  0,      (1/8),  0,      0,      0,      0,      0,      (1/7),  0,      0,      0    ],
                        [0,      0,      0,      (1/7),  (1/6),  0,      0,      (1/8),  0,      0,      0,      0,      0,      0,      0,      (1/7),  (1/6),  0    ],
                        [(1/9),  0,      0,      (1/7),  (1/6),  0,      0,      0,      0,      0,      (1/7),  (1/6),  0,      (1/5),  0,      0,      (1/6),  0    ],
                        [(1/9),  0,      (1/7),   0,     (1/6),  0,      0,      0,      0,      (1/6),  0,      0,      (1/8),  0,      (1/7),  0,      0,      0    ],
                        [(1/9),  (1/9),  (1/7),   0,     0,      0,      (1/4),  (1/8),  (1/8),  (1/6),  (1/7),  (1/6),  0,      (1/5),  (1/7),  0,      0,      0    ],
                        [0,      (1/9),  (1/7),   0,     0,      (1/6),  0,      0,      (1/8),   0,     0,      (1/6),  (1/8),   0,     0,      0,      (1/6),  0    ]])

def RankThe(ConnectionMap, damping):
    """Ranks a connection matrix to a see wich site is more important acording of the links it has from other pages.

    Parameters
    ----------
    ConnectionMap = A 2D square matrix that shows which site links to what and the values are always between 0 and 1
    damping = True or False. This sets if the ranking uses daming to determin the ranking
    Returns
    -------
    A 1xN array with a value between 0 and 1 this indicates the ranking or returns "Not Found" if it is not close enough in the given TIMEOUT
    """

    columnCount = ConnectionMap.shape[1]
    EigenVector = np.ones(columnCount) / columnCount
    if damping == False:
        for i in range(0, TIMEOUT):
            NewEigenVector = ConnectionMap @ EigenVector
            #NewEigenVector = np.dot(ConnectionMap, EigenVector)
            if np.allclose(EigenVector, NewEigenVector, rtol=1e-04, atol=1e-08):
                return NewEigenVector
            EigenVector = NewEigenVector
        return "Not found"
    if damping == True:
        for i in range(0, TIMEOUT):
            NewEigenVector = DAMPINGFACTOR * (ConnectionMap @ EigenVector) + ((1-DAMPINGFACTOR)/columnCount)
            if np.allclose(EigenVector, NewEigenVector, rtol=1e-05, atol=1e-08):
                return NewEigenVector
            EigenVector = NewEigenVector
        return "Not found"

def PrintInHumanTerms(rank):
    """Prints the final listing in a easy to see way.
    
    Parameters
    ----------
    rank = 1xN array with the calculated ranking 
    Returns
    -------
    Nothing
    """
    columnCount = rank.shape[0]
    for i in range(0, columnCount):
        place = np.where(rank == max(rank))
        print(str(i+1),":",chr(65+int(place[0][0])))
        rank[place[0][0]] = 0

def diagram(connectionMap, ranking):

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on TEST_INTERNET matrix
    for i in range(connectionMap.shape[0]):
        for j in range(connectionMap.shape[1]):
            if connectionMap[i, j] > 0:
                G.add_edge(chr(j + 65), chr(i + 65))  # Map node index to letter


    # Determine the scaling factor for node sizes
    scaling_factor = 10000

    # Create node sizes based on the PageRank scores
    node_sizes = [rank * scaling_factor for rank in ranking]

    node_colors = plt.cm.rainbow(np.linspace(0, 1, len(G.nodes)))

    # Draw the graph with adjusted layout to minimize edge overlaps and spread out the nodes
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=1.5, seed=42)  # Increase the value of k for more spread out nodes
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color="gray", font_size=10, font_weight="bold", arrows=True)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    Ranking = RankThe(BIG_INTERNET, False)
    print(Ranking)
    diagram(BIG_INTERNET, Ranking)
    PrintInHumanTerms(Ranking)

    #addings