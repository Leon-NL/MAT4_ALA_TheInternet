#Authers: Leon Hulsebos, Elena Serrano, Justin Biju Thomas
#Date: 10-6-2024
#Class: M4B
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Options
CHOICE = "custom"       # "test", "normal", "big", "custom". Custom creates matrix with MATRIXSIZE
USE_DAMPING = False     # True or False

# Additional options     
DAMPING_FACTOR = 0.85   # Damping factor to account for random clics
MATRIX_SIZE = 2400      # MATRIX_SIZE will always be sized back to 60 if it is higher than 60

# Specific options
# Not necessary to change 
TIMEOUT = 100           # After how many tries the ranking gets a timeout
GRAPH_SPACING = 1       # Spacing between the nodes
NODE_SIZE = 12000       # Size of the nodes




# Standard connection matrixes 
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

# Functions
def CreateCustomInternet(size):
    if size > 60:
        size = 60

    # Set the dimensions of the matrix
    rows, cols = size, size
        
    # Create a matrix with random 1's and 0's
    matrix = np.random.randint(2, size=(rows, cols))

    # Set the diagonal elements to zero
    np.fill_diagonal(matrix, 0)

    # Calculate the sum of the 1's in each column
    matrix_sum = matrix.sum(axis=0)

    # Avoid division by zero by replacing zero sums with ones temporarily
    matrix_sum_safe = np.where(matrix_sum == 0, 1, matrix_sum)

    # Divide each element by the sum of the 1's in its column
    normalized_matrix = matrix / matrix_sum_safe
    return normalized_matrix

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
            NewEigenVector = DAMPING_FACTOR * (ConnectionMap @ EigenVector) + ((1-DAMPING_FACTOR)/columnCount)
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
    rankCopy = np.copy(rank)
    columnCount = rankCopy.shape[0]
    for i in range(0, columnCount):
        place = np.where(rankCopy == max(rankCopy))
        print(str(i+1),":",chr(65+int(place[0][0])))
        rankCopy[place[0][0]] = 0

def diagram(connectionMap, ranking, scale, spacing):

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on TEST_INTERNET matrix
    for i in range(connectionMap.shape[0]):
        for j in range(connectionMap.shape[1]):
            if connectionMap[i, j] > 0:
                G.add_edge(chr(j + 65), chr(i + 65))  # Map node index to letter

    # Create node sizes based on the PageRank scores
    node_sizes = [rank * scale for rank in ranking]

    node_colors = plt.cm.rainbow(np.linspace(0, 1, len(G.nodes)))

    # Draw the graph with adjusted layout to minimize edge overlaps and spread out the nodes
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=spacing, seed=None)  # Increase the value of k for more spread out nodes
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color="gray", font_size=10, font_weight="bold", arrows=True)

    # Display the plot
    plt.show()

def InternetChoice(choice):
    if choice == "test":
        return TEST_INTERNET
    elif choice == "normal":
        return INTERNET
    elif choice == "big":
        return BIG_INTERNET
    elif choice == "custom":
        return CreateCustomInternet(MATRIX_SIZE)

# Main loop
if __name__ == "__main__":
    internet = InternetChoice(CHOICE)
    Ranking = RankThe(internet, USE_DAMPING)
    print(Ranking)
    PrintInHumanTerms(Ranking)
    diagram(internet, Ranking, NODE_SIZE, GRAPH_SPACING)

    #addings