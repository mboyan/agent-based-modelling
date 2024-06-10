from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa_discretechoice import MesaDiscretechoice


# A simple server that visualizes time evolution of the agents' location
def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Color": 'red',
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}

    return portrayal

grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
server = ModularServer(MesaDiscretechoice,
                       [grid],
                       "mesa + discrete choice",
                       {})

server.port = 8521
server.launch()
