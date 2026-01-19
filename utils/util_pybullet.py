import pybullet as p
from utils.util_pyqtgraph import visualGeometryType2mesh

def bullet2pyqtgraph(Id):
    shape_data = p.getVisualShapeData(int(Id))
    mesh_items = []
    for i in range(len(shape_data)):
        # process item i
        mesh = visualGeometryType2mesh(shape_data[i])
        mesh_items.append(mesh)
    return mesh_items

