import numpy as np
from vispy import app,  visuals, scene, io


def load_matrix_from_txt(filename, shape= (50, 50, 5)):
    activation = np.empty((50,50,5))
    with open(filename,'r') as fh:
        for line in fh:
            x,y,z,val = line.split(",")
            activation[int(x),int(y),int(z)] = val
    print(f'Finish reading from file {filename} in shape {activation.shape}')
    return activation

featuremap = load_matrix_from_txt('test.txt')

canvas = scene.SceneCanvas(keys='interactive', show=True)

view = canvas.central_widget.add_view()
view.camera = 'fly'
view.camera.fov = 50
view.camera.distance = 5

volume = scene.visuals.Volume(vol=featuremap, cmap='viridis', parent=view.scene)
axis = scene.visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    app.run()
