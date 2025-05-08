import numpy as np
from cellpose import models
from cellpose.dynamics import get_centers
from scipy.ndimage import find_objects

def cell_center_model(diameter=25, flow_threshold=0.4, channels=[0,0]):
    model = models.Cellpose(gpu=True, model_type='nuclei')

    # based on https://github.com/MouseLand/cellpose/blob/main/cellpose/dynamics.py
    def eval_model(images, im_size):
        assert(images[0].shape == (1, 512, 512))
        images = [images[0, ::]]
        scale_x = im_size[0] / 512
        scale_y = im_size[1] / 512

        masks, flows, styles, diams = model.eval(images, diameter=diameter, channels=channels,
                                                    flow_threshold=flow_threshold, do_3D=False)



        slices = find_objects(masks[0])
        # turn slices into array
        slices = np.array([
            np.array([i, si[0].start, si[0].stop, si[1].start, si[1].stop])
            for i, si in enumerate(slices)
            if si is not None
        ])
        
        centers, _ = get_centers(masks[0], slices)
        centers_x = [round(scale_x * c[0]) for c in centers]
        centers_y = [round(scale_y * c[1]) for c in centers]

        return centers_x, centers_y
    return eval_model