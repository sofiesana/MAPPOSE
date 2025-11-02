import gymnasium
from rware.warehouse import ImageLayer, Warehouse
import numpy as np

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1

def get_global_image(
    env,
    image_layers=[
        ImageLayer.SHELVES,
        ImageLayer.GOALS,
    ],
    recompute=False,
    pad_to_shape=None,
):
    """
    Get global image observation
    :param image_layers: image layers to include in global image
    :param recompute: bool whether image should be recomputed or taken from last computation
        (for default params, image will be constant for environment so no recomputation needed
            but if agent or request information is included, then should be recomputed)
        :param pad_to_shape: if given than pad environment global image shape into this
            shape (if doesn't fit throw exception)
    """
    if recompute or env.global_image is None:
        layers = []
        for layer_type in image_layers:
            if layer_type == ImageLayer.SHELVES:
                layer = env.grid[_LAYER_SHELFS].copy().astype(np.float32)
                # set all occupied shelf cells to 1.0 (instead of shelf ID)
                layer[layer > 0.0] = 1.0
            elif layer_type == ImageLayer.REQUESTS:
                layer = np.zeros(env.grid_size, dtype=np.float32)
                for requested_shelf in env.request_queue:
                    layer[requested_shelf.y, requested_shelf.x] = 1.0
            elif layer_type == ImageLayer.AGENTS:
                layer = env.grid[_LAYER_AGENTS].copy().astype(np.float32)
                # set all occupied agent cells to 1.0 (instead of agent ID)
                layer[layer > 0.0] = 1.0
            elif layer_type == ImageLayer.AGENT_DIRECTION:
                layer = np.zeros(env.grid_size, dtype=np.float32)
                for ag in env.agents:
                    agent_direction = ag.dir.value + 1
                    layer[ag.y, ag.x] = float(agent_direction)
            elif layer_type == ImageLayer.AGENT_LOAD:
                layer = np.zeros(env.grid_size, dtype=np.float32)
                for ag in env.agents:
                    if ag.carrying_shelf is not None:
                        layer[ag.y, ag.x] = 1.0
            elif layer_type == ImageLayer.GOALS:
                layer = np.zeros(env.grid_size, dtype=np.float32)
                for goal_y, goal_x in env.goals:
                    layer[goal_x, goal_y] = 1.0
            elif layer_type == ImageLayer.ACCESSIBLE:
                layer = np.ones(env.grid_size, dtype=np.float32)
                for ag in env.agents:
                    layer[ag.y, ag.x] = 0.0
            else:
                raise ValueError(f"Unknown image layer type: {layer_type}")
            layers.append(layer)
        env.global_image = np.stack(layers)
        if pad_to_shape is not None:
            padding_dims = [
                pad_dim - global_dim
                for pad_dim, global_dim in zip(
                    pad_to_shape, env.global_image.shape
                )
            ]
            assert all([dim >= 0 for dim in padding_dims])
            pad_before = [pad_dim // 2 for pad_dim in padding_dims]
            pad_after = [
                pad_dim // 2 if pad_dim % 2 == 0 else pad_dim // 2 + 1
                for pad_dim in padding_dims
            ]
            env.global_image = np.pad(
                env.global_image,
                pad_width=tuple(zip(pad_before, pad_after)),
                mode="constant",
                constant_values=0,
            )
    return env.global_image

def get_true_coords(matrix):
    coords = np.argwhere(matrix == 1)
    return coords

def get_full_state(env, flatten=True):

    attributes_dict = {0: ImageLayer.SHELVES, 1: ImageLayer.GOALS, 2: ImageLayer.AGENTS,
                       3: ImageLayer.REQUESTS, 4: ImageLayer.AGENT_DIRECTION,
                       5: ImageLayer.AGENT_LOAD, 6: ImageLayer.ACCESSIBLE}
    s = get_global_image(env.unwrapped, image_layers=[
        ImageLayer.SHELVES,
        ImageLayer.REQUESTS,
        ImageLayer.GOALS,
        ImageLayer.AGENT_DIRECTION
        ], 
        recompute=True)

    all_shelf_info = s[0] + s[1]
    goals = get_true_coords(s[2])
    all_agent_info = s[3]
    # final_as_matrices is not used, can be removed for efficiency

    if flatten:
        # Efficiently concatenate flattened arrays using numpy
        flat = np.concatenate([
            all_shelf_info.ravel(),
            all_agent_info.ravel(),
            goals.ravel()
        ])
        # print(flat)
        return flat
    else:
        # Efficiently add shelf info and goals, then stack as numpy array
        all_shelf_info = all_shelf_info + 3 * s[2]  # shelves + goals
        final_global_state = np.stack([all_shelf_info, all_agent_info])
        # print("Final global state shape:", final_global_state.shape)
        # print("Final global state array:\n", final_global_state)
        return final_global_state