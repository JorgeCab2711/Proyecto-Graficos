from gl import Renderer, color, V3, V2
from texture import Texture
from shaders import *

width = 960
height = 540

rend = Renderer(width, height)

rend.dirLight = V3(-1, 0, 0)

# rend.active_texture = Texture("models/model.bmp")
# rend.normal_map = Texture("models/model_normal.bmp")

# rend.active_shader = normalMap
rend.glLoadModel("models/model.obj",
                 translate=V3(-4, 0, -10),
                 scale=V3(1, 1, 1),
                 rotate=V3(0, 0, 0))

# rend.active_shader = gourad
# rend.glLoadModel("models/model.obj",
#                  translate=V3(4, 0, -10),
#                  scale=V3(4, 4, 4),
#                  rotate=V3(0, 0, 0))

rend.glFinish("output.bmp")
