from gl import Raytracer, V3
from texture import *
from figures import *
from lights import *
import time


start = time.time()

width = 1500
height = 512


# Materiales
brick = Material(diffuse=(0.8, 0.3, 0.3), spec=16)
stone = Material(diffuse=(0.4, 0.4, 0.4), spec=8)
grass = Material(diffuse=(0.3, 1.0, 0.3), spec=64)


#marble = Material(spec=64, texture=Texture("marble.bmp"), matType=REFLECTIVE)
obsidian = Material(spec=60, texture=Texture("obsi.bmp"), matType=OPAQUE)
portal = Material(spec=60, texture=Texture(
    "portal.bmp"), matType=OPAQUE)

mirror = Material(diffuse=(0.9, 0.9, 0.9), spec=64, matType=REFLECTIVE)
glass = Material(diffuse=(0.9, 0.9, 0.9), spec=64,
                 ior=1.5, matType=TRANSPARENT)

rtx = Raytracer(width, height)

rtx.envMap = Texture("mine.bmp")

rtx.lights.append(AmbientLight(intensity=0.1))
rtx.lights.append(DirectionalLight(direction=(0, 0, -1), intensity=0.5))
rtx.lights.append(PointLight(point=(-1, -1, 0)))


# Figures
# obsidians
# left obsidian
# 1
rtx.scene.append(AABB(position=(-3, 3, -15),
                 size=(2, 2, 2), material=obsidian))
# 2
rtx.scene.append(AABB(position=(-3, 1, -15),
                 size=(2, 2, 2), material=obsidian))
# 3
rtx.scene.append(AABB(position=(-3, -1, -15),
                 size=(2, 2, 2), material=obsidian))
# 4
rtx.scene.append(AABB(position=(-3, -3, -15),
                 size=(2, 2, 2), material=obsidian))

# # right obsidian
# 5
rtx.scene.append(AABB(position=(3, 3, -15),
                 size=(2, 2, 2), material=obsidian))
#  6
rtx.scene.append(AABB(position=(3, 1, -15),
                 size=(2, 2, 2), material=obsidian))
#  7
rtx.scene.append(AABB(position=(3, -1, -15),
                 size=(2, 2, 2), material=obsidian))
# 8
rtx.scene.append(AABB(position=(3, -3, -15),
                 size=(2, 2, 2), material=obsidian))


# bottom
# 15
rtx.scene.append(AABB(position=(-1, -5, -15),
                 size=(2, 2, 2), material=obsidian))
# 11
rtx.scene.append(AABB(position=(1, -5, -15),
                 size=(2, 2, 2), material=obsidian))
# 12
rtx.scene.append(AABB(position=(3, -5, -15),
                 size=(2, 2, 2), material=obsidian))
# 13
rtx.scene.append(AABB(position=(-3, -5, -15),
                 size=(2, 2, 2), material=obsidian))


#  Top
# 13
rtx.scene.append(AABB(position=(-1, 3, -15),
                 size=(2, 2, 2), material=obsidian))
# 14
rtx.scene.append(AABB(position=(1, 3, -15),
                 size=(2, 2, 2), material=obsidian))

# # obsidian matter
rtx.scene.append(AABB(position=(-1, -3, -15),
                 size=(2, 2, 2), material=portal))

rtx.scene.append(AABB(position=(1, -3, -15),
                 size=(2, 2, 2), material=portal))

rtx.scene.append(AABB(position=(-1, -1, -15),
                 size=(2, 2, 2), material=portal))

rtx.scene.append(AABB(position=(1, -1, -15),
                 size=(2, 2, 2), material=portal))

rtx.scene.append(AABB(position=(-1, 1, -15),
                 size=(2, 2, 2), material=portal))

rtx.scene.append(AABB(position=(1, 1, -15),
                 size=(2, 2, 2), material=portal))

rtx.glRender()

rtx.glFinish("z-output.bmp")
end = time.time()

print("Compilation time: " + str((end - start)/60)+" minutes")
