from gl import Raytracer, V3
from texture import *
from figures import *
from lights import *


width = 1000
height = 512


# Materiales
brick = Material(diffuse=(0.8, 0.3, 0.3), spec=16)
stone = Material(diffuse=(0.4, 0.4, 0.4), spec=8)
grass = Material(diffuse=(0.3, 1.0, 0.3), spec=64)


marble = Material(spec=64, texture=Texture("marble.bmp"), matType=REFLECTIVE)

mirror = Material(diffuse=(0.9, 0.9, 0.9), spec=64, matType=REFLECTIVE)
glass = Material(diffuse=(0.9, 0.9, 0.9), spec=64,
                 ior=1.5, matType=TRANSPARENT)

rtx = Raytracer(width, height)

rtx.envMap = Texture("mine.bmp")

rtx.lights.append(AmbientLight(intensity=0.1))
rtx.lights.append(DirectionalLight(direction=(0, 0, -1), intensity=0.5))
rtx.lights.append(PointLight(point=(-1, -1, 0)))


rtx.scene.append(AABB(position=(-2, 1, -10), size=(2, 2, 2), material=None))


rtx.glRender()

rtx.glFinish("z-output.bmp")
