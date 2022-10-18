from numpy import arctan2, arccos, pi
import numpy as np
import struct
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8


class Texture(object):
    def __init__(self, filename):

        with open(filename, "rb") as image:
            image.seek(10)
            headerSize = struct.unpack('=l', image.read(4))[0]

            image.seek(18)
            self.width = struct.unpack('=l', image.read(4))[0]
            self.height = struct.unpack('=l', image.read(4))[0]

            image.seek(headerSize)

            self.pixels = []

            for y in range(self.height):
                pixelRow = []

                for x in range(self.width):
                    b = ord(image.read(1)) / 255
                    g = ord(image.read(1)) / 255
                    r = ord(image.read(1)) / 255
                    pixelRow.append([r, g, b])

                self.pixels.append(pixelRow)

    def getColor(self, u, v):
        if 0 <= u < 1 and 0 <= v < 1:
            return self.pixels[int(v * self.height)][int(u * self.width)]
        else:
            return None

    def getEnvColor(self, dir):
        dir = dir / np.linalg.norm(dir)

        x = int((arctan2(dir[2], dir[0]) / (2 * pi) + 0.5) * self.width)
        y = int(arccos(-dir[1]) / pi * self.height)

        return self.pixels[y][x]
