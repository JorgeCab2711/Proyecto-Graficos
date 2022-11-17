import struct
from collections import namedtuple
import numpy as np
from figures import *
from lights import *
from math import cos, sin, tan, pi
from obj import Obj
from numba import jit, cuda

STEPS = 1
MAX_RECURSION_DEPTH = 4

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])


def char(c):
    # 1 byte
    return struct.pack('=c', c.encode('ascii'))


def word(w):
    # 2 bytes
    return struct.pack('=h', w)


def dword(d):
    # 4 bytes
    return struct.pack('=l', d)


def color(r, g, b):
    return bytes([int(b * 255),
                  int(g * 255),
                  int(r * 255)])


def baryCoords(A, B, C, P):

    areaPBC = (B.y - C.y) * (P.x - C.x) + (C.x - B.x) * (P.y - C.y)
    areaPAC = (C.y - A.y) * (P.x - C.x) + (A.x - C.x) * (P.y - C.y)
    areaABC = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)

    try:
        # PBC / ABC
        u = areaPBC / areaABC
        # PAC / ABC
        v = areaPAC / areaABC
        # 1 - u - v
        w = 1 - u - v
    except:
        return -1, -1, -1
    else:
        return u, v, w


def matrix(rows, columns, anyList):
    matrix = []
    for m in range(rows):
        ListRow = []
        for k in range(columns):
            ListRow.append(anyList[rows * m + k])
        matrix.append(ListRow)
    return matrix


class Raytracer(object):
    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.fov = 60
        self.nearPlane = 0.1
        self.camPosition = V3(0, 0, 0)

        self.scene = []
        self.lights = []

        self.envMap = None
        self.active_shader = None

        self.clearColor = color(0, 0, 0)
        self.currColor = color(1, 1, 1)

        self.glViewMatrix()
        self.glViewport(0, 0, self.width, self.height)

        self.glClear()

    def glViewport(self, posX, posY, width, height):
        self.vpX = posX
        self.vpY = posY
        self.vpWidth = width
        self.vpHeight = height

        self.viewportMatrix = matrix(4, 4, [width/2, 0, 0, posX+width/2,
                                            0, height/2, 0, posY+height/2,
                                            0, 0, 0.5, 0.5,
                                            0, 0, 0, 1])
        self.glProjectionMatrix()

    def glClearColor(self, r, g, b):
        self.clearColor = color(r, g, b)

    def glColor(self, r, g, b):
        self.currColor = color(r, g, b)

    def glClear(self):
        self.pixels = [[self.clearColor for y in range(self.height)]
                       for x in range(self.width)]

    def glClearViewport(self, clr=None):
        for x in range(self.vpX, self.vpX + self.vpWidth):
            for y in range(self.vpY, self.vpY + self.vpHeight):
                self.glPoint(x, y, clr)

    def glPoint(self, x, y, clr=None):  # Window Coordinates
        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[x][y] = clr or self.currColor

    def scene_intersect(self, orig, dir, sceneObj):
        depth = float('inf')
        intersect = None

        for obj in self.scene:
            hit = obj.ray_intersect(orig, dir)
            if hit != None:
                if sceneObj != hit.sceneObj:
                    if hit.distance < depth:
                        intersect = hit
                        depth = hit.distance

        return intersect

    def cast_ray(self, orig, dir, sceneObj=None, recursion=0):
        intersect = self.scene_intersect(orig, dir, sceneObj)

        if intersect == None or recursion >= MAX_RECURSION_DEPTH:
            if self.envMap:
                return self.envMap.getEnvColor(dir)
            else:
                return (self.clearColor[0] / 255,
                        self.clearColor[1] / 255,
                        self.clearColor[2] / 255)

        material = intersect.sceneObj.material

        finalColor = np.array([0, 0, 0])
        objectColor = np.array([material.diffuse[0],
                                material.diffuse[1],
                                material.diffuse[2]])

        if material.matType == OPAQUE:
            for light in self.lights:
                diffuseColor = light.getDiffuseColor(intersect, self)
                specColor = light.getSpecColor(intersect, self)
                shadowIntensity = light.getShadowIntensity(intersect, self)

                lightColor = (diffuseColor + specColor) * (1 - shadowIntensity)

                finalColor = np.add(finalColor, lightColor)

        elif material.matType == REFLECTIVE:
            reflect = reflectVector(intersect.normal, np.array(dir) * -1)
            reflectColor = self.cast_ray(
                intersect.point, reflect, intersect.sceneObj, recursion + 1)
            reflectColor = np.array(reflectColor)

            specColor = np.array([0, 0, 0])
            for light in self.lights:
                specColor = np.add(
                    specColor, light.getSpecColor(intersect, self))

            finalColor = reflectColor + specColor

        elif material.matType == TRANSPARENT:
            outside = np.dot(dir, intersect.normal) < 0
            bias = intersect.normal * 0.001

            specColor = np.array([0, 0, 0])
            for light in self.lights:
                specColor = np.add(
                    specColor, light.getSpecColor(intersect, self))

            reflect = reflectVector(intersect.normal, np.array(dir) * -1)
            reflectOrig = np.add(intersect.point, bias) if outside else np.subtract(
                intersect.point, bias)
            reflectColor = self.cast_ray(
                reflectOrig, reflect, None, recursion + 1)
            reflectColor = np.array(reflectColor)

            kr = fresnel(intersect.normal, dir, material.ior)

            refractColor = np.array([0, 0, 0])
            if kr < 1:
                refract = refractVector(intersect.normal, dir, material.ior)
                refractOrig = np.subtract(
                    intersect.point, bias) if outside else np.add(intersect.point, bias)
                refractColor = self.cast_ray(
                    refractOrig, refract, None, recursion + 1)
                refractColor = np.array(refractColor)

            finalColor = reflectColor * kr + \
                refractColor * (1 - kr) + specColor

        finalColor *= objectColor

        if material.texture and intersect.texcoords:
            texColor = material.texture.getColor(
                intersect.texcoords[0], intersect.texcoords[1])
            if texColor is not None:
                finalColor *= np.array(texColor)

        r = min(1, finalColor[0])
        g = min(1, finalColor[1])
        b = min(1, finalColor[2])

        return (r, g, b)

    # Load OBJ
    def glViewMatrix(self, translate=V3(0, 0, 0), rotate=V3(0, 0, 0)):
        self.camMatrix = self.glCreateObjectMatrix(translate, rotate)
        self.viewMatrix = np.linalg.inv(self.camMatrix)

    def glProjectionMatrix(self, n=0.1, f=1000, fov=60):
        aspectRatio = self.vpWidth / self.vpHeight
        t = tan((fov * pi / 180) / 2) * n
        r = t * aspectRatio

        self.projectionMatrix = np.matrix([[n/r, 0, 0, 0],
                                           [0, n/t, 0, 0],
                                           [0, 0, -(f+n)/(f-n), -
                                            (2*f*n)/(f-n)],
                                           [0, 0, -1, 0]])

    def glCreateObjectMatrix(self, translate=V3(0, 0, 0), rotate=V3(0, 0, 0), scale=V3(1, 1, 1)):

        translation = np.matrix([[1, 0, 0, translate.x],
                                 [0, 1, 0, translate.y],
                                 [0, 0, 1, translate.z],
                                 [0, 0, 0, 1]])

        rotation = self.glCreateRotationMatrix(rotate.x, rotate.y, rotate.z)

        scaleMat = np.matrix([[scale.x, 0, 0, 0],
                              [0, scale.y, 0, 0],
                              [0, 0, scale.z, 0],
                              [0, 0, 0, 1]])

        return translation * rotation * scaleMat

    def glCreateRotationMatrix(self, pitch=0, yaw=0, roll=0):

        pitch *= pi/180
        yaw *= pi/180
        roll *= pi/180

        pitchMat = np.matrix([[1, 0, 0, 0],
                              [0, cos(pitch), -sin(pitch), 0],
                              [0, sin(pitch), cos(pitch), 0],
                              [0, 0, 0, 1]])

        yawMat = np.matrix([[cos(yaw), 0, sin(yaw), 0],
                            [0, 1, 0, 0],
                            [-sin(yaw), 0, cos(yaw), 0],
                            [0, 0, 0, 1]])

        rollMat = np.matrix([[cos(roll), -sin(roll), 0, 0],
                             [sin(roll), cos(roll), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        return pitchMat * rollMat * yawMat

    def glTransform(self, vertex, matrix):
        v = V4(vertex[0], vertex[1], vertex[2], 1)
        vt = matrix @ v
        vt = vt.tolist()[0]
        vf = V3(vt[0] / vt[3],
                vt[1] / vt[3],
                vt[2] / vt[3])

        return vf

    def glCamTransform(self, vertex):
        v = V4(vertex[0], vertex[1], vertex[2], 1)
        vt = self.viewportMatrix @ self.projectionMatrix @ self.viewMatrix @ v
        vt = vt.tolist()[0]
        vf = V3(vt[0] / vt[3],
                vt[1] / vt[3],
                vt[2] / vt[3])

        return vf

    def glDirTransform(self, dirVector, rotMatrix):
        v = V4(dirVector[0], dirVector[1], dirVector[2], 0)
        vt = rotMatrix @ v
        vt = vt.tolist()[0]
        vf = V3(vt[0],
                vt[1],
                vt[2])

        return vf

    def glTriangle_bc(self, A, B, C, verts=(), texCoords=(), normals=(), clr=None):
        # bounding box
        minX = round(min(A.x, B.x, C.x))
        minY = round(min(A.y, B.y, C.y))
        maxX = round(max(A.x, B.x, C.x))
        maxY = round(max(A.y, B.y, C.y))

        edge1 = np.subtract(verts[1], verts[0])
        edge2 = np.subtract(verts[2], verts[0])

        triangleNormal = np.cross(edge1, edge2)
        triangleNormal = triangleNormal / np.linalg.norm(triangleNormal)

        deltaUV1 = np.subtract(texCoords[1], texCoords[0])
        deltaUV2 = np.subtract(texCoords[2], texCoords[0])
        f = 1 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])

        tangent = [f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
                   f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
                   f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])]
        tangent = tangent / np.linalg.norm(tangent)

        bitangent = np.cross(triangleNormal, tangent)
        bitangent = bitangent / np.linalg.norm(bitangent)

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):

                u, v, w = baryCoords(A, B, C, V2(x, y))

                if 0 <= u and 0 <= v and 0 <= w:

                    z = A.z * u + B.z * v + C.z * w

                    if 0 <= x < self.width and 0 <= y < self.height:
                        if z < self.zbuffer[x][y] and -1 <= z <= 1:
                            self.zbuffer[x][y] = z

                            if self.active_shader:
                                r, g, b = self.active_shader(self,
                                                             baryCoords=(
                                                                 u, v, w),
                                                             vColor=clr or self.currColor,
                                                             texCoords=texCoords,
                                                             normals=normals,
                                                             triangleNormal=triangleNormal,
                                                             tangent=tangent,
                                                             bitangent=bitangent)

                                self.glPoint(x, y, color(r, g, b))
                            else:
                                self.glPoint(x, y, clr)

    def glClear(self):
        self.pixels = [[self.clearColor for y in range(self.height)]
                       for x in range(self.width)]

        self.zbuffer = [[float('inf') for y in range(self.height)]
                        for x in range(self.width)]

    def glLoadModel(self, filename, translate=V3(0, 0, 0), rotate=V3(0, 0, 0), scale=V3(1, 1, 1)):
        model = Obj(filename)
        modelMatrix = self.glCreateObjectMatrix(translate, rotate, scale)
        rotationMatrix = self.glCreateRotationMatrix(
            rotate[0], rotate[1], rotate[2])

        for face in model.faces:
            vertCount = len(face)

            v0 = model.vertices[face[0][0] - 1]
            v1 = model.vertices[face[1][0] - 1]
            v2 = model.vertices[face[2][0] - 1]

            v0 = self.glTransform(v0, modelMatrix)
            v1 = self.glTransform(v1, modelMatrix)
            v2 = self.glTransform(v2, modelMatrix)

            A = self.glCamTransform(v0)
            B = self.glCamTransform(v1)
            C = self.glCamTransform(v2)

            vt0 = model.texcoords[face[0][1] - 1]
            vt1 = model.texcoords[face[1][1] - 1]
            vt2 = model.texcoords[face[2][1] - 1]

            vn0 = model.normals[face[0][2] - 1]
            vn1 = model.normals[face[1][2] - 1]
            vn2 = model.normals[face[2][2] - 1]
            vn0 = self.glDirTransform(vn0, rotationMatrix)
            vn1 = self.glDirTransform(vn1, rotationMatrix)
            vn2 = self.glDirTransform(vn2, rotationMatrix)

            self.glTriangle_bc(A, B, C,
                               verts=(v0, v1, v2),
                               texCoords=(vt0, vt1, vt2),
                               normals=(vn0, vn1, vn2))

    def glRender(self):
        # Proyeccion
        t = tan((self.fov * np.pi / 180) / 2) * self.nearPlane
        r = t * self.vpWidth / self.vpHeight

        for y in range(self.vpY, self.vpY + self.vpHeight + 1, STEPS):
            for x in range(self.vpX, self.vpX + self.vpWidth + 1, STEPS):
                # Pasar de coordenadas de ventana a
                # coordenadas NDC (-1 a 1)
                Px = ((x + 0.5 - self.vpX) / self.vpWidth) * 2 - 1
                Py = ((y + 0.5 - self.vpY) / self.vpHeight) * 2 - 1

                Px *= r
                Py *= t

                direction = V3(Px, Py, -self.nearPlane)
                direction = direction / np.linalg.norm(direction)

                rayColor = self.cast_ray(self.camPosition, direction)

                if rayColor is not None:
                    rayColor = color(rayColor[0], rayColor[1], rayColor[2])
                    self.glPoint(x, y, rayColor)

    def glFinish(self, filename):
        with open(filename, "wb") as file:
            # Header
            file.write(bytes('B'.encode('ascii')))
            file.write(bytes('M'.encode('ascii')))
            file.write(dword(14 + 40 + (self.width * self.height * 3)))
            file.write(dword(0))
            file.write(dword(14 + 40))

            # InfoHeader
            file.write(dword(40))
            file.write(dword(self.width))
            file.write(dword(self.height))
            file.write(word(1))
            file.write(word(24))
            file.write(dword(0))
            file.write(dword(self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))

            # Color table
            for y in range(self.height):
                for x in range(self.width):
                    file.write(self.pixels[x][y])
