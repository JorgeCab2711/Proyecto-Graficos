class Obj(object):
    def __init__(self, filename):
        with open(filename, "r") as file:
            self.lines = file.read().splitlines()
        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.faces = []
        maxValue = 0

        for line in self.lines:
            try:
                prefix, value = line.split(' ', 1)
            except:
                continue

            if prefix == 'v':  # Vertices
                arr = list(map(float, value.split(' ')))
                ma = max(arr)
                if ma > maxValue:
                    maxValue = ma
                self.vertices.append(arr)
            elif prefix == 'vt':
                arr = list(map(float, value.split(' ')))
                if len(arr) == 2:
                    arr.append(0.0)
                self.texcoords.append(arr)
            elif prefix == 'vn':
                self.normals.append(list(map(float, value.split(' '))))
            elif prefix == 'f':
                vertList = []
                for vert in value.strip().split(' '):
                    indices = vert.split('/')
                    indices = ["0" if item == "" else item for item in indices]
                    indices = map(int, indices)
                    indices = list(indices)
                    vertList.append(indices)
                self.faces.append(vertList)

        for vert in self.vertices:
            vert[0] /= maxValue
            vert[1] /= maxValue
            vert[2] /= maxValue
