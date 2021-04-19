from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate


class QGate:
    def __init__(self, type=None, qiskitGate=None, mainBit=None, controlBit=None, isControl=False):
        self.type = type
        self.mainBit = mainBit
        self.controlBit = controlBit
        self.qiskitGate = qiskitGate
        self.isControl = isControl

    def __str__(self):
        return str([self.type, self.mainBit, self.controlBit])

    def __repr__(self):
        return str([self.type, self.mainBit, self.controlBit])

    def qasmString(self):
        if not self.isControl:
            return self.type + ' q[' + self.mainBit + '];\n'
        else:
            return self.type + ' q[' + self.controlBit + '], q[' + self.mainBit + '];\n'

    def swap(self, type, mainBit=None, controlBit=None):
        if mainBit != None:
            self.mainBit = mainBit
        if controlBit != None:
            self.controlBit = controlBit
        self.type = type

    def clear(self):
        self.type = None
        self.mainBit = None
        self.controlBit = None
        self.qiskitGate = None

    def getMatrix(self):
        return self.qiskitGate.to_matrix()

    def getInverse(self):
        return self.qiskitGate.inverse()
