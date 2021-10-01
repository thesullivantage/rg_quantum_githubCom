OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cz q[1],q[0];
u2(0.72983379,1.1765903) q[1];
ry(1.7396931) q[0];
