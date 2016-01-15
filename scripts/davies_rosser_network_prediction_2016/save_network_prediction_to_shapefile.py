import shapefile

"""
assume the following are defined:
x: a list of edges
b: a N_DAYS x N_SEGMENTS array of predicted risk values
"""

fields = ["risk%d" % i for i in range(0, 180, 30)]

w = shapefile.Writer(shapefile.POLYLINE)
for f in fields:
    w.field(f, 'F', size=24, decimal=12)

# for i in range(len(x)):
#     w.line(parts=[x[i].linestring.coords[:]])
#     rec = dict([(fields[j1], b[j2, i]) for j1, j2 in enumerate(range(0, 180, 30))])
#     w.record(**rec)

# idx is a list of indices for edges to include

for i in idx:
    w.line(parts=[x[i].linestring.coords[:]])
    rec = dict([(fields[j1], b[j2, i]) for j1, j2 in enumerate(range(0, 180, 30))])
    w.record(**rec)

w.save('birmingham_network_risk')