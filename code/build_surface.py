import numpy as np
from skimage.measure import marching_cubes
import json

# Regular tetrahedron, unit circumradius
s3 = np.sqrt(3)
tet = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float) / s3

N, LIM = 80, 2.2
c = np.linspace(-LIM, LIM, N)
X, Y, Z = np.meshgrid(c, c, c, indexing='ij')

F = sum(np.sqrt((X-v[0])**2 + (Y-v[1])**2 + (Z-v[2])**2) for v in tet)
print(f"Field range: {F.min():.3f} to {F.max():.3f}")

sp = (c[1]-c[0],)*3
out = {}
for k in [4.4, 5.0, 6.0, 7.5]:
    sv, sf, sn, _ = marching_cubes(F, k, spacing=sp)
    sv[:,0] += c[0]; sv[:,1] += c[0]; sv[:,2] += c[0]
    # Compute per-vertex normals and radius (for coloring)
    r = np.linalg.norm(sv, axis=1)
    out[str(k)] = {
        'verts': sv.tolist(),
        'faces': sf.tolist(),
        'radius': r.tolist()
    }
    print(f"k={k}: {len(sv)} verts, {len(sf)} faces")

with open('/sessions/adoring-happy-cerf/mnt/outputs/mesh_data.json','w') as f:
    json.dump({'meshes': out, 'tet': tet.tolist()}, f)
print("Done.")
