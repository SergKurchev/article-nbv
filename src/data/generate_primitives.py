import os
import math
from pathlib import Path
import shutil

def write_stl(filename, name, faces):
    lines = [f"solid {name}"]
    for face in faces:
        v1, v2, v3 = face
        
        # Skip degenerate triangles (duplicate vertices)
        if v1 == v2 or v1 == v3 or v2 == v3:
            continue
            
        # normal computation
        u = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
        v = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]]
        nx = u[1]*v[2] - u[2]*v[1]
        ny = u[2]*v[0] - u[0]*v[2]
        nz = u[0]*v[1] - u[1]*v[0]
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 1e-12:
            nx /= length; ny /= length; nz /= length
        else:
            # Skip triangles with practically zero area
            continue
        
        lines.append(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}")
        lines.append(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}")
        lines.append(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

def write_json(filename, faces):
    import json
    # Convert list of faces [[v1,v2,v3], ...] to vertices/indices
    vertices = []
    indices = []
    v_map = {}
    
    for face in faces:
        face_indices = []
        for v in face:
            v_tuple = tuple(v)
            if v_tuple not in v_map:
                v_map[v_tuple] = len(vertices)
                vertices.append(list(v))
            face_indices.append(v_map[v_tuple])
        indices.extend(face_indices)
        
    with open(filename, 'w') as f:
        json.dump({"vertices": vertices, "indices": indices}, f)

def get_box_faces(dx, dy, dz):
    v = [
        [-dx,-dy,-dz], [dx,-dy,-dz], [dx,dy,-dz], [-dx,dy,-dz],
        [-dx,-dy, dz], [dx,-dy, dz], [dx,dy, dz], [-dx,dy, dz]
    ]
    # Fixed winding order for outward normals (counter-clockwise when viewed from outside)
    indices = [
        [0,1,2], [0,2,3], # bottom (-Z face)
        [4,6,5], [4,7,6], # top (+Z face)
        [0,5,1], [0,4,5], # front (-Y face)
        [1,6,2], [1,5,6], # right (+X face)
        [2,7,3], [2,6,7], # back (+Y face)
        [3,4,0], [3,7,4]  # left (-X face)
    ]
    return [[v[i] for i in face] for face in indices]

def get_sphere_faces(radius, rings=16, sectors=16):
    R = 1.0/(rings-1)
    S = 1.0/(sectors-1)
    v = []
    for r in range(rings):
        for s in range(sectors):
            y = math.sin(-math.pi/2 + math.pi * r * R)
            x = math.cos(2*math.pi * s * S) * math.sin(math.pi * r * R)
            z = math.sin(2*math.pi * s * S) * math.sin(math.pi * r * R)
            v.append([x*radius, y*radius, z*radius])

    faces = []
    for r in range(rings-1):
        for s in range(sectors):
            s_next = (s + 1) % sectors
            i0 = r * sectors + s
            i1 = r * sectors + s_next
            i2 = (r+1) * sectors + s_next
            i3 = (r+1) * sectors + s
            # Fixed winding order for outward normals
            faces.append([v[i0], v[i2], v[i1]])
            faces.append([v[i0], v[i3], v[i2]])
    return faces

def get_cylinder_faces(radius, height, sectors=32):
    v = []
    # Bottom center
    v.append([0, 0, -height/2])
    # Top center
    v.append([0, 0, height/2])

    base_idx = 2
    for s in range(sectors):
        angle = 2*math.pi * s / sectors
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        v.append([x, y, -height/2])
        v.append([x, y, height/2])

    faces = []
    for s in range(sectors):
        s_next = (s + 1) % sectors
        i0 = base_idx + s*2
        i1 = base_idx + s*2 + 1
        i2 = base_idx + s_next*2
        i3 = base_idx + s_next*2 + 1

        # Side faces (outward normals) - fixed winding order
        faces.append([v[i0], v[i2], v[i1]])
        faces.append([v[i1], v[i2], v[i3]])
        # Bottom cap (normal pointing down -Z)
        faces.append([v[0], v[i2], v[i0]])
        # Top cap (normal pointing up +Z)
        faces.append([v[1], v[i1], v[i3]])
    return faces

def get_cone_faces(radius, height, sectors=32):
    v = [[0, 0, height/2], [0, 0, -height/2]] # Tip and base center
    base_idx = 2
    for s in range(sectors):
        angle = 2*math.pi * s / sectors
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        v.append([x, y, -height/2])

    faces = []
    for s in range(sectors):
        s_next = (s + 1) % sectors
        i1 = base_idx + s
        i2 = base_idx + s_next
        # Fixed winding order for outward normals
        faces.append([v[0], v[i2], v[i1]]) # side (tip to base, counter-clockwise)
        faces.append([v[1], v[i1], v[i2]]) # base (center to edge, counter-clockwise from below)
    return faces

def get_hourglass_faces(radius, height, sectors=32):
    # Two cones meeting at origin (center point)
    # Upper cone: tip at origin, base at top
    faces1 = get_cone_faces(radius, height/2, sectors)
    # Flip vertically so tip is at origin and base is at +height/2
    f1 = [[[vx, vy, -vz + height/4] for vx, vy, vz in face] for face in faces1]

    # Lower cone: tip at origin, base at bottom
    faces2 = get_cone_faces(radius, height/2, sectors)
    # Flip vertically so tip is at origin and base is at -height/2
    f2 = [[[vx, vy, vz - height/4] for vx, vy, vz in face] for face in faces2]
    return f1 + f2

def get_pyramid_faces(base, height):
    v = [
        [0,0,height/2], # tip
        [-base/2, -base/2, -height/2],
        [base/2, -base/2, -height/2],
        [base/2, base/2, -height/2],
        [-base/2, base/2, -height/2]
    ]
    # Fixed winding order for outward normals
    return [
        [v[0], v[2], v[1]], [v[0], v[3], v[2]],  # sides (counter-clockwise from outside)
        [v[0], v[4], v[3]], [v[0], v[1], v[4]],
        [v[1], v[2], v[4]], [v[2], v[3], v[4]]   # base (counter-clockwise from below)
    ]

def get_octahedron(size):
    # double pyramid
    f1 = get_pyramid_faces(size, size)
    f2 = [[[vx, -vy, -vz] for vx, vy, vz in face] for face in f1]
    return f1 + f2

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent / "src" / "data" / "objects"
    prim_dir = base_dir / "primitives"
    prim_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 8 primitives
    # (Scale is arbitrary, as asset_loader scales them to 15cm max dimension)
    shapes = [
        ("Object_01", get_sphere_faces(1.0)),                   # Sphere
        ("Object_02", get_box_faces(1.0, 1.0, 1.0)),            # Cube
        ("Object_03", get_box_faces(0.5, 0.5, 1.5)),            # Elongated Box
        ("Object_04", get_pyramid_faces(2.0, 2.0)),             # Pyramid
        ("Object_05", get_cone_faces(1.0, 2.0)),                # Cone
        ("Object_06", get_cylinder_faces(1.0, 2.0)),            # Cylinder
        ("Object_07", get_hourglass_faces(1.0, 3.0)),           # Hourglass
        ("Object_08", get_octahedron(2.0))                      # Double Pyramid
    ]
    
    for obj_name, faces in shapes:
        obj_folder = prim_dir / obj_name
        obj_folder.mkdir(exist_ok=True)
        
        # Save STL (for visualization/user)
        stl_path = obj_folder / f"{obj_name}.STL"
        write_stl(str(stl_path), obj_name, faces)
        
        # Save JSON (for stable PyBullet loading)
        json_path = obj_folder / "mesh.json"
        write_json(str(json_path), faces)
        
        print(f"Generated {obj_name} (STL + JSON)")
        
        # Copy texture if exists
        tex_path = base_dir / "texture.png"
        if tex_path.exists():
            shutil.copy2(tex_path, obj_folder / "texture.png")
