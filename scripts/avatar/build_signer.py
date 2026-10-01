"""Run inside Blender: build the realistic 3D signer with MPFB2 and export static/models/signer.glb.

    tools/blender-5.2.2-windows-x64/blender.exe --background --python scripts/avatar/build_signer.py
    (set SIGNER_PRESET=man first for the male signer -> static/models/signer-male.glb)

Needs MPFB2 and its asset packs installed (scripts/avatar/install_mpfb_assets.py). Every choice below
(body, skin, hair, clothes) is plain data at the top - edit and re-run to change the look.
The export matches the manual MPFB "Export copy" + glTF steps: game_engine rig (all finger bones),
ARKit face units as shape keys, helpers removed, textures shrunk for the browser.
All MPFB assets used are CC0.
"""
import os
import sys
from pathlib import Path

import bpy

from bl_ext.user_default.mpfb.services.assetservice import AssetService
from bl_ext.user_default.mpfb.services.exportservice import ExportService
from bl_ext.user_default.mpfb.services.faceservice import FaceService
from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.objectservice import ObjectService
from bl_ext.user_default.mpfb.services.targetservice import TargetService

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ the look (edit these)
# SIGNER_PRESET picks the character: "woman" (default) -> signer.glb, "man" -> signer-male.glb.
PRESETS = {
    "woman": {
        "file": "signer",
        "macro": {  # 0..1 sliders, as in MPFB's Model > Phenotype panel
            "gender": 0.0,        # 0 = female, 1 = male
            "age": 0.5,           # 0.5 = about 25 years
            "muscle": 0.5,
            "weight": 0.48,
            "height": 0.45,
            "proportions": 0.6,
            "race": {"asian": 0.3, "caucasian": 0.45, "african": 0.25},  # a South Asian mix
        },
        "skin": "cutoff3d_indian_female_skin",
        "hair": "ponytail01",       # tied back: keeps the face, the pallu and the signing space clear
                                    # (long and open: "faydaen_hair_1")
        "earrings": True,           # small gold hoops, attached to the head bone
        "eyebrows": "eyebrow001",
        "eyelashes": "eyelashes01",
        "clothes": ["toigo_basic_tucked_t-shirt", "toigo_long_full_skirt"],  # blouse, saree skirt (barefoot)
        "saree": {  # the pallu is generated (add_pallu); the skirt is recoloured to match
            "color": (0.46, 0.04, 0.10),   # deep red silk
            "border": (0.80, 0.58, 0.18),  # gold zari border
        },
    },
    "man": {
        "file": "signer-male",
        "macro": {
            "gender": 1.0,
            "age": 0.55,          # late twenties
            "muscle": 0.55,
            "weight": 0.5,
            "height": 0.55,
            "proportions": 0.6,
            # MakeHuman's "asian" shapes East Asian features; a South Asian face is mostly the other two
            "race": {"asian": 0.05, "caucasian": 0.65, "african": 0.3},
        },
        "skin": "young_caucasian_male",  # no Indian male skin in the CC0 packs: neutral features, and the
                                         # browser tints it to a warm brown (CHARACTERS in avatar.js)
        "hair": "short04",               # black, side-swept
        "earrings": False,
        "eyebrows": "eyebrow002",
        "eyelashes": "eyelashes02",
        "clothes": ["elvs_male_shirt_untucked_bd1", "toigo_wool_pants", "shoes02",
                    os.getenv("SIGNER_FACIAL_HAIR", "rehmanpolanski_moustache_viking")],  # kurta (tailored below), pyjama,
                                                                                    # moustache (keeps mouth and chin visible)
        "kurta": "elvs_male_shirt_untucked_bd1",  # stretched to the knee by tailor_kurta()
    },
}
PRESET = PRESETS[os.getenv("SIGNER_PRESET", "woman")]
OUT_GLB = Path(os.getenv("SIGNER_OUT", ROOT / "static" / "models" / f"{PRESET['file']}.glb"))  # SIGNER_OUT: try a variant elsewhere
OUT_BLEND = ROOT / "data" / "avatar" / f"{PRESET['file']}.blend"   # editable source, open it in Blender to tweak by hand

MACRO = PRESET["macro"]
SKIN = os.getenv("SIGNER_SKIN", PRESET["skin"])
HAIR = os.getenv("SIGNER_HAIR", PRESET["hair"])
EARRINGS = PRESET["earrings"]
BODY_PARTS = [  # (asset folder, asset name, MPFB asset type)
    ("eyes", "low-poly", "Eyes"),
    ("eyebrows", PRESET["eyebrows"], "Eyebrows"),
    ("eyelashes", PRESET["eyelashes"], "Eyelashes"),
    ("teeth", "teeth_base", "Teeth"),
    ("tongue", "tongue01", "Tongue"),
    ("hair", HAIR, "Hair"),
    *[("clothes", c, "Clothes") for c in PRESET["clothes"]],
]
RIG = "game_engine"
MAX_TEXTURE = {"skin": 2048, "other": 1024}  # px; keeps the .glb small enough for the browser


def find_asset(folder: str, name: str, material: bool = False) -> str:
    paths = AssetService.list_mhmat_assets(folder) if material else AssetService.list_mhclo_assets(folder)
    for p in paths:
        if Path(p).stem == name:
            return str(p)
    raise FileNotFoundError(f"MPFB asset '{name}' not found in '{folder}' - is its asset pack installed?")


def shrink_textures() -> None:
    for img in bpy.data.images:
        if not img.size[0]:
            continue
        is_skin = any(k in img.name.lower() for k in ("skin", "cutoff3d", SKIN.lower()))
        limit = MAX_TEXTURE["skin"] if is_skin else MAX_TEXTURE["other"]
        w, h = img.size
        if max(w, h) > limit:
            s = limit / max(w, h)
            img.scale(int(w * s), int(h * s))
            print(f"  texture {img.name}: {w}x{h} -> {int(w * s)}x{int(h * s)}")


SEE_THROUGH = ("eyebrow", "eyelash", "ponytail", "hair", "beard", "moustache")  # textures that need their alpha cut-outs


def make_opaque(objects) -> None:
    """MPFB's game materials all feed texture alpha into the shader, so glTF marks every material BLEND:
    the body then sorts wrongly (eyes and teeth show through the face, clothes vanish). Disconnect alpha
    on everything except hair-like cards; those are drawn with an alpha cut-out in the browser."""
    for obj in objects:
        for slot in getattr(obj, "material_slots", []):
            mat = slot.material
            if not mat or not mat.use_nodes or any(k in mat.name.lower() for k in SEE_THROUGH):
                continue
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    for link in list(node.inputs["Alpha"].links):
                        mat.node_tree.links.remove(link)
                    node.inputs["Alpha"].default_value = 1.0
            if hasattr(mat, "surface_render_method"):
                mat.surface_render_method = "DITHERED"


CLOTHES_OFFSET = 0.004  # metres: clothes pushed out along their normals so the body never pokes through


def tailor_kurta(objects, rig, shirt: str, hem_above_knee: float = 0.04, flare: float = 0.32) -> None:
    """Turn a long-sleeved untucked shirt into a kurta: below the waist, every vertical strip of the shirt
    is stretched so its own hem lands just above the knee (a shirt's curved hem becomes a level kurta
    hem) and flared a little, so it hangs as one tube around both legs. The legs stay still while
    signing, so the long hem never cuts through them."""
    import math

    from mathutils import Vector

    obj = next((o for o in objects if o.type == "MESH" and shirt in o.name), None)
    if obj is None:
        print(f"  kurta: no '{shirt}' mesh found", flush=True)
        return
    bones = rig.data.bones
    pelvis = rig.matrix_world @ bones["pelvis"].head_local
    knee = rig.matrix_world @ bones["calf_l"].head_local
    mw, inv = obj.matrix_world, obj.matrix_world.inverted()
    waist, target = pelvis.z + 0.06, knee.z + hem_above_knee
    bins = 48

    def sector(p) -> int:
        return int((math.atan2(p.y - pelvis.y, p.x - pelvis.x) + math.pi) / (2 * math.pi) * bins) % bins

    hem = [waist] * bins                                  # lowest point of the shirt in each direction
    for v in obj.data.vertices:
        p = mw @ v.co
        hem[sector(p)] = min(hem[sector(p)], p.z)
    for i in range(bins):                                 # smooth over neighbours: no single low tail
        hem[i] = min(hem[(i - 1) % bins], hem[i], hem[(i + 1) % bins]) if hem[i] >= waist - 0.01 else hem[i]

    def stretch(co):
        p = mw @ co
        if p.z >= waist:
            return co
        h = min(hem[sector(p)], waist - 0.05)
        z = waist - (waist - p.z) * (waist - target) / (waist - h)
        # widens quickly below the waist (clear of the thighs), then hangs almost straight to the hem
        grow = 1.0 + flare * math.sqrt(min(1.0, (waist - z) / (waist - target)))
        return inv @ Vector((pelvis.x + (p.x - pelvis.x) * grow, pelvis.y + (p.y - pelvis.y) * grow, z))

    # the mesh may carry shape keys (then the basis key, not v.co, is what gets exported): stretch all of them
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    for kb in keys:
        for d in kb.data:
            d.co = stretch(d.co.copy())
    for v in obj.data.vertices:
        v.co = stretch(v.co.copy())
    obj.data.update()
    print(f"  kurta: hem {min(hem):.2f}-{max(hem):.2f} m -> {target:.2f} m (knee {knee.z:.2f} m)", flush=True)


def _saree_texture(color, border, w: int = 64, h: int = 256):
    """Silk with a gold border along both long edges and a broad decorated band at the pallu's end."""
    import numpy as np

    px = np.empty((h, w, 4), np.float32)
    px[...] = (*color, 1.0)
    u = np.arange(w) / (w - 1)
    px[:, (u < 0.09) | (u > 0.91), :3] = border
    px[:, ((u > 0.12) & (u < 0.14)) | ((u > 0.86) & (u < 0.88)), :3] = border
    v = np.arange(h) / (h - 1)                         # 0 at the hip .. 1 at the hanging end
    band = v > 0.86
    px[band, :, :3] = border
    px[band & (np.sin(v * 220) > 0.55), :, :3] = color
    img = bpy.data.images.new("saree_pallu", w, h)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def _flat_material(name: str, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*(c ** 2.2 for c in color), 1.0)  # sRGB -> linear
    bsdf.inputs["Roughness"].default_value = 0.45     # a little silk sheen
    return mat


def dress_saree(rig, objects, color, border) -> None:
    """Saree: the long skirt and the generated pallu in the saree colour, with a gold border."""
    skirt = next((o for o in objects if o.type == "MESH" and "skirt" in o.name), None)
    if skirt:
        slim_skirt(skirt, rig)
        skirt.data.materials.clear()
        skirt.data.materials.append(_flat_material("saree.skirt", color))
    add_pallu(rig, objects, color, border)


def slim_skirt(obj, rig, hem_radius: float = 0.24) -> None:
    """A full skirt flares like a gown; a saree is wrapped. Below the hips, pull the skirt in towards
    the body's axis, more and more down to the hem, until the hem is about hem_radius wide."""
    from mathutils import Vector

    pel = rig.matrix_world @ rig.data.bones["pelvis"].head_local
    mw, inv = obj.matrix_world, obj.matrix_world.inverted()
    world = [mw @ v.co for v in obj.data.vertices]
    start, low = pel.z - 0.08, min(p.z for p in world)
    radii = sorted(((p.x - pel.x) ** 2 + (p.y - pel.y) ** 2) ** 0.5 for p in world if p.z < low + 0.15)
    r_hem = radii[int(len(radii) * 0.9)] if radii else hem_radius   # the outer flare, not the inner edge
    ratio = min(1.0, hem_radius / max(r_hem, 1e-6))

    def slim(co):
        p = mw @ co
        if p.z >= start:
            return co
        k = 1.0 - (1.0 - ratio) * min(1.0, (start - p.z) / (start - low))
        return inv @ Vector((pel.x + (p.x - pel.x) * k, pel.y + (p.y - pel.y) * k, p.z))

    for kb in (obj.data.shape_keys.key_blocks if obj.data.shape_keys else []):
        for d in kb.data:
            d.co = slim(d.co.copy())
    for v in obj.data.vertices:
        v.co = slim(v.co.copy())
    obj.data.update()
    print(f"  saree skirt: hem radius {r_hem:.2f} m -> {r_hem * ratio:.2f} m", flush=True)


def add_pallu(rig, objects, color, border) -> None:
    """Generate the pallu: a strip of cloth from the right hip, diagonally across the chest, over the left
    shoulder and down the back to the waist, laid 2 cm above the blouse and skirt (found by casting rays).
    It is skinned by copying, for each of its vertices, the bone weights of the nearest body/clothes
    vertex, so it moves with the torso and the left shoulder."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.kdtree import KDTree

    deps = bpy.context.evaluated_depsgraph_get()
    surfaces = [o for o in objects if o.type == "MESH" and (
        any(k in o.name for k in ("t-shirt", "skirt"))
        or (o.data.shape_keys and "eyeBlinkLeft" in o.data.shape_keys.key_blocks))]  # blouse, skirt, body
    verts, polys, owners = [], [], []
    for o in surfaces:
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        base = len(verts)
        verts += [o.matrix_world @ v.co for v in me.vertices]
        owners += [(o, i) for i in range(len(me.vertices))]
        polys += [[base + i for i in poly.vertices] for poly in me.polygons]
        ev.to_mesh_clear()
    bvh = BVHTree.FromPolygons(verts, polys)

    def bone(n):
        return rig.matrix_world @ rig.data.bones[n].head_local

    pel, sp2, sp3, nk, clav, ua = (bone(n) for n in ("pelvis", "spine_02", "spine_03", "neck_01", "clavicle_l", "upperarm_l"))
    side = 1.0 if ua.x > 0 else -1.0                  # the character's left shoulder (Blender +X)
    mid_y = (pel.y + sp3.y) / 2
    sh_x = clav.x + 0.65 * (ua.x - clav.x)            # over the left shoulder, clear of the neck
    F, T, K = Vector((0, 1, 0)), Vector((0, 0, -1)), Vector((0, -1, 0))  # rays from the front, above, behind
    # (ray origin, ray direction, cloth width) along the pallu; Blender -Y is the front of the body
    ctrl = [
        (Vector((-0.11 * side, -0.6, pel.z + 0.04)), F, 0.30),
        (Vector((-0.03 * side, -0.6, (pel.z + sp2.z) / 2)), F, 0.26),
        (Vector((0.05 * side, -0.6, sp3.z)), F, 0.22),
        (Vector((sh_x * 0.8, -0.6, nk.z - 0.05)), F, 0.16),
        (Vector((sh_x, mid_y, nk.z + 0.5)), T, 0.12),
        (Vector((sh_x * 0.85, 0.6, nk.z - 0.06)), K, 0.20),
        (Vector((sh_x * 0.8, 0.6, sp3.z - 0.05)), K, 0.24),    # falls straight down the back
        (Vector((sh_x * 0.8, 0.6, pel.z + 0.12)), K, 0.24),    # ends above the waist: below it would stick out
    ]

    def catmull(pts, t):
        n = len(pts) - 1
        i = min(int(t * n), n - 1)
        u = t * n - i
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n)]
        return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                      + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)

    def lerp(vals, t):
        n = len(vals) - 1
        i = min(int(t * n), n - 1)
        u = t * n - i
        return vals[i] * (1 - u) + vals[i + 1] * u

    origins, dirs, widths = [c[0] for c in ctrl], [c[1] for c in ctrl], [c[2] for c in ctrl]
    S, A, LIFT = 90, 7, 0.018                          # LIFT: metres above the blouse, so it never pokes through
    centres, normals, ts = [], [], []
    for k in range(S):
        t = k / (S - 1)
        hit = bvh.ray_cast(catmull(origins, t), lerp(dirs, t).normalized())
        if hit[0] is None:
            continue
        centres.append(hit[0] + hit[1] * LIFT)
        normals.append(hit[1].normalized())
        ts.append(t)
    for _ in range(4):                                # smooth the bumps of the surface out of the line
        centres = [centres[0]] + [(centres[i - 1] + centres[i] * 2 + centres[i + 1]) / 4
                                  for i in range(1, len(centres) - 1)] + [centres[-1]]
        normals = [normals[0]] + [(normals[i - 1] + normals[i] * 2 + normals[i + 1]).normalized()
                                  for i in range(1, len(normals) - 1)] + [normals[-1]]

    pv, uvs = [], []
    for i, (c, nrm, t) in enumerate(zip(centres, normals, ts)):
        tan = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)]).normalized()
        across = tan.cross(nrm).normalized()
        w = lerp(widths, t)
        for j in range(A):
            q = c + across * ((j / (A - 1) - 0.5) * w)
            hit = bvh.ray_cast(q + nrm * 0.12, -nrm, 0.3)   # lay each point on the body as well, but not
            near = hit[0] is not None and (hit[0] + hit[1] * LIFT - q).length < 0.03  # onto the neck or an arm
            pv.append(hit[0] + hit[1] * LIFT if near else q)
            uvs.append((j / (A - 1), t))
    rows = len(centres)
    faces = [(r * A + j, r * A + j + 1, (r + 1) * A + j + 1, (r + 1) * A + j)
             for r in range(rows - 1) for j in range(A - 1)]
    me = bpy.data.meshes.new("saree_pallu")
    me.from_pydata([tuple(v) for v in pv], [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            uv.data[li].uv = uvs[me.loops[li].vertex_index]

    mat = _flat_material("saree.pallu", color)
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = _saree_texture(color, border)
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.use_backface_culling = False                  # both sides of the cloth can show
    me.materials.append(mat)
    pallu = bpy.data.objects.new("saree_pallu", me)
    bpy.context.scene.collection.objects.link(pallu)

    # skin it: bone weights of the nearest body / blouse / skirt vertex
    kd = KDTree(len(verts))
    for i, v in enumerate(verts):
        kd.insert(v, i)
    kd.balance()
    groups = {}
    for vi, v in enumerate(me.vertices):
        _, idx, _ = kd.find(v.co)
        src, si = owners[idx]
        for g in src.data.vertices[si].groups:
            name = src.vertex_groups[g.group].name
            if name not in groups:
                groups[name] = pallu.vertex_groups.new(name=name)
            groups[name].add([vi], g.weight, "REPLACE")
    pallu.parent = rig
    mod = pallu.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    print(f"  pallu: {rows} rows x {A}, {len(groups)} bone groups", flush=True)


def inflate_clothes(objects) -> None:
    """Add a small outward Displace (before the Armature modifier) to clothing meshes; glTF export applies it."""
    for obj in objects:
        if obj.type != "MESH" or not any(k in obj.name.lower() for k in ("shirt", "pants", "sweater", "top")):
            continue
        mod = obj.modifiers.new("inflate", "DISPLACE")
        mod.mid_level, mod.strength = 0.0, CLOTHES_OFFSET
        with bpy.context.temp_override(object=obj):
            while obj.modifiers.find("inflate") > 0:
                bpy.ops.object.modifier_move_up(modifier="inflate")


def add_earrings(rig, lobes) -> None:
    """Gold hoop earrings hung from the ear lobes, parented to the head bone so they move with the head."""
    from mathutils import Matrix, Vector

    gold = bpy.data.materials.new("earring_gold")
    gold.use_nodes = True
    bsdf = next(n for n in gold.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.83, 0.62, 0.25, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.28
    for side, lobe in lobes.items():
        bpy.ops.mesh.primitive_torus_add(major_radius=0.011, minor_radius=0.0013, major_segments=32, minor_segments=8)
        ring = bpy.context.active_object
        ring.name = f"earring_{'l' if side > 0 else 'r'}"
        ring.data.materials.append(gold)
        world = Matrix.Translation(lobe + Vector((0.0015 * side, 0.0, -0.0105))) @ Matrix.Rotation(1.5708, 4, "Y")
        ring.parent, ring.parent_type, ring.parent_bone = rig, "BONE", "head"
        ring.matrix_parent_inverse = Matrix.Identity(4)
        ring.matrix_world = world
        print(f"  earring at {tuple(round(c, 3) for c in lobe)}", flush=True)


def ear_lobes(rig, body) -> dict:
    """{+1: lobe, -1: lobe} world positions: the lowest point of each ear's outer edge on the body mesh."""
    verts = [body.matrix_world @ v.co for v in body.data.vertices]
    head = rig.matrix_world @ rig.data.bones["head"].head_local
    lobes = {}
    for side in (1, -1):  # Blender +X is the character's left
        band = [v for v in verts if v.x * side > 0.04 and head.z - 0.02 < v.z < head.z + 0.1 and abs(v.y - head.y) < 0.07]
        if band:
            outer = max(v.x * side for v in band)
            lobes[side] = min((v for v in band if v.x * side > outer - 0.012), key=lambda v: v.z)
    return lobes


def clear_face_and_ears(hair, eyes, lobes) -> None:
    """Delete hair strands hanging in front of the face (signers' faces must stay visible: expressions
    carry grammar) and around the ear lobes (hair tucked behind the ears, so earrings show)."""
    import bmesh
    from mathutils import Vector

    corners = [eyes.matrix_world @ Vector(c) for c in eyes.bound_box]
    eye_z = sum(c.z for c in corners) / 8
    eye_front = min(c.y for c in corners)            # Blender -Y is the front of the face
    bm = bmesh.new()
    bm.from_mesh(hair.data)
    mw = hair.matrix_world
    doomed = []
    for v in bm.verts:
        p = mw @ v.co
        in_face = abs(p.x) < 0.062 and eye_z - 0.12 < p.z < eye_z + 0.03 and p.y < eye_front + 0.035
        near_ear = any((p - (lobe + Vector((0, 0, -0.008)))).length < 0.035 and p.x * side > abs(lobe.x) - 0.012
                       for side, lobe in lobes.items())
        if in_face or near_ear:
            doomed.append(v)
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(hair.data)
    bm.free()
    print(f"  hair: removed {len(doomed)} vertices in front of the face / around the ears", flush=True)


def main() -> None:
    for obj in list(bpy.data.objects):  # start from an empty scene (keeps the MPFB extension loaded)
        bpy.data.objects.remove(obj, do_unlink=True)

    macro = TargetService.get_default_macro_info_dict()
    macro.update({k: v for k, v in MACRO.items() if k != "race"})
    macro["race"] = dict(MACRO["race"])
    basemesh = HumanService.create_human(macro_detail_dict=macro)
    print("created human", basemesh.name, flush=True)

    HumanService.set_character_skin(find_asset("skins", SKIN, material=True), basemesh, skin_type="GAMEENGINE")
    # rig first: assets added afterwards are parented to (and weighted for) the rig, so the export copy takes them
    HumanService.add_builtin_rig(basemesh, RIG, import_weights=True)
    print("  rig", RIG, flush=True)
    hair_name = next(name for _, name, atype in BODY_PARTS if atype == "Hair")
    for folder, name, atype in BODY_PARTS:
        HumanService.add_mhclo_asset(find_asset(folder, name), basemesh, asset_type=atype,
                                     subdiv_levels=0, material_type="GAMEENGINE")
        print("  added", atype, name, flush=True)

    OUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))

    # --- export copy (same steps as MPFB's "Create export copy" button)
    copy_root = ExportService.create_character_copy(basemesh, name_suffix="_export_copy")
    new_basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(copy_root)
    TargetService.bake_targets(new_basemesh)
    FaceService.load_targets(new_basemesh, load_microsoft_visemes=False, load_meta_visemes=False,
                             load_arkit_faceunits=True)
    FaceService.interpolate_targets(new_basemesh)
    ExportService.bake_modifiers_remove_helpers(new_basemesh, bake_masks=True, bake_subdiv=False,
                                                remove_helpers=True, also_proxy=True)
    for obj in [new_basemesh, *new_basemesh.children]:
        for mod in list(getattr(obj, "modifiers", [])):
            if mod.type == "SUBSURF":
                obj.modifiers.remove(mod)
    shrink_textures()

    # --- select the copy's hierarchy and export
    rig = copy_root if copy_root.type == "ARMATURE" else (new_basemesh.parent or copy_root)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [rig, *rig.children_recursive]:
        obj.select_set(True)
    make_opaque(rig.children_recursive)
    if PRESET.get("kurta"):
        tailor_kurta(rig.children_recursive, rig, PRESET["kurta"])
    inflate_clothes(rig.children_recursive)
    if PRESET.get("saree"):
        dress_saree(rig, rig.children_recursive, PRESET["saree"]["color"], PRESET["saree"]["border"])
    lobes = ear_lobes(rig, new_basemesh)
    eyes_name = next(name for _, name, atype in BODY_PARTS if atype == "Eyes")
    eyes = next((o for o in rig.children_recursive if f".{eyes_name}" in o.name), None)
    hair = [o for o in rig.children_recursive if hair_name in o.name]
    for h in hair:  # a predictable material name, so the browser can tint any hair style (LOOK.hair)
        for slot in h.material_slots:
            if slot.material and "hair" not in slot.material.name.lower():
                slot.material.name = f"hair.{slot.material.name}"
    if eyes:
        for h in hair:
            clear_face_and_ears(h, eyes, lobes if EARRINGS else {})
    if EARRINGS:
        add_earrings(rig, lobes)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [rig, *rig.children_recursive]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    print("exporting", [o.name for o in bpy.context.selected_objects], flush=True)
    OUT_GLB.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT_GLB), export_format="GLB", use_selection=True, export_yup=True,
        export_apply=True, export_morph=True, export_morph_normal=False, export_skins=True,
        export_def_bones=True, export_animations=False, export_image_format="JPEG", export_jpeg_quality=85,
    )
    print(f"EXPORTED {OUT_GLB} {OUT_GLB.stat().st_size / 1e6:.1f} MB", flush=True)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
