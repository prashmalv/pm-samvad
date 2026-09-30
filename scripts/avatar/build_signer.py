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
        "hair": "faydaen_hair_1",   # long, open and wavy (tied back, e.g. "ponytail01", keeps the signing space clearer)
        "earrings": True,           # small gold hoops, attached to the head bone
        "eyebrows": "eyebrow001",
        "eyelashes": "eyelashes01",
        "clothes": ["toigo_basic_tucked_t-shirt", "toigo_wool_pants", "shoes02"],  # plain top, as signers wear
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
            "race": {"asian": 0.3, "caucasian": 0.45, "african": 0.25},
        },
        "skin": "young_asian_male",  # no Indian male skin in the CC0 packs; the browser warms the tone (LOOK)
        "hair": "short01",           # short and neat, keeps the face and signing space clear
        "earrings": False,
        "eyebrows": "eyebrow002",
        "eyelashes": "eyelashes02",
        "clothes": ["namuhekam_male_polo_shirt", "toigo_wool_pants", "shoes02"],
    },
}
PRESET = PRESETS[os.getenv("SIGNER_PRESET", "woman")]
OUT_GLB = Path(os.getenv("SIGNER_OUT", ROOT / "static" / "models" / f"{PRESET['file']}.glb"))  # SIGNER_OUT: try a variant elsewhere
OUT_BLEND = ROOT / "data" / "avatar" / f"{PRESET['file']}.blend"   # editable source, open it in Blender to tweak by hand

MACRO = PRESET["macro"]
SKIN = PRESET["skin"]
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


SEE_THROUGH = ("eyebrow", "eyelash", "ponytail", "hair")  # textures that need their alpha cut-outs


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
    inflate_clothes(rig.children_recursive)
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
