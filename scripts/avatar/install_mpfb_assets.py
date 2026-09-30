"""Run inside Blender: install the MPFB asset packs from Downloads/avatar into MPFB's user data folder.

    tools/blender-5.2.2-windows-x64/blender.exe --background --python scripts/avatar/install_mpfb_assets.py
"""
import sys
from pathlib import Path

from bl_ext.user_default.mpfb.services.assetservice import AssetService
from bl_ext.user_default.mpfb.services.locationservice import LocationService

ROOT = Path(__file__).resolve().parents[2]
PACKS = ["makehuman_system_assets_cc0.zip", "skins01_cc0.zip", "skins02_cc0.zip", "hair01_cc0.zip",
         "shirts01_cc0.zip", "pants01_cc0.zip", "faceunits01.zip"]

target = LocationService.get_user_data()
for name in PACKS:
    zip_path = ROOT / "Downloads" / "avatar" / name
    if not zip_path.exists():
        print(f"MISSING {name}")
        continue
    err = AssetService.fix_and_extract_asset_pack_zip(str(zip_path), target)
    print(f"{'FAILED' if err else 'installed'} {name} {err or ''}", flush=True)
AssetService.update_all_asset_lists() if hasattr(AssetService, "update_all_asset_lists") else None
for sub in ("skins", "hair", "clothes", "eyes", "eyebrows", "eyelashes", "teeth", "tongue"):
    items = AssetService.list_mhclo_assets(sub) if sub != "skins" else AssetService.list_mhmat_assets(sub)
    print(f"ASSETS {sub}: " + " | ".join(Path(p).stem for p in items))
sys.exit(0)
