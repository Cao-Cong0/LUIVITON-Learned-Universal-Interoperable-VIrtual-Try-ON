
import os
import json
import glob

GALLERY_DIR = "assets/Gallery"
OUTPUT_FILE = "assets/gallery.json"

def generate_manifest():
    manifest = []
    
    # Get all subdirectories in Gallery folder
    root_path = os.path.abspath(GALLERY_DIR)
    if not os.path.exists(root_path):
        print(f"Error: Directory {root_path} does not exist")
        return

    subdirs = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    
    for folder_name in subdirs:
        folder_path = os.path.join(root_path, folder_name)
        
        # Find all .obj files in this folder
        obj_files = sorted(glob.glob(os.path.join(folder_path, "*.obj")))
        
        objs_list = []
        for obj_path in obj_files:
            filename = os.path.basename(obj_path)
            mtl_filename = filename.replace(".obj", ".mtl")
            
            # Check if corresponding mtl exists
            mtl_path = os.path.join(folder_path, mtl_filename)
            has_mtl = os.path.exists(mtl_path)
            
            objs_list.append({
                "obj": filename,
                "mtl": mtl_filename if has_mtl else None
            })
            
        if objs_list:
            manifest.append({
                "folder": folder_name,
                "label": folder_name, # Start with folder name as label
                "models": objs_list
            })
            print(f"Found {len(objs_list)} models for {folder_name}")
        else:
            print(f"Warning: No OBJs found in {folder_name}")

    # Write to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated {OUTPUT_FILE} with {len(manifest)} entries.")

if __name__ == "__main__":
    generate_manifest()
