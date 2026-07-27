import os, glob, subprocess
from pathlib import Path
import numpy as np
import SimpleITK as sitk

INPUT_DIR = Path(os.environ.get("PENGWIN_INPUT", "/input/images/peripelvic-fracture-ct"))
OUTPUT_DIR = Path(os.environ.get("PENGWIN_OUTPUT", "/output/images/peripelvic-fracture-ct-segmentation"))
MODEL_DIR = os.environ.get("PENGWIN_MODEL", "/opt/ml/model")
BONES = [("sacrum",1,0),("leftHip",2,50),("rightHip",3,100),("femur",4,150)]
MIN_MM3 = 500.0; CAP = 50
WORK = Path(os.environ.get("PENGWIN_WORK", "/tmp/work"))
FRAC = os.environ.get("PENGWIN_FRAC", "/opt/app/frac_to_instance.py")

def _sh(cmd): print(">>"," ".join(cmd),flush=True); subprocess.run(cmd,check=True)
def _find_input():
    f=sorted(glob.glob(str(INPUT_DIR/"*.mha"))+glob.glob(str(INPUT_DIR/"*.tif*")))
    if not f: raise RuntimeError(f"pas d'entrée dans {INPUT_DIR}")
    return f[0]

def run():
    os.environ["nnUNet_results"]=MODEL_DIR
    os.environ["nnUNet_raw"]=str(WORK/"raw"); os.environ["nnUNet_preprocessed"]=str(WORK/"pp")
    os.environ.setdefault("nnUNet_n_proc_DA","2")
    for s in ["raw","pp","anat_in","anat_out","csm_in","csm_out","inst"]: (WORK/s).mkdir(parents=True,exist_ok=True)
    in_path=_find_input(); print("Input:",in_path,flush=True)
    ct_img=sitk.ReadImage(in_path); ct_arr=sitk.GetArrayFromImage(ct_img); uid=Path(in_path).name.split(".")[0]
    sitk.WriteImage(ct_img,str(WORK/"anat_in"/"case_0000.nii.gz"),useCompression=True)
    _sh(["nnUNetv2_predict","-i",str(WORK/"anat_in"),"-o",str(WORK/"anat_out"),
         "-d","601","-c","3d_fullres","-p","nnUNetPlans","-tr","nnUNetTrainer_250epochs","-f","0","-npp","1","-nps","1"])
    anat_arr=sitk.GetArrayFromImage(sitk.ReadImage(str(WORK/"anat_out"/"case.nii.gz")))
    present=[]
    for name,cls,off in BONES:
        m=anat_arr==cls
        if not m.any(): print(f"{name}: absent",flush=True); continue
        mi=sitk.GetImageFromArray(np.where(m,ct_arr,0).astype(ct_arr.dtype)); mi.CopyInformation(ct_img)
        sitk.WriteImage(mi,str(WORK/"csm_in"/f"{name}_0000.nii.gz"),useCompression=True); present.append((name,cls,off))
    fused=np.zeros(ct_arr.shape,np.uint16)
    if present:
        _sh(["nnUNetv2_predict","-i",str(WORK/"csm_in"),"-o",str(WORK/"csm_out"),
             "-d","002","-c","3d_fullres","-p","nnUNetResEncUNetMPlans","-tr","nnUNetTrainer","-f","0","1","2","3","4","-npp","1","-nps","1"])
        _sh(["python",FRAC,"-i",str(WORK/"csm_out"),"-o",str(WORK/"inst"),"-k","5","-c","100","--device","cuda"])
        sx,sy,sz=ct_img.GetSpacing(); vox=sx*sy*sz
        for name,cls,off in present:
            f=WORK/"inst"/f"{name}.nii.gz"
            if not f.exists(): continue
            arr=sitk.GetArrayFromImage(sitk.ReadImage(str(f))); bone=anat_arr==cls; nid=1
            for fid in [int(x) for x in np.unique(arr) if x!=0]:
                frag=(arr==fid)&bone
                if frag.sum()*vox < MIN_MM3: continue
                if nid>CAP: break
                fused[frag]=off+nid; nid+=1
            print(f"{name}: {nid-1} fragments",flush=True)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    out=sitk.GetImageFromArray(fused); out.CopyInformation(ct_img)
    sitk.WriteImage(out,str(OUTPUT_DIR/f"{uid}.mha"),useCompression=True)
    u=np.unique(fused); print(f"Output OK | {len(u)-1} fragments | {u[u>0].tolist()[:15]}",flush=True)
    return 0

if __name__=="__main__": raise SystemExit(run())
