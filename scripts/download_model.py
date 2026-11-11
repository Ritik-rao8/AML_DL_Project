"""Download only the pinned DistilBERT inference/training files; verify model SHA-256."""
from pathlib import Path
import hashlib
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'

def main():
    cfg=json.loads((ROOT/'configs/phase2.json').read_text());out=ROOT/'outputs/phase2/pretrained';out.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','LICENSE','model.safetensors']:
        path=out/name
        if path.exists():
            if name!='model.safetensors':continue
            with path.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED:continue
        temp=path.with_suffix(path.suffix+'.part')
        subprocess.run(['curl','-fsSL','--retry','3',f'https://huggingface.co/{cfg["model"]}/resolve/{cfg["revision"]}/{name}','-o',str(temp)],check=True)
        if name=='model.safetensors':
            with temp.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=EXPECTED:raise ValueError('Model checksum mismatch')
        temp.replace(path)
    (out/'provenance.json').write_text(json.dumps({'model':cfg['model'],'revision':cfg['revision'],'model_sha256':EXPECTED},indent=2)+'\n')

if __name__=='__main__':main()
