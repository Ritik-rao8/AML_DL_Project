"""Phase 2 CLI with explicit device override and safe reuse of completed runs."""
import argparse
import json
import sys
import fcntl
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
import torch
from src import dl, phase2

if __name__ == '__main__':
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    options.add_argument('--force',action='store_true')
    args,remaining=options.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    if args.device!='auto':
        dl.choose_device=lambda:torch.device(args.device)
        phase2.choose_device=dl.choose_device
    if remaining and remaining[0]=='train':
        experiment=remaining[remaining.index('--experiment')+1] if '--experiment' in remaining else 'main'
        if experiment not in ['main','frozen','unweighted']:phase2.main()
        directory=phase2.OUT/experiment;directory.mkdir(parents=True,exist_ok=True)
        with (directory/'training.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            marker=directory/'COMPLETE.json'
            if marker.exists() and json.loads(marker.read_text())['signature']==phase2.signature() and not args.force:
                print(experiment,'already complete; use --force for an explicit retrain')
            else:phase2.main()
    else:phase2.main()
