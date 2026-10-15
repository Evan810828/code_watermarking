import hashlib,json,os,sys,socket
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
sys.path.insert(0,str(Path.cwd()/'src'))
import utils
from run_seqmark import load_encoder,atomic_json
from wm.seqmark import SeqMark,SeqMarkConfig
root=Path.cwd()
data=json.loads((root/'out/seqmark/mbpp/seqmark_mbpp_port_sbert_qwen3.5-4b.json').read_text())
meta=data['meta'];wm=SeqMark(load_encoder(SimpleNamespace(embedder=meta['watermark']['embedder'],embedder_path=meta['embedder_path'],embedder_device='cuda')),SeqMarkConfig(embedder=meta['watermark']['embedder']))
report={'kind':'operational_numerical_diagnostic','host':socket.gethostname(),'gpu':torch.cuda.get_device_name(),'scope':'Saved SBERT/MBPP tasks 176 and 242; no language-model generation','tasks':[]}
for row in data['results']:
 if row['task_id'] not in (176,242):continue
 a=json.loads((root/row['audit_file']).read_text());program=utils.extract_program(row['watermarked']['completion']);center=np.asarray(a['generation_center'],dtype=np.float32)
 programs=[utils.extract_program(x['completion']) or '' for x in a['generation_center_samples']]
 recomputed,_=wm.estimate_center(a['generation_center_samples'],utils.extract_program)
 out={'task_id':row['task_id'],'generation_center_unique_programs':len(set(programs)),'calibration_equal_to_selected':sum(x==program for x in programs),'center_recompute_max_abs_error':float(np.max(np.abs(recomputed-center))),'recorded_generation_signature':a['sampling']['candidate_audit'][-1]['signature'],'recorded_generation_accepted':a['sampling']['accepted'],'accepted_regions':row['watermarked']['detection']['accepted_regions'],'batch_checks':[]}
 single=None
 for batch in (1,2,8,20):
  e=wm.embed([program]*batch)[0]
  if single is None:single=e
  residual=e-center;projections=wm.normals.dot(residual);sig=wm.signatures([e],center)[0]
  out['batch_checks'].append({'repeated_program_batch_size':batch,'signature':sig,'green':sig in out['accepted_regions'],'embedding_max_abs_diff_from_single':float(np.max(np.abs(e-single))),'centered_norm':float(np.linalg.norm(residual)),'embedding_norm':float(np.linalg.norm(e)),'projected_residual':projections.tolist()})
 report['tasks'].append(out)
atomic_json(root/'out/seqmark/diagnostics/verification_20260914T162831Z/verify_boundary.json',report)
print(json.dumps(report),flush=True)

