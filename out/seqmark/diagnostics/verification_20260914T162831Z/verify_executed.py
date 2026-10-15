
import ast,contextlib,hashlib,io,json,os,socket,time,sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np,torch
from nearpy.hashes import RandomBinaryProjections
from sklearn.metrics import roc_curve,auc,accuracy_score,f1_score,precision_score,recall_score
sys.path.insert(0,str(Path.cwd()/'src'))
import utils,dataloader
from run_seqmark import load_encoder,atomic_json,source_identity
from transformers import AutoTokenizer
official_source="def get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int, hash_key: int):\n    n_bins = 2**lsh_dim\n    n_accept = int(n_bins * accept_rate)\n    rng.manual_seed(hash_key * seed)\n    vocab_permutation = torch.randperm(n_bins, device=device, generator=rng)\n    greenlist_ids = vocab_permutation[:n_accept]\n    return greenlist_ids.to(device)\n\ndef get_hash(sents, encoder, hasher, mean_emb=None):\n    embd = encoder.encode(sents) # size (num_sents, embd_dim)\n    if mean_emb is not None:\n        embd = [e-mean_emb for e in embd]\n    # print(f\"embedding: {embd}\")\n    hash_strs = [hasher.hash_vector(e)[0] for e in embd]\n    hash_ints = [int(s, 2) for s in hash_strs]\n    # return hash_ints, hash_strs\n    return hash_ints, embd\n\ndef detect_lsh(source, target, encoder, lmbd, lsh_dim, hash_key=None, cutoff=None, hasher=None, mean_embedding=None):\n    if cutoff == None:\n        cutoff = lsh_dim\n\n    n_watermark = 0\n    lsh_seed, _ = get_hash([source], encoder, hasher)\n    lsh_seed = lsh_seed[0]\n    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed, hash_key)\n    lsh_candidate, _ = get_hash([target[0]], encoder, hasher, mean_embedding)\n    lsh_candidate = lsh_candidate[0]\n    if lsh_candidate in accept_mask:\n        n_watermark += 1\n    watemark_details = {\n        \"n_watemarks\": [lsh_candidate in accept_mask],\n        \"accepted_regions\": [accept_mask.tolist()],\n        \"lsh_signature\": [lsh_candidate],\n    }\n    lsh_seed = lsh_candidate\n    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed, hash_key)\n    for i in range(1, len(target)):\n        lsh_candidate, _ = get_hash([target[i]])\n        lsh_candidate = lsh_candidate[0]\n        if lsh_candidate in accept_mask:\n            n_watermark += 1\n        watemark_details[\"n_watemarks\"].append(lsh_candidate in accept_mask)\n        watemark_details[\"accepted_regions\"].append(accept_mask.tolist())\n        watemark_details[\"lsh_signature\"].append(lsh_candidate)\n        lsh_seed = lsh_candidate\n        accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed, hash_key)\n    n_test_sent = len(target)  # exclude the prompt and the ending\n    num = n_watermark - lmbd * (n_test_sent)\n    denom = np.sqrt((n_test_sent) * lmbd * (1 - lmbd))\n    # print(f\"n_watermark: {n_watermark}, n_test_sent: {n_test_sent}\")\n    zscore = num / denom if denom > 0 else None\n    # print(f\"zscore: {zscore}\", type(zscore))\n    watemark_details['z_score'] = zscore\n    return zscore, n_watermark, watemark_details"
class_source="def compute_classification_metrics(\n    pos_scores: list, \n    neg_scores: list, \n    threshold: float = None):\n\n    labels = [1] * len(pos_scores) + [0] * len(neg_scores)\n    scores = pos_scores + neg_scores \n    \n    if not threshold:\n\n        fpr, tpr, thresholds = roc_curve(labels, scores)\n        print(f\"fpr={fpr}\")\n        print(f\"tpr={tpr}\")\n        print(f\"thresholds={thresholds}\")\n\n        # find the threshold that maximize TPR / FPR \n        max_ratio, max_tpr, max_fpr = 0.0, 0.0, 0.0\n        for i in range(len(thresholds)):\n            if fpr[i] == 0:\n                continue\n            ratio = tpr[i] / fpr[i]\n            if ratio > max_ratio:\n                max_ratio = ratio\n                max_tpr = tpr[i]\n                max_fpr = fpr[i]\n                threshold = thresholds[i]\n \n    preds = [1 if score >= threshold else 0 for score in scores]\n\n    accuracy = accuracy_score(labels, preds)\n    f1 = f1_score(labels, preds)\n    precision = precision_score(labels, preds)\n    recall = recall_score(labels, preds)\n\n    return {\n        \"threshold\": threshold,\n        \"fpr\": max_fpr,\n        \"tpr\": max_tpr,\n        \"tpr/fpr\": max_ratio, \n        \"accuracy\": accuracy,\n        \"f1\": f1,\n        \"precision\": precision,\n        \"recall\": recall,\n    }"
ns={'np':np,'torch':torch,'device':'cuda','rng':torch.Generator('cuda')}
exec(compile(official_source,'upstream_correct_watermark.py','exec'),ns)
cn={'roc_curve':roc_curve,'accuracy_score':accuracy_score,'f1_score':f1_score,'precision_score':precision_score,'recall_score':recall_score}
exec(compile(class_source,'upstream_classification.py','exec'),cn)
impl=os.environ['SEQMARK_VERIFY_IMPL']
source=Path.cwd()
report_path=Path("/projects/standard/jusun/shared/wu001810/llm_watermark/ast_watermarking/out/seqmark/diagnostics/verification_20260914T162831Z")/('verify_'+impl+'.json')
sample_meta=json.loads((utils.OUT_DIR/'seqmark/humaneval'/f'seqmark_humaneval_{impl}_qwen3.5-4b.json').read_text())['meta']
args=SimpleNamespace(embedder=sample_meta['watermark']['embedder'],embedder_path=sample_meta['embedder_path'],embedder_device='cuda')
report={'kind':'operational_verification','status':'running','started_at':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'gpu':torch.cuda.get_device_name(),'encoder':args.embedder,'encoder_path':args.embedder_path,'source':source_identity(),'upstream_revision':'3c1ae8f9988e025e3d564839817a15a85bdcf0ff','upstream_detector_sha256':"f96a036fea30c337242716c2dc1e39b2c7fbdc2fb3988c4bfa3b1ad39464e374",'scope':'Recompute existing outputs; compare independent and cached centres; no new LLM generations.','datasets':{}}
atomic_json(report_path,report)
try:
 encoder=load_encoder(args);tokenizer=AutoTokenizer.from_pretrained(sample_meta['model_path'],local_files_only=True)
 hasher=RandomBinaryProjections('rbp_perm',projection_count=3,rand_seed=12345);hasher.reset(encoder.get_sentence_embedding_dimension())
 class CachedEncoder:
  def __init__(self):self.cache={}
  def encode(self,texts):
   out=[]
   for text in texts:
    if text not in self.cache:self.cache[text]=encoder.encode([text],batch_size=32,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=False)[0]
    out.append(self.cache[text])
   return np.asarray(out)
 ce=CachedEncoder()
 for name in ('humaneval','mbpp'):
  path=utils.OUT_DIR/'seqmark'/name/f'seqmark_{name}_{impl}_qwen3.5-4b.json'
  data=json.loads(path.read_text());ds=dataloader.get(name);tasks={ds.task_id(t):t for t in ds.load_tasks()}
  rows=[];mismatches=[];scores={v:{b:[] for b in ('watermarked','unwatermarked')} for v in ('recorded','independent_center','generation_center_float32','generation_center_cached_float64')}
  for i,row in enumerate(data['results']):
   prompt=utils.task_chat_text(tokenizer,ds,tasks[row['task_id']])
   audit=json.loads((source/row['audit_file']).read_text())
   centers={'independent_center':np.asarray(audit['detection_center'],dtype=np.float32),'generation_center_float32':np.asarray(audit['generation_center'],dtype=np.float32),'generation_center_cached_float64':np.asarray(audit['generation_center'],dtype=np.float64)}
   evidence={'task_id':row['task_id'],'generation_accepted':audit['sampling']['accepted'],'branches':{}}
   ce.cache={}
   for branch in ('watermarked','unwatermarked'):
    program=utils.extract_program(row[branch]['completion']);assert program
    stored=row[branch]['detection'];scores['recorded'][branch].append(stored['z_score'])
    evidence['branches'][branch]={}
    for variant,center in centers.items():
     z,n,details=ns['detect_lsh'](prompt,[program],ce,.25,3,hash_key=12345,hasher=hasher,mean_embedding=center)
     scores[variant][branch].append(float(z))
     evidence['branches'][branch][variant]={'z_score':float(z),'green':bool(n),'signature':details['lsh_signature'][0],'accepted_regions':details['accepted_regions'][0]}
     if variant=='independent_center':
      if not (np.isclose(z,stored['z_score'],atol=1e-12,rtol=0) and details['lsh_signature'][0]==stored['lsh_signature'] and sorted(details['accepted_regions'][0])==stored['accepted_regions']):
       mismatches.append({'task_id':row['task_id'],'branch':branch,'stored':stored,'recomputed':evidence['branches'][branch][variant]})
   rows.append(evidence)
   if (i+1)%50==0:print('VERIFY',impl,name,i+1,'/',len(data['results']),'mismatches',len(mismatches),flush=True)
  summaries={}
  for variant,v in scores.items():
   with contextlib.redirect_stdout(io.StringIO()):c=cn['compute_classification_metrics'](v['watermarked'],v['unwatermarked'])
   y=[1]*len(v['watermarked'])+[0]*len(v['unwatermarked']);fpr,tpr,threshold=roc_curve(y,v['watermarked']+v['unwatermarked'],drop_intermediate=False)
   summaries[variant]={'n':len(data['results']),'classification':{k:float(x) for k,x in c.items()},'auroc':float(auc(fpr,tpr)),'tpr_at_1pct':float(max(tpr[fpr<=.01])),'tpr_at_5pct':float(max(tpr[fpr<=.05]))}
  report['datasets'][name]={'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mismatches':mismatches,'summaries':summaries,'per_task':rows}
  atomic_json(report_path,report)
  print('SUMMARY',impl,name,json.dumps(summaries),flush=True)
 report['status']='completed'
except BaseException as exc:
 report['status']='failed';report['error']=repr(exc);raise
finally:
 report['ended_at']=datetime.now(timezone.utc).isoformat();atomic_json(report_path,report)
print('VERIFICATION_FINISHED',report_path,flush=True)

