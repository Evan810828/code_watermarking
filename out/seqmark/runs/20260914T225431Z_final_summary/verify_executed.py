from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, socket, sys
sys.path.insert(0, str(Path('src').resolve()))
import run_seqmark
ledger=Path(__file__).resolve().parent
summary_manifest={"run_id":ledger.name,"study":"baseline_watermarking_seqmark","stage":"final_verification_and_summary","status":"running","started_at":run_seqmark.now(),"ended_at":None,"code":run_seqmark.source_identity(),"verification_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"command":sys.argv,"environment":{"host":socket.gethostname(),"python":platform.python_version()},"selection":"out/seqmark/sweep/selected_config.json","datasets":["humaneval","mbpp"],"encoders":["google/embeddinggemma-300m","AbeHou/SemStamp-c4-sbert"],"seed":42}
summary_manifest["prior_verification"]="/projects/standard/jusun/shared/wu001810/llm_watermark/ast_watermarking/out/seqmark/runs/20260914T225004Z_final_summary/manifest.json"
run_seqmark.atomic_json(ledger/"manifest.json",summary_manifest)
try:
    from pathlib import Path
    import hashlib, json, math, sys
    sys.path.insert(0, str(Path('src').resolve()))
    import dataloader, utils, run_seqmark
    from wm.seqmark import stream_seed
    root=Path.cwd(); base=root/'out/seqmark'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read=lambda p:json.loads(Path(p).read_text())
    source='952a90f4b7f79f76f05f07fb62aeea349b2008ba724fec0e312c683aacb3362d'
    evaluator='1da51e14d30eb34f869936159cd0993a2605241317745f6d0cb5dc77fb03368e'
    assert run_seqmark.source_identity()['source_fingerprint']==source
    assert sha(root/'src/eval_pass1.py')==evaluator
    selection_path=base/'sweep/selected_config.json'; selection=read(selection_path)
    assert selection['selected']==max(selection['all_settings'],key=lambda m:(m['mean_f1'],m['mean_auroc'],-m['mean_trials'],-m['lsh_dim'],-m['gamma']))
    train=read(base/'sweep/mbpp_train100.json'); test=dataloader.get('mbpp').load_tasks()
    assert len(train)==len({t['task_id'] for t in train})==100
    assert not {int(t['task_id']) for t in train}&{int(t['task_id']) for t in test}
    assert not {t['text'] for t in train}&{t['text'] for t in test}
    sweep_rows=[]
    for s in selection['sources']:
        assert sha(s['path'])==s['sha256']
        data=read(s['path']); assert read(s['manifest'])['status']=='completed'
        assert data['meta']['source_fingerprint']==source
        sweep_rows.extend(data['results'])
    assert sorted(r['task_id'] for r in sweep_rows)==sorted(t['task_id'] for t in train)
    for i, setting in enumerate(selection['all_settings']):
        for encoder in utils.SEQMARK_EMBEDDERS:
            values=[r['encoders'][encoder][i] for r in sweep_rows]
            actual=run_seqmark.paper_classification([v['watermarked_z'] for v in values],[v['unwatermarked_z'] for v in values])
            for k,v in actual.items():assert math.isclose(v,setting['encoders'][encoder][k],rel_tol=0,abs_tol=1e-12),(i,encoder,k)
    all_data={}; all_audits={}; all_verdicts={}; outputs=[]; configs={}; total=0; mismatch=0
    for encoder in utils.SEQMARK_EMBEDDERS:
        for name in utils.SEQMARK_DATASETS:
            impl=utils.seqmark_impl_tag(encoder,'generation_cache')
            path=base/name/f'seqmark_{name}_{impl}_qwen3.5-4b.json'
            data=read(path); meta=data['meta']; rows=data['results']
            ds=dataloader.get(name); tasks=ds.load_tasks(); expected=[ds.task_id(t) for t in tasks]
            assert [r['task_id'] for r in rows]==meta['task_ids']==expected
            assert meta['status']=='completed' and meta['source_fingerprint']==source
            assert meta['data_fingerprint']==run_seqmark.digest(tasks)
            assert meta['watermark']['embedder']==encoder
            for k,v in {'lsh_dim':5,'gamma':.1,'center_samples':50,'center_temperature':1.2,'max_trials':50,'candidate_batch_size':1,'hash_key':12345,'detector_center':'generation_cache'}.items():
                assert meta['watermark'][k]==v,(encoder,name,k)
            assert {k:meta['generation'][k] for k in selection['contract']['generation']}==selection['contract']['generation']
            assert meta['generation']['dtype']=='bfloat16' and meta['generation']['do_sample'] is True
            assert meta['parameter_selection']['sha256']==sha(selection_path)
            assert meta['embedder_pipeline']==run_seqmark.encoder_pipeline(encoder)
            assert meta['embedder_max_seq_length']==(2048 if encoder==utils.SEQMARK_EMBEDDERS[0] else 512)
            verdict_path=path.parent/f'pass1_seqmark_{impl}_qwen3.5-4b_{name}.json'
            verdicts=read(verdict_path); provenance=read(verdict_path.with_suffix('.provenance.json'))
            assert provenance['input_sha256']==sha(path) and provenance['verdict_sha256']==sha(verdict_path)
            assert [v['task_id'] for v in verdicts]==expected
            for shard_path in meta['source_shards']:
                shard=read(shard_path); manifest=read(shard['meta']['manifest'])
                assert shard['meta']['status']==manifest['status']=='completed'
                assert manifest['code']['source_fingerprint']==source
                assert manifest['ended_at'] is not None
            audits={}
            for row in rows:
                ap=Path(row['audit_file']); audit=read(ap)
                if ap.parent not in configs:configs[ap.parent]=read(ap.parent/'config.json')
                contract=configs[ap.parent]
                assert audit['contract_sha256']==run_seqmark.digest(contract)
                assert contract['source_fingerprint']==source and contract['watermark']==meta['watermark']
                assert audit['result']==row
                assert len(audit['generation_center_samples'])==50
                assert audit['detection_center_samples']==[]
                assert len(audit['generation_center'])==768
                assert audit['generation_center']==audit['detection_center']
                assert audit['seeds']=={stage:stream_seed(42,name,row['task_id'],stage) for stage in ('generation_center','detection_center','baseline')}
                sampling=audit['sampling']; candidates=sampling['candidate_audit']
                assert sampling['trials']==sampling['generated_candidates']==len(candidates)
                assert 1<=sampling['trials']<=50
                assert [c['trial'] for c in candidates]==list(range(1,len(candidates)+1))
                assert not any(c['accepted'] for c in candidates[:-1])
                assert candidates[-1]['accepted']==sampling['accepted']
                assert sampling['fallback']==(not sampling['accepted'])
                if sampling['fallback']:assert sampling['trials']==50
                assert candidates[-1]['completion_sha256']==hashlib.sha256(row['watermarked']['completion'].encode()).hexdigest()
                for branch in ('watermarked','unwatermarked'):
                    record=row[branch]; det=record['detection']
                    assert record['extracted_code']==utils.extract_code(record['completion'])
                    if det is not None:
                        assert len(det['accepted_regions'])==3
                        green=det['lsh_signature'] in det['accepted_regions']
                        assert green==det['is_watermarked']==bool(det['num_green_segments'])
                        assert math.isclose(det['z_score'],(int(green)-.1)/math.sqrt(.09),rel_tol=0,abs_tol=1e-12)
                    if branch=='watermarked' and det is not None:
                        mismatch+=int(det['is_watermarked']!=sampling['accepted'])
                audits[row['task_id']]=audit; total+=1
            pos=[r['watermarked']['detection']['z_score'] for r in rows if r['watermarked']['detection'] is not None]
            neg=[r['unwatermarked']['detection']['z_score'] for r in rows if r['unwatermarked']['detection'] is not None]
            threshold=meta['parameter_selection']['threshold']
            fixed=run_seqmark.paper_classification(pos,neg,threshold)
            posthoc=run_seqmark.paper_classification(pos,neg)
            assert fixed==posthoc
            summary=utils.detection_summary(rows)
            assert all((summary[k]==v if not isinstance(v,float) else math.isclose(summary[k],v,rel_tol=0,abs_tol=1e-12)) for k,v in meta['summary'].items())
            quality={b:sum(v[b]['passed'] for v in verdicts) for b in ('watermarked','unwatermarked')}
            all_data[encoder,name]=data; all_audits[encoder,name]=audits; all_verdicts[encoder,name]=verdicts
            outputs.append({'encoder':encoder,'dataset':name,'input':str(path),'input_sha256':sha(path),'verdict_provenance':str(verdict_path.with_suffix('.provenance.json')),'n':len(rows),'summary':summary,'classification':fixed,'true_positives':sum(z>=threshold for z in pos),'false_positives':sum(z>=threshold for z in neg),'pass_counts':quality})
    paired=0
    for name in utils.SEQMARK_DATASETS:
        gemma,sbert=utils.SEQMARK_EMBEDDERS
        gdata=all_data[gemma,name]; sdata=all_data[sbert,name]
        assert sdata['meta']['reference_draws']['sha256']==sha(sdata['meta']['reference_draws']['output'])
        for grow,srow in zip(gdata['results'],sdata['results']):
            tid=grow['task_id']; ga=all_audits[gemma,name][tid]; sa=all_audits[sbert,name][tid]
            assert grow['task_id']==srow['task_id'] and grow['prompt_text']==srow['prompt_text']
            assert ga['generation_center_samples']==sa['generation_center_samples']
            assert ga['detection_center_samples']==sa['detection_center_samples']
            assert grow['unwatermarked']['completion']==srow['unwatermarked']['completion']
            assert grow['unwatermarked']['num_gen_tokens']==srow['unwatermarked']['num_gen_tokens']
            assert sa['reference_draws']['sha256']==sha(sa['reference_draws']['audit_file'])
            paired+=1
        for gv,sv in zip(all_verdicts[gemma,name],all_verdicts[sbert,name]):
            assert gv['unwatermarked']==sv['unwatermarked'],(name,gv['task_id'],gv['unwatermarked'],sv['unwatermarked'])
    text=(root/'out/README.md').read_text(); start=text.index('| method | impl | dataset | n | scored |',text.index('## Results')); end=text.index('\n\n',start)
    table=text[start:end].splitlines(); original=[line for line in table if not line.startswith('| SeqMark |')]
    assert hashlib.sha256('\n'.join(original).encode()).hexdigest()=='fad0320da07b5d8d3a5bc1a963c5547dc041652aa83a7d952f2f1248844ff135'
    assert len([l for l in table if l.startswith('| SeqMark |')])==4
    assert not any('livecodebench' in l for l in table if l.startswith('| SeqMark |'))
    assert total==1328 and paired==664
    print(json.dumps({'checks':'passed','audited_task_results':total,'paired_tasks':paired,'sweep_tasks':len(train),'sweep_configurations':16,'selection_sha256':sha(selection_path),'generation_detection_acceptance_differences':mismatch,'source_fingerprint':source,'evaluator_sha256':evaluator,'results':outputs},indent=2))
    registered={}
    for result in outputs:
        key=result["encoder"]+"/"+result["dataset"]
        registered[key]={**result["summary"],**result["classification"],"true_positives":result["true_positives"],"false_positives":result["false_positives"]}
        registered[key].update({f"pass@1_{branch}":count/result["n"] for branch,count in result["pass_counts"].items()})
    run_seqmark.atomic_json(ledger/"metrics.json",registered)
    run_seqmark.atomic_json(ledger/"diagnostics.json",{"audited_task_results":total,"paired_tasks":paired,"sweep_tasks":len(train),"sweep_configurations":16,"generation_detection_acceptance_differences":mismatch,"original_non_seqmark_rows_preserved":True,"sweep_test_overlap":0,"all_assertions_passed":True})
    summary_manifest.update(status="completed",source_results=[{k:r[k] for k in ("encoder","dataset","input","input_sha256","verdict_provenance")} for r in outputs],selection_sha256=sha(selection_path),evaluator_sha256=evaluator)
except BaseException as error:
    summary_manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
    raise
finally:
    summary_manifest["ended_at"]=run_seqmark.now()
    run_seqmark.atomic_json(ledger/"manifest.json",summary_manifest)
