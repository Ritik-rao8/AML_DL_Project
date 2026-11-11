"""Small Phase 2 workflow: prepare, pilot, train, evaluate, demo."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from src.config import PROJECT_ROOT
from src.relations import system_id, parse_claim, resolve_cycles
from src.dl import DocumentClassifier, seed_all, choose_device, encode_rows, collate, fit, predict

ROOT=Path(PROJECT_ROOT); OUT=ROOT/'outputs/phase2'; PRETRAINED=OUT/'pretrained'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def config():
    return json.loads((ROOT/'configs/phase2.json').read_text())


def signature():
    paths=[ROOT/'configs/phase2.json',ROOT/'src/dl.py',ROOT/'src/phase2.py',ROOT/'src/graph_review.py',
           ROOT/'src/relations.py',ROOT/'corr2cause/download_manifest.json',ROOT/'requirements-phase2.txt',ROOT/'data/graph_reviews/cpd_assumptions.json']
    from importlib.metadata import version
    versions=json.dumps({n:version(n) for n in ['torch','transformers','numpy','pandas','scikit-learn']},sort_keys=True).encode()
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+versions).hexdigest()


def load_data():
    pairs={s:pd.read_pickle(ROOT/f'outputs/phase1/cache/{s}.pkl') for s in ['train','dev','test']}
    return pairs


def select_threshold(y,scores):
    candidates=np.linspace(.05,.95,19)
    threshold=max(candidates,key=lambda t:(f1_score(y,np.asarray(scores)>=t,average='macro'),t))
    return float(threshold),float(f1_score(y,np.asarray(scores)>=threshold,average='macro'))


def prepare():
    cfg=config(); pairs=load_data();train=pairs['train'][0]
    # Whole premise groups, selected without looking at validation/test performance.
    groups=list(train.groupby('premise',sort=True).indices.values())
    rng=np.random.default_rng(cfg['seed']);rng.shuffle(groups)
    positions=[]
    for group in groups:
        positions.extend(group.tolist())
        if len(positions)>=cfg['training_budget_rows']:break
    positions=sorted(positions)
    excluded=set(train.premise)|set(pairs['test'][0].premise)
    mask=~pairs['dev'][0].premise.isin(excluded)
    frames={'train':train.iloc[positions].reset_index(drop=True),
            'dev':pairs['dev'][0][mask].reset_index(drop=True),'test':pairs['test'][0]}
    tokenizer=AutoTokenizer.from_pretrained(PRETRAINED,local_files_only=True)
    data={}
    manifest={'scope':'Local compute-budget, whole-premise subset; matched ML trained on the same rows',
              'seed':cfg['seed'],'source_revision':cfg['revision'],'splits':{}}
    for split,frame in frames.items():
        rows=encode_rows(frame,tokenizer);data[split]=rows
        manifest['splits'][split]={'rows':len(rows),'positive':int(frame.label.sum()),
            'systems':int(frame.premise.nunique()),'docIds':frame.docId.tolist(),
            'max_windows':max(len(r['windows']) for r in rows),
            'multi_window_rows':sum(len(r['windows'])>1 for r in rows),
            'max_pair_tokens':max(r['premise_tokens']+r['hypothesis_tokens']+3 for r in rows)}
        print(split,{k:v for k,v in manifest['splits'][split].items() if k!='docIds'},flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'prepared.pkl').open('wb') as f:pickle.dump({'rows':data,'frames':frames,'positions':positions,'mask':mask},f)
    manifest['preparation_signature']=signature()
    write(OUT/'data_manifest.json',manifest)
    # Matched classical model: same train examples; calibration only within training.
    columns=[c for c in pairs['train'][1] if c not in {'label','docId','template'}]
    ml=CalibratedClassifierCV(RandomForestClassifier(n_estimators=100,max_depth=10,class_weight='balanced',random_state=cfg['seed'],n_jobs=2),cv=3)
    ml.fit(pairs['train'][1].iloc[positions][columns],frames['train'].label)
    scores={s:np.round(ml.predict_proba(pairs[s][1][mask][columns] if s=='dev' else pairs[s][1][columns])[:,1],12) for s in ['dev','test']}
    fullmodel=pickle.load((ROOT/'outputs/phase1/models/graph_model.pkl').open('rb'))
    fullscores={s:np.round(fullmodel['model'].predict_proba(pairs[s][1][mask][fullmodel['features']] if s=='dev' else pairs[s][1][fullmodel['features']])[:,1],12) for s in ['dev','test']}
    np.savez(OUT/'ml_scores.npz',**scores,full_dev=fullscores['dev'],full_test=fullscores['test'])
    with (OUT/'matched_ml.pkl').open('wb') as f:pickle.dump({'model':ml,'features':columns},f)


def prepared():
    if json.loads((OUT/'data_manifest.json').read_text())['preparation_signature']!=signature():
        raise ValueError('Code/config/data changed. Run prepare and retrain instead of reusing stale results.')
    with (OUT/'prepared.pkl').open('rb') as f:return pickle.load(f)


def pilot():
    from torch.utils.data import DataLoader
    cfg=config();data=prepared()['rows'];seed_all(cfg['seed']);device=choose_device()
    model=DocumentClassifier(PRETRAINED).to(device);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    batch=next(iter(DataLoader(data['train'][:8],batch_size=cfg['batch_size'],collate_fn=collate)))
    batch={k:v.to(device) for k,v in batch.items()}
    # Warm up MPS compilation before timing steady training steps.
    for _ in range(3):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(**batch),batch['labels']);loss.backward();optimizer.step()
    if device.type=='mps':torch.mps.synchronize()
    start=time.time()
    for _ in range(20):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(**batch),batch['labels']);loss.backward();optimizer.step()
    if device.type=='mps':torch.mps.synchronize()
    seconds=(time.time()-start)/20
    document_seconds=seconds/cfg['batch_size']
    record={'device':str(device),'batch_seconds':seconds,'seconds_per_document':document_seconds,
            'estimated_full_3_epoch_hours':205734*3*document_seconds/3600,
            'training_rows':len(data['train']),'loss':float(loss.detach().cpu()),
            'note':'Estimate from one pilot batch; window counts and device load change actual cost.'}
    write(OUT/'pilot.json',record);print(record,flush=True)


def train(experiment):
    cfg=config();data=prepared();seed_all(cfg['seed']);directory=OUT/experiment;directory.mkdir(exist_ok=True)
    (directory/'COMPLETE.json').unlink(missing_ok=True)
    model=DocumentClassifier(PRETRAINED,frozen=experiment=='frozen')
    model,record=fit(model,data['rows']['train'],data['rows']['dev'],cfg,experiment,directory,select_threshold)
    scores=predict(model,data['rows']['test'],choose_device(),cfg['eval_batch_size'])
    np.save(directory/'test_scores.npy',scores)
    record.update(signature=signature(),config=cfg,experiment=experiment)
    write(directory/'COMPLETE.json',record)


def classification(y,scores,threshold):
    pred=np.asarray(scores)>=threshold
    return {'macro_f1':float(f1_score(y,pred,average='macro')),'accuracy':float(accuracy_score(y,pred)),
            'threshold':threshold,'per_class':classification_report(y,pred,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,pred).tolist()}


def graph_benchmark(frame,scores,threshold):
    import networkx as nx
    groups={}
    for row,score in zip(frame.itertuples(),scores):
        kind,u,v=parse_claim(row.hypothesis)
        if kind!='direct_cause':continue
        key=system_id(row.premise)
        g=groups.setdefault(key,{'n':int(row.num_variables),'pairs':{}})
        if (u,v) in g['pairs'] and g['pairs'][u,v]['label']!=int(row.label):raise ValueError('Contradictory labels')
        old=g['pairs'].get((u,v),{'score':0})
        g['pairs'][u,v]={'score':max(old['score'],float(score)),'label':int(row.label)}
    result=[]
    for key,group in groups.items():
        if len(group['pairs'])!=group['n']*(group['n']-1):continue
        gold={pair for pair,v in group['pairs'].items() if v['label']==1}
        raw=nx.DiGraph()
        for pair,v in group['pairs'].items():
            if v['score']>=threshold:raw.add_edge(*pair,confidence=v['score'])
        dag,dropped=resolve_cycles(raw)
        rec={'systemId':key,'gold_edges':[list(e) for e in sorted(gold)],'cyclic_before':not nx.is_directed_acyclic_graph(raw),'removed':len(dropped)}
        for stage,graph in [('before',raw),('after',dag)]:
            edges=set(graph.edges)
            rec[stage]={'tp':len(edges&gold),'fp':len(edges-gold),'fn':len(gold-edges),'edges':[list(e) for e in sorted(edges)]}
        result.append(rec)
    summary={'systems':len(result),'cyclic_before':sum(r['cyclic_before'] for r in result),'removed':sum(r['removed'] for r in result)}
    for stage in ['before','after']:
        tp,fp,fn=[sum(r[stage][name] for r in result) for name in ['tp','fp','fn']]
        summary[stage]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'systems_with_edges':sum(bool(r[stage]['edges']) for r in result)}
    return {'summary':summary,'systems':result,'target':'Entailed direct relations, not the full underlying generating DAG'}


def evaluate():
    data=prepared();rows=data['rows'];frames=data['frames'];ml=np.load(OUT/'ml_scores.npz')
    variants={'Matched ML':(ml['dev'],ml['test']),'Full-train ML (context only)':(ml['full_dev'],ml['full_test'])}
    for experiment in ['main','frozen','unweighted']:
        path=OUT/experiment;record=json.loads((path/'COMPLETE.json').read_text())
        if record['signature']!=signature():raise ValueError('Stale experiment: '+experiment)
        variants[experiment]=(np.load(path/'dev_scores.npy'),np.load(path/'test_scores.npy'))
    variants['Hybrid matched ML + DL']=tuple((variants['Matched ML'][i]+variants['main'][i])/2 for i in [0,1])
    # Commit selection on dev before using test labels below.
    selections={n:dict(zip(['threshold','dev_macro_f1'],select_threshold([r['label'] for r in rows['dev']],s[0]))) for n,s in variants.items()}
    eligible=['main','Hybrid matched ML + DL']
    winner=max(eligible,key=lambda n:selections[n]['dev_macro_f1'])
    write(OUT/'selection.json',{'winner':winner,'variants':selections,'signature':signature()})
    results={};graphs={}
    for name,(dev_scores,test_scores) in variants.items():
        results[name]=classification(frames['test'].label,test_scores,selections[name]['threshold'])
        graphs[name]=graph_benchmark(frames['test'],test_scores,selections[name]['threshold'])
        predframe=frames['test'][['docId','premise','hypothesis','label','template','num_variables']].copy()
        predframe['score']=test_scores;predframe['prediction']=(test_scores>=selections[name]['threshold']).astype(int)
        predframe.to_csv(OUT/(name.replace(' ','_').replace('/','_')+'_predictions.csv'),index=False)
    write(OUT/'metrics.json',results);write(OUT/'graph_metrics.json',graphs)
    # Presentation cases are selected only from development inputs/scores, not test labels.
    from src.graph_review import review_demo
    reviews=review_demo(frames['dev'],variants[winner][0],selections[winner]['threshold'])
    reviews['integration_model']=winner
    if reviews['integration'] is None:
        fallback='Full-train ML (context only)'
        reviews=review_demo(frames['dev'],variants[fallback][0],selections[fallback]['threshold'])
        reviews['integration_model']=fallback
        reviews['integration_note']='Selected DL/hybrid supplied no approved development direct edge; integration uses the preserved ML baseline, not a DL extraction success.'
    write(OUT/'demo.json',reviews)
    plot_results(results,graphs)
    text=['# Phase 2 results','',f'Train subset: {len(rows["train"])} records in whole premise groups; dev {len(rows["dev"])}; test {len(rows["test"])}.',
          'This is a local compute-budget experiment, not a full-205,734-row DL training run.',
          '', '| Model | Dev macro F1 | Test macro F1 | Positive P | Positive R |','|---|---:|---:|---:|---:|']
    for name,m in results.items():
        text.append(f'| {name} | {selections[name]["dev_macro_f1"]:.4f} | {m["macro_f1"]:.4f} | {m["per_class"]["1"]["precision"]:.4f} | {m["per_class"]["1"]["recall"]:.4f} |')
    text+=['',f'Demo classifier selected on development data: {winner}.',
           'Full-train ML is contextual only. Fair budget comparison uses Matched ML and the same DL training rows.',
           'Graph scores measure entailed direct relations, with cycle removal evaluated separately.',
           'Demo review uses an explicit finite-DAG constraint check on small systems; it is automated, not human approval.',
           'CPDs in the numerical demo are supplied assumptions, not estimated from Corr2Cause.',
           'Integration graph source: '+reviews['integration_model']+'. '+reviews.get('integration_note',''),
           'One seed (42); no claim of statistically established superiority or real-world causal discovery.']
    (OUT/'RESULTS.md').write_text('\n'.join(text)+'\n')
    write(OUT/'SUCCESS.json',{'signature':signature(),'complete':True,'training_scope':'documented subset','timestamp':pd.Timestamp.now(tz='UTC').isoformat()})
    print('\n'.join(text),flush=True)


def plot_results(results,graphs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Matched ML','frozen','main','unweighted','Hybrid matched ML + DL']
    labels=['Matched ML','Frozen encoder','Fine-tuned DL','Unweighted loss','ML + DL average']
    fig,ax=plt.subplots(figsize=(10,4.8));bars=ax.bar(labels,[results[n]['macro_f1'] for n in names],color=['#8491a3','#c4cbd5','#286b81','#93b9c6','#e29c49'])
    for bar,n in zip(bars,names):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f'{results[n]["macro_f1"]:.3f}',ha='center')
    ax.set(ylim=(0,1),ylabel='Held-out test macro F1',title='Equal training data: ML, DL and ablations');fig.tight_layout();fig.savefig(OUT/'classification.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(3);gn=['Matched ML','main','Hybrid matched ML + DL']
    for i,stage in enumerate(['before','after']):ax.bar(x+(i-.5)*.35,[(graphs[n]['summary'][stage]['precision'] or 0) for n in gn],width=.35,label=stage+' cycle handling')
    ax.set_xticks(x,['Matched ML','Fine-tuned DL','ML + DL average']);ax.set(ylim=(0,1),ylabel='Entailed-edge precision',title='Complete-pair test systems');ax.legend();fig.tight_layout();fig.savefig(OUT/'graph_precision.png',dpi=160);plt.close(fig)


def demo():
    success=json.loads((OUT/'SUCCESS.json').read_text())
    if success['signature']!=signature():raise ValueError('Stale results')
    print((OUT/'RESULTS.md').read_text())
    print(json.dumps(json.loads((OUT/'demo.json').read_text()),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','pilot','train','evaluate','demo'])
    p.add_argument('--experiment',choices=['main','frozen','unweighted'],default='main');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.command in ['prepare','train','evaluate']:(OUT/'SUCCESS.json').unlink(missing_ok=True)
    if args.command=='train':train(args.experiment)
    else:globals()[args.command]()
