"""One DistilBERT reader shared across windows, followed by one document decision."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def seed_all(seed):
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def choose_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def token_windows(premise_ids, hypothesis_ids, tokenizer, limit=512):
    capacity = limit - len(hypothesis_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    if capacity <= 0:
        raise ValueError('Hypothesis leaves no space for premise; refusing truncation')
    segments = [premise_ids[i:i+capacity] for i in range(0, len(premise_ids), capacity)] or [[]]
    return [tokenizer.build_inputs_with_special_tokens(segment, hypothesis_ids) for segment in segments]


def encode_rows(frame, tokenizer):
    # Repeated premises/hypotheses are tokenized only once.
    premises = frame.premise.unique().tolist(); hypotheses = frame.hypothesis.unique().tolist()
    def encode(texts):
        return dict(zip(texts, tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)['input_ids']))
    pmap, hmap = encode(premises), encode(hypotheses)
    rows = []
    for row in frame.itertuples():
        p, h = pmap[row.premise], hmap[row.hypothesis]
        rows.append({'windows': token_windows(p, h, tokenizer), 'label': int(row.label),
                     'docId': row.docId, 'premise_tokens': len(p), 'hypothesis_tokens': len(h)})
    return rows


def collate(rows):
    windows = [w for row in rows for w in row['windows']]
    owners = [i for i, row in enumerate(rows) for _ in row['windows']]
    length = 512  # fixed encoder shape avoids unbounded MPS graph caching
    if any(len(w)>length for w in windows):raise ValueError("Window exceeds encoder limit")
    ids = torch.zeros((len(windows), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, w in enumerate(windows):
        ids[i, :len(w)] = torch.tensor(w); mask[i, :len(w)] = 1
    return {'input_ids': ids, 'attention_mask': mask,
            'owners': torch.tensor(owners), 'labels': torch.tensor([r['label'] for r in rows])}


def mean_windows(features, owners, documents):
    # Mask-free flattened windows: padded document windows never enter the encoder.
    membership=(torch.arange(documents,device=features.device)[:,None]==owners[None,:]).to(features.dtype)
    return (membership/membership.sum(1,keepdim=True)) @ features


class DocumentClassifier(nn.Module):
    def __init__(self, pretrained, frozen=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(pretrained), local_files_only=True, attn_implementation='sdpa')
        self.head = nn.Sequential(nn.Linear(self.encoder.config.dim, self.encoder.config.dim), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(self.encoder.config.dim, 2))
        self.frozen = frozen
        if frozen:
            self.encoder.requires_grad_(False)

    def forward(self, input_ids, attention_mask, owners, labels=None):
        if self.frozen:
            self.encoder.eval()
        with torch.set_grad_enabled(self.training and not self.frozen):
            vectors = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        pooled = mean_windows(vectors, owners, len(labels) if labels is not None else int(owners.max())+1)
        return self.head(pooled)


def predict(model, rows, device, batch_size=4):
    model.eval(); values = []
    with torch.inference_mode():
        for batch in DataLoader(rows, batch_size=batch_size, collate_fn=collate, shuffle=False):
            output = model(**{k: v.to(device) for k,v in batch.items()})
            probabilities=output.softmax(-1)[:,1]
            if not torch.isfinite(probabilities).all():raise FloatingPointError('Non-finite model scores')
            values.extend(probabilities.cpu().tolist())
            if device.type=='mps':torch.mps.empty_cache()
    return np.array(values)


def fit(model, rows, dev_rows, config, experiment, out, select_threshold):
    from sklearn.metrics import f1_score
    if experiment == 'frozen':
        return fit_frozen(model, rows, dev_rows, config, out, select_threshold)
    seed_all(config['seed']); device = choose_device(); model.to(device)
    weighted = experiment != 'unweighted'
    counts = np.bincount([r['label'] for r in rows], minlength=2)
    weights = torch.tensor(len(rows)/(2*counts), dtype=torch.float32, device=device) if weighted else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['head_lr'] if experiment=='frozen' else config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    loader = DataLoader(rows, batch_size=config['batch_size'], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config['seed']))
    best=-1.; history=[]; patience=0; start=time.time()
    for epoch in range(config['epochs']):
        model.train(); optimizer.zero_grad(); losses=[]
        for step,batch in enumerate(loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            loss=loss_fn(model(**batch), batch['labels'])
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            losses.append(float(loss.detach().cpu()))
            # Normalize each accumulation group, including a shorter last group.
            group_start=(step//config['accumulation'])*config['accumulation']
            group_steps=min(config['accumulation'],len(loader)-group_start)
            (loss/group_steps).backward()
            if (step+1)%config['accumulation']==0 or step+1==len(loader):
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); optimizer.zero_grad()
            if device.type=='mps' and (step+1)%25==0:
                torch.mps.empty_cache()
            if step%50==0:
                print(f'{experiment} epoch {epoch+1} step {step}/{len(loader)} loss={losses[-1]:.4f}',flush=True)
        scores=predict(model,dev_rows,device,config['eval_batch_size'])
        threshold,score=select_threshold([r['label'] for r in dev_rows],scores)
        record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'dev_macro_f1':score,
                'threshold':threshold,'elapsed_seconds':time.time()-start}
        history.append(record);print(experiment,record,flush=True)
        if score>best:
            best=score;patience=0
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
            np.save(out/'dev_scores.npy',scores)
            best_record=record.copy()
        else:
            patience+=1
            if patience>=config['patience']: break
    model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    return model, {'best':best_record,'history':history,'training_rows':len(rows),
                   'class_counts':counts.tolist(),'weighted_loss':weighted,'device':str(device),
                   'elapsed_seconds':time.time()-start}


def fit_frozen(model, rows, dev_rows, config, out, select_threshold):
    """Encode once, then train only the small head on cached document vectors."""
    seed_all(config['seed']);device=choose_device();model.to(device);model.eval();start=time.time()
    def embeddings(records):
        values=[]
        with torch.inference_mode():
            for i,b in enumerate(DataLoader(records,batch_size=config['eval_batch_size'],collate_fn=collate)):
                b={k:v.to(device) for k,v in b.items()}
                vectors=model.encoder(input_ids=b['input_ids'],attention_mask=b['attention_mask']).last_hidden_state[:,0]
                values.append(mean_windows(vectors,b['owners'],len(b['labels'])).cpu())
                if i%100==0:print('frozen embeddings batch',i,flush=True)
        return torch.cat(values).to(device)
    x=embeddings(rows);xd=embeddings(dev_rows)
    y=torch.tensor([r['label'] for r in rows],device=device)
    counts=np.bincount(y.cpu().numpy(),minlength=2)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(rows)/(2*counts),dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=config['head_lr'],weight_decay=config['weight_decay'])
    history=[];best=-1.;patience=0
    for epoch in range(config['epochs']):
        model.head.train();indices=torch.randperm(len(rows),device=device);losses=[]
        for idx in indices.split(64):
            optimizer.zero_grad();loss=loss_fn(model.head(x[idx]),y[idx]);loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
        model.head.eval()
        with torch.inference_mode():scores=model.head(xd).softmax(-1)[:,1].cpu().numpy()
        threshold,score=select_threshold([r['label'] for r in dev_rows],scores)
        record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'dev_macro_f1':score,'threshold':threshold,'elapsed_seconds':time.time()-start}
        history.append(record);print('frozen',record,flush=True)
        if score>best:
            best=score;patience=0;best_record=record
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt');np.save(out/'dev_scores.npy',scores)
        else:
            patience+=1
            if patience>=config['patience']:break
    model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    return model,{'best':best_record,'history':history,'training_rows':len(rows),'class_counts':counts.tolist(),
                  'weighted_loss':True,'device':str(device),'elapsed_seconds':time.time()-start,
                  'frozen_head_batch_size':64,'note':'Frozen document vectors encoded once; only the head is optimized.'}
