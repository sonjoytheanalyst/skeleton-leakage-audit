# ============================================================================
#  Export release artefacts: split indices and per-sequence logits
#  Paste into the bottom of notebook 01 (or any notebook with setup cells run).
#  Requires: X, SUBJ_LABEL, ACT_LABEL, SPLITS, names, subject, setup, action,
#            RESULTS, SkeletonDS, STGCN, _load_ckpt, CKPT_DIR, RES_DIR, DEVICE
#  Runtime: ~15 minutes. No training.
# ============================================================================
import os, json, glob
import numpy as np
import torch

REL = os.path.join(RES_DIR, 'release')
os.makedirs(os.path.join(REL, 'splits'), exist_ok=True)
os.makedirs(os.path.join(REL, 'logits'), exist_ok=True)


# ---------------------------------------------------------------- 1. splits
def export_splits():
    """Write every split as sample NAMES, not integer indices.

    Index-based splits are only meaningful against one particular cache build.
    Names are stable, so anyone can rebuild the same partition from the raw
    annotations.
    """
    nm = np.asarray(names)
    out = {}
    for key, (tr, te) in SPLITS.items():
        out[key] = {'train': nm[np.asarray(tr)].tolist(),
                    'test': nm[np.asarray(te)].tolist(),
                    'n_train': int(len(tr)), 'n_test': int(len(te))}
        print('%-16s %6d train / %6d test' % (key, len(tr), len(te)))

    for key, v in out.items():
        with open(os.path.join(REL, 'splits', key + '.json'), 'w') as f:
            json.dump(v, f)

    # provenance so a reader can verify the partition logic, not just copy it
    meta = {
        'dataset': 'NTU RGB+D 60 (pyskl ntu60_3danno.pkl)',
        'n_sequences': int(len(nm)),
        'seed': int(SEED),
        'definitions': {
            'xsub': 'standard cross-subject split, training subjects listed in XSUB_TRAIN',
            'reid_rand': 'stratified random 80/20 by sequence over all 40 subjects',
            'reid_xview': 'train on cameras 1 and 2, test on camera 3',
            'reid_take': 'take-grouped: all camera views of a (setup, performer, '
                         'replication, action) tuple on the same side, stratified by subject',
            'reid_xsetup': 'train on even-numbered setups, test on odd',
            'reid_k5': 'limited-budget attacker: 5 labelled sequences per identity'},
        'xsub_train_subjects': sorted(int(s) for s in XSUB_TRAIN)}
    with open(os.path.join(REL, 'splits', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('\nsplits written to', os.path.join(REL, 'splits'))


# ---------------------------------------------------------------- 2. logits
@torch.no_grad()
def export_logits(tag, ckpt, split, labels, num_classes, bs=256):
    """Save per-sequence logits so readers can recompute intervals and
    significance tests without retraining anything."""
    out_p = os.path.join(REL, 'logits', tag + '.npz')
    if os.path.exists(out_p):
        print('%-26s already exported' % tag); return
    if not os.path.exists(ckpt):
        print('%-26s checkpoint missing, skipped' % tag); return

    m = STGCN(num_classes).to(DEVICE)
    sd = _load_ckpt(ckpt)
    m.load_state_dict(sd.get('model', sd), strict=False)
    m.eval()

    idx = np.sort(SPLITS[split][1])
    ld = torch.utils.data.DataLoader(SkeletonDS(X, labels, idx, False), bs,
                                     shuffle=False, num_workers=4)
    chunks = []
    for xb, _ in ld:
        with torch.autocast('cuda', enabled=(DEVICE == 'cuda')):
            chunks.append(m(xb.to(DEVICE)).float().cpu().numpy())
    del m
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    logits = np.concatenate(chunks, 0).astype(np.float16)
    y = np.asarray(labels)[idx]
    acc = 100.0 * (logits.argmax(1) == y).mean()
    np.savez_compressed(out_p, logits=logits, labels=y,
                        names=np.asarray(names)[idx], split=split)
    print('%-26s %6d sequences, top-1 %.2f%%' % (tag, len(y), acc))


# ---------------------------------------------------------------- 3. run
export_splits()

print('\nexporting logits')
JOBS = [
    ('A_action_xsub',      'xsub',       ACT_LABEL,  60),
    ('B_reid_rand',        'reid_rand',  SUBJ_LABEL, 40),
    ('B_reid_xview',       'reid_xview', SUBJ_LABEL, 40),
    ('B_reid_take',        'reid_take',  SUBJ_LABEL, 40),
    ('B_reid_xsetup',      'reid_xsetup', SUBJ_LABEL, 40),
    ('C_probe_frozen',     'reid_rand',  SUBJ_LABEL, 40),
    ('D_finetune_from_A',  'reid_rand',  SUBJ_LABEL, 40),
]
for tag, split, lab, nc in JOBS:
    ck = None
    for root in [CKPT_DIR, '/content/drive/MyDrive/PhD_Q1/checkpoints']:
        p = os.path.join(root, tag + '_best.pt')
        if os.path.exists(p):
            ck = p; break
    export_logits(tag, ck or '', split, lab, nc)

# ---------------------------------------------------------------- 4. manifest
man = {'splits': sorted(os.listdir(os.path.join(REL, 'splits'))),
       'logits': sorted(os.listdir(os.path.join(REL, 'logits')))}
with open(os.path.join(REL, 'MANIFEST.json'), 'w') as f:
    json.dump(man, f, indent=2)
print('\nrelease artefacts in', REL)
print(json.dumps(man, indent=2))
