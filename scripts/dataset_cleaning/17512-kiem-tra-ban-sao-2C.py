
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image

root = Path(os.environ['TEASMART_DATASET_PATH'])
splits = ('train','valid','test')
rows = []
invalid = []
unexpected = []
warnings_seen = []
counts = {s: Counter() for s in splits}
extensions,formats,sizes,modes,orientation = (Counter() for i in range(5))
raw_groups, pixel_groups, stem_groups = (defaultdict(list) for i in range(3))
thumbs = []
hashes_p, hashes_d = [], []

def as_int(bits):
    return int.from_bytes(np.packbits(bits).tobytes(),'big')

def transforms(a):
    return [x for turn in range(4) for x in (np.rot90(a,turn), np.rot90(a,turn)[:,::-1])]

def signatures(a):
    g = cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    p = cv2.dct(cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].flatten()
    p_bits = p > np.median(p[1:])
    p_bits[0] = False
    d = cv2.resize(g,(9,8),interpolation=cv2.INTER_AREA)
    return as_int(p_bits),as_int((d[:,1:]>d[:,:-1]).flatten())

for split in splits:
    split_dir = root/split
    for p in sorted(split_dir.rglob('*')):
        if not p.is_file():
            continue
        parts = p.relative_to(split_dir).parts
        rel = p.relative_to(root).as_posix()
        if len(parts)!=2:
            unexpected.append(rel)
        label = parts[0] if len(parts)>1 else '(outside class)'
        extensions[p.suffix.lower()] += 1
        try:
            data = p.read_bytes()
        except OSError as exc:
            invalid.append((rel,str(exc)))
            continue
        index = len(rows)
        try:
            with warnings.catch_warnings(record=True) as logs:
                warnings.simplefilter('always')
                with Image.open(io.BytesIO(data)) as img:
                    img.verify()
                with Image.open(io.BytesIO(data)) as img:
                    img.load()
                    shape = img.size
                    rgb = img.convert('RGB')
                    reduced = np.array(rgb.resize((64,64),Image.Resampling.LANCZOS))
                    color_bytes = rgb.tobytes()
                    f, m, o = img.format, img.mode, img.getexif().get(274,1)
                warnings_seen.extend((rel,str(log.message)) for log in logs)
        except (OSError,ValueError,SyntaxError) as exc:
            invalid.append((rel,str(exc)))
            continue
        counts[split][label] += 1
        formats[f] += 1
        modes[m] += 1
        orientation[o] += 1
        sizes[shape] += 1
        raw_groups[hashlib.sha256(data).digest()].append(index)
        pixel_groups[hashlib.sha256(str(shape).encode('ascii')+color_bytes).digest()].append(index)
        stem_groups[p.name.casefold().split('.rf.',1)[0]].append(index)
        rows.append((split,label,rel))
        thumbs.append(reduced)
        ph,dh = zip(*(signatures(x) for x in transforms(reduced)))
        hashes_p.append(ph)
        hashes_d.append(dh)
print('inventory '+json.dumps(dict(root=sorted(p.name for p in root.iterdir()), splits=counts, extensions=extensions,
             formats=formats, sizes={str(k):v for k,v in sizes.items()}, modes=modes, orientation=orientation,
             invalid_count=len(invalid), invalid=invalid[:12], nesting=unexpected[:12], warnings=warnings_seen[:12],
             warning_count=len(warnings_seen))))
def pairkey(i,j):
    return '/'.join(sorted((rows[i][0],rows[j][0])))

def group_report(groups, title):
    multi = [x for x in groups.values() if len(x)>1]
    pairs = [pair for group in multi for pair in combinations(group,2)]
    cross = [(i,j) for i,j in pairs if rows[i][0]!=rows[j][0]]
    conflicts = [(i,j) for i,j in pairs if rows[i][1]!=rows[j][1]]
    print(title+' '+json.dumps(dict(groups=len(multi), images=sum(map(len,multi)),
          cross_groups=sum(len({rows[i][0] for i in group})>1 for group in multi),
          pairs=Counter(pairkey(i,j) for i,j in pairs), conflict_pairs=len(conflicts),
          cross_samples=[(rows[i][2],rows[j][2]) for i,j in cross[:8]],
          conflict_samples=[(rows[i][2],rows[j][2]) for i,j in conflicts[:5]])))
    return set(pairs)
raw_pairs = group_report(raw_groups,'file_sha256')
pixel_pairs = group_report(pixel_groups,'decoded_rgb_sha256')
stem_pairs = group_report(stem_groups,'filename_source_stem')

def image_stats(a):
    a = a.astype(np.float32)
    mean = cv2.GaussianBlur(a,(7,7),1.5)
    var = cv2.GaussianBlur(a*a,(7,7),1.5)-mean*mean
    return a,mean,var

def compare(a,b,stats1):
    y,m2,v2 = image_stats(b)
    x,m1,v1 = stats1
    ma = float(np.mean(np.abs(x-y)))
    c = cv2.GaussianBlur(x*y,(7,7),1.5)-m1*m2
    c1,c2 = (2.55**2,7.65**2)
    sim = float(np.mean(((2*m1*m2+c1)*(2*c+c2)/((m1*m1+m2*m2+c1)*(v1+v2+c2)))[3:-3,3:-3]))
    return sim,ma

phashes = np.asarray(hashes_p,dtype=np.uint64)
dhashes = np.asarray(hashes_d,dtype=np.uint64)
candidates, strong, same_label, diff_label, moderate = (Counter() for i in range(5))
strong_pairs = []
moderate_pairs = []
top = defaultdict(list)
n=len(rows)
for i in range(n-1):
    pd = np.bitwise_count(phashes[i+1:] ^ phashes[i,0])
    dd = np.bitwise_count(dhashes[i+1:] ^ dhashes[i,0])
    possible = (pd<=6) | ((pd<=10)&(dd<=8))
    ii = np.flatnonzero(np.any(possible, axis=1))
    if len(ii)==0:
        continue
    astats = image_stats(thumbs[i])
    for offset in ii:
        j = int(offset)+i+1
        if (i,j) in pixel_pairs:
            continue
        k = pairkey(i,j)
        candidates[k]+=1
        best = None
        orientations = transforms(thumbs[j])
        for t in np.flatnonzero(possible[offset]):
            sim,ma = compare(thumbs[i],orientations[t],astats)
            if best is None or sim>best[0]:
                best = (sim,ma,int(pd[offset,t]),int(dd[offset,t]),int(t))
        sim,ma,*rest = best
        cross = rows[i][0]!=rows[j][0]
        record = (round(sim,4),round(ma,2),*rest,rows[i][2],rows[j][2])
        if cross:
            top[k].append(record)
        if sim>=0.95 and ma<=10:
            strong[k]+=1
            (same_label if rows[i][1]==rows[j][1] else diff_label)[k]+=1
            strong_pairs.append((i,j,record))
        elif sim>=0.90 and ma<=15:
            moderate[k]+=1
            moderate_pairs.append((i,j,record))

def family_summary(pairs):
    parent = {}
    def head(a):
        while parent.get(a,a)!=a:
            a=parent[a]
        return a
    for i,j,_ in pairs:
        if head(i)!=head(j):
            parent[head(j)]=head(i)
    components=defaultdict(set)
    for i,j,_ in pairs:
        components[head(i)].update((i,j))
    cross_sets=[v for v in components.values() if len({rows[i][0] for i in v})>1]
    return dict(total=len(components),cross=len(cross_sets),
                cross_images=Counter(rows[i][0] for v in cross_sets for i in v),
                cross_labels=Counter(rows[i][1] for v in cross_sets for i in v),
                max_cluster=max(map(len,components.values()),default=0))

import csv
with (Path(os.environ['TEASMART_2C_AUDIT'])/'cap-anh.csv').open(newline='',encoding='utf-8-sig') as f:
    pair_review=list(csv.DictReader(f))
kept={x[2] for x in rows}
expected={frozenset((r['path_a'],r['path_b'])):r for r in pair_review if r['review']=='INSUFFICIENT_EVIDENCE'
    and r['split_a']!=r['split_b'] and r['path_a'] in kept and r['path_b'] in kept}
recomputed={frozenset((rows[i][2],rows[j][2])) for i,j,_ in moderate_pairs if rows[i][0]!=rows[j][0]}
assert len(rows)==5228 and not invalid and not warnings_seen and not unexpected
assert not raw_pairs and not pixel_pairs
assert not any(rows[i][0]!=rows[j][0] for i,j,_ in strong_pairs)
assert set(expected)==recomputed and len(recomputed)==165
assert all(r['review']=='INSUFFICIENT_EVIDENCE' for r in expected.values())
print('RECHECK_OK '+json.dumps(dict(retained_cross_split_candidates=len(recomputed),
     touching_test=sum(r['split_a']=='test' or r['split_b']=='test' for r in expected.values()),
     cross_split_candidates=Counter('/'.join(sorted((r['split_a'],r['split_b']))) for r in expected.values()),
     strong_cross_split=0, exact_sha256_duplicates=0, decoded_pixel_duplicates=0, invalid_images=len(invalid))))
