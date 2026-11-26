
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
print('near_dups '+json.dumps(dict(hash_candidates=candidates,strong=strong,same_label=same_label,
      diff_label=diff_label,moderate=moderate,clusters=family_summary(strong_pairs),
      similarities={k:sorted(v,reverse=True)[:3] for k,v in top.items()},
      diff_examples=[r for i,j,r in strong_pairs if rows[i][1]!=rows[j][1]][:10],
      moderate_cross_examples=[r for i,j,r in moderate_pairs if rows[i][0]!=rows[j][0]][:5])))

import csv
from PIL import ImageDraw, ImageFont
out=Path(os.environ['TEASMART_LEAKAGE_OUTPUT']).resolve()
assert not out.is_relative_to(root.resolve()) and not root.resolve().is_relative_to(out)
out.mkdir(exist_ok=False)
fontpath=Path(os.environ['WINDIR'])/'Fonts'
def font(name,size):
    return ImageFont.truetype(str(fontpath/name),size)
heading=font('segoeuib.ttf',24)
tag=font('segoeuib.ttf',14)
small=font('segoeui.ttf',13)
mono=font('consola.ttf',13)
pair_order={frozenset(('train','test')):0,frozenset(('valid','test')):1,frozenset(('train','valid')):2}
split_order={'train':0,'valid':1,'test':2}
def is_cross(pair):
    i,j,_=pair
    return rows[i][0]!=rows[j][0]
def seq_for(pair):
    i,j,r=pair
    return (pair_order.get(frozenset((rows[i][0],rows[j][0])),3), -r[0], rows[i][2],rows[j][2])
strong_cross=sorted(filter(is_cross,strong_pairs),key=seq_for)
moderate_cross=sorted(filter(is_cross,moderate_pairs),key=seq_for)
inside=sorted((pair for pair in strong_pairs if not is_cross(pair)),key=seq_for)
assert (len(strong_cross),len(moderate_cross),len(inside))==(15,225,18)
sheets=(
    ('S','01','High-similarity cross-split pairs',strong_cross,3,3,214,True),
    ('M','02','Moderate-similarity cross-split pairs',moderate_cross,3,5,128,False),
    ('I','03','High-similarity within-split pairs',inside,3,3,172,True),
)
def source_stem(filename):
    return filename.split('_jpg.rf.')[0]
def text_block(draw, x, y, name, max_width, face):
    buf=''
    for part in name.split('.rf.',1):
        line=part if not buf else '.rf.'+part
        while draw.textlength(line,font=face)>max_width:
            n=len(line)
            while n>1 and draw.textlength(line[:n],font=face)>max_width:
                n-=1
            draw.text((x,y),line[:n],font=face,fill='#263a33')
            line=line[n:]
            y+=15
        draw.text((x,y),line,font=face,fill='#263a33')
        y+=15
        buf='split'
def main_view(im,match):
    j=0 if not match else match
    # The strong sheets always show the file as exported. Moderate sheets align orientation when helpful.
    if j//2:
        im=im.rotate(j//2*90)
    if j%2:
        im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return im
records=[]
for prefix,seq,title,pairs,cols,per_col,width,full_names in sheets:
    page_size=cols*per_col
    page_count=(len(pairs)+page_size-1)//page_size
    card_width=2*width+24
    card_height=width+(108 if full_names else 76)
    for page in range(page_count):
        selected=pairs[page*page_size:(page+1)*page_size]
        pic=Image.new('RGB',(cols*(card_width+10)+10,52+card_height*((len(selected)+cols-1)//cols)+12),'#e9eee9')
        draw=ImageDraw.Draw(pic)
        draw.text((15,11),f'{title}  |  {page+1}/{page_count}',font=heading,fill='#143e30')
        contact_name=f'{seq}-{prefix}-{page+1:02}.png'
        for local,(i,j,record) in enumerate(selected):
            number=page*page_size+local+1
            label=f'{prefix}{number:03}'
            if split_order[rows[i][0]]>split_order[rows[j][0]]:
                i,j=j,i
                t=record[4]
                t=t if t%2 else ((8-t)%8)
            else:
                t=record[4]
            x=10+(local%cols)*(card_width+10)
            y=49+(local//cols)*card_height
            draw.rounded_rectangle((x,y,x+card_width,y+card_height-7),radius=8,fill='white',outline='#bfcdc4')
            draw.text((x+10,y+5),f'{label}   SSIM {record[0]:.4f}   MAE {record[1]:.2f}',font=tag,fill='#143e30')
            for k,idx in enumerate((i,j)):
                sx=x+8+k*(width+8)
                split,cl,rel=rows[idx]
                with Image.open(root / rel) as opened:
                    shown=main_view(opened.convert('RGB'), t if k and prefix=='M' else 0)
                    pic.paste(shown.resize((width,width),Image.Resampling.LANCZOS),(sx,y+29))
                ytext=y+width+32
                note=f'{split.upper()} | {cl}'
                draw.text((sx,ytext),note,font=small,fill='#143e30')
                name=Path(rel).name
                if full_names:
                    text_block(draw,sx,ytext+17,name,width,mono)
                else:
                    draw.text((sx,ytext+15),source_stem(name),font=small,fill='#263a33')
                    if k and t:
                        draw.text((sx,ytext+29),f'{90*(t//2)} deg'+(' + flip' if t%2 else ''),font=small,fill='#87641f')
            records.append(dict(id=label,split_a=rows[i][0],split_b=rows[j][0],class_a=rows[i][1],class_b=rows[j][1],
                                path_a=rows[i][2],path_b=rows[j][2],ssim_64=record[0],mae_64=record[1],
                                phash_distance=record[2],dhash_distance=record[3],rotation_b=90*(t//2),flip_b=bool(t%2),
                                sheet=contact_name))
        pic.save(out/contact_name,optimize=True)
with (out/'cap-anh.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print('REVIEW_FILES '+ json.dumps(dict(path=str(out),sheets=[x.name for x in sorted(out.glob('*.png'))],
     ids={p:[r['id'] for r in records if r['id'].startswith(p) and r['split_b']=='test'][-1:] for p in ('S','M')},
     class_mismatch=[r['id'] for r in records if r['class_a']!=r['class_b']])))
