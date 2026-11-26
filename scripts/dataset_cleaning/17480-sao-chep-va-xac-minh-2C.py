
import csv, hashlib, io, json, os, shutil, stat
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

def need(condition,message):
    if not condition:
        raise ValueError(message)

root_arg=Path(os.environ['TEASMART_2C_SOURCE'])
dest=Path(os.environ['TEASMART_2C_DEST'])
audit=Path(os.environ['TEASMART_2C_AUDIT']).resolve(strict=True)
write=os.environ['TEASMART_2C_MODE']=='copy'
require_absent=lambda p: need(not os.path.lexists(p),f'Destination already exists: {p}')
require_absent(dest)
root=root_arg.resolve(strict=True)
need(dest.name=='tea-leaf-clean-v1' and root.name=='My First Project.v1i.folder','Unexpected source or destination name')
parent=dest.parent.resolve(strict=True)
need(root_arg==root and dest.parent==parent and root.parent.parent==parent,'Unexpected path resolution')
need(not dest.is_relative_to(root) and not root.is_relative_to(dest),'Destination overlaps source')
def safe_stat(p,kind):
    s=p.lstat()
    need(not s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,f'Unsupported reparse point: {p}')
    need((stat.S_ISDIR if kind=='dir' else stat.S_ISREG)(s.st_mode),f'Expected {kind}: {p}')
    return s
safe_stat(root,'dir')
safe_stat(parent,'dir')
folds=('train','valid','test')
priority={key:i for i,key in enumerate(folds)}
labels=(
 '1. Tea algal leaf spot','2. Brown Blight','3. Gray Blight','4. Helopeltis',
 '5. Red spider','6. Green mirid bug','7. Healthy leaf',
)
baseline=((296,91,31),(345,110,51),(726,183,103),(412,131,64),(358,105,52),(902,247,133),(650,188,97))
approved=((296,91,31),(335,109,51),(721,183,103),(401,130,63),(357,103,52),(890,247,133),(647,188,97))
expected_actions={
 'KEEP':(3647,1051,530),
 'EXCLUDE_NEAR_DUPLICATE':(31,3,0),
 'EXCLUDE_EVAL_REPEAT':(0,1,1),
 'QUARANTINE_PENDING_REVIEW':(11,0,0),
}
original_docs=('README.dataset.txt','README.roboflow.txt')
def entries(directory):
    safe_stat(directory,'dir')
    return {p.name:p for p in directory.iterdir()}

def scan_layout(directory,top_docs,expected):
    found={}
    top=entries(directory)
    need(set(top)==set(folds).union(top_docs),f'Unexpected top-level entries in {directory}')
    for name in top_docs:
        safe_stat(top[name],'file')
    for y,fold in enumerate(folds):
        cats=entries(top[fold])
        need(set(cats)==set(labels),f'Class folders differ in {directory}/{fold}')
        for x,label in enumerate(labels):
            photos=entries(cats[label])
            need(len(photos)==expected[x][y],f'Image count differs: {directory}/{fold}/{label} has {len(photos)} expected {expected[x][y]}')
            for name,path in photos.items():
                need(path.suffix=='.jpg',f'Unexpected extension: {path}')
                safe_stat(path,'file')
                found[f'{fold}/{label}/{name}']=path
    return found
images=scan_layout(root,original_docs,baseline)

def read_csv(name, fields):
    path=audit/name
    safe_stat(path,'file')
    data=path.read_bytes()
    evidence_hashes[name]=hashlib.sha256(data).hexdigest()
    reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig'),newline=''))
    need(reader.fieldnames==fields.split(),f'Unexpected fields in {name}')
    rows=list(reader)
    need(all(None not in r and all(v is not None for v in r.values()) for r in rows),f'Malformed rows in {name}')
    return rows
def index(rows,key,name):
    result={}
    for r in rows:
        v=r[key]
        need(bool(v) and v not in result,f'Duplicate or missing {key} in {name}: {v}')
        result[v]=r
    return result

evidence_hashes={}
m=index(read_csv('de-xuat-theo-anh.csv','path original_split original_label proposed_split proposed_action reason same_leaf_group precaution_group'),'path','manifest 2B')
p=index(read_csv('cap-anh.csv','id split_a split_b class_a class_b path_a path_b ssim_64 mae_64 phash_distance dhash_distance rotation_b flip_b sheet review review_note same_leaf_group precaution_group'),'id','pair review')
g=index(read_csv('nhom-anh.csv','group_id group_type splits preferred_split images image_count pair_evidence associations proposed_impact'),'group_id','group review')
report=(audit/'bao-cao.md').read_bytes()
evidence_hashes['bao-cao.md']=hashlib.sha256(report).hexdigest()
report_text=report.decode('utf-8-sig')
need(all(x in report_text for x in ('5.275','3.647','1.051','530','5.228')),'2B report quantities differ')
need(len(m)==5275 and set(m)==set(images),'2B manifest does not match source paths exactly')
need(len(p)==258 and Counter(x[0] for x in p)=={'S':15,'M':225,'I':18},'2B pair list differs')
need(len(g)==59 and Counter(x[0] for x in g)=={'G':50,'Q':9},'2B group list differs')

counts=Counter()
by_class=Counter()
for path,row in m.items():
    fold,label,_=PurePosixPath(path).parts
    action=row['proposed_action']
    need(row['original_split']==fold and row['original_label']==label and action in expected_actions and row['reason'],
         f'Invalid split, label, action or missing reason: {path}')
    need(row['proposed_split']==(fold if action=='KEEP' else ''),f'Unapproved split move or exclusion: {path}')
    counts[(action,fold)]+=1
    if action=='KEEP':
        by_class[(label,fold)]+=1
need({a:tuple(counts[(a,f)] for f in folds) for a in expected_actions}==expected_actions,'Action counts differ from 2B')
need(tuple(tuple(by_class[(label,fold)] for fold in folds) for label in labels)==approved,'Class counts differ from 2B')
keep={x for x,r in m.items() if r['proposed_action']=='KEEP'}

high='VERY_LIKELY_SAME_LEAF'
suspect='SUSPECT_REVIEW'
unknown='INSUFFICIENT_EVIDENCE'
seen=set()
review_counts=Counter()
for key,r in p.items():
    paths=frozenset(r['path_'+side] for side in 'ab')
    need(len(paths)==2 and paths<=m.keys() and paths not in seen,f'Invalid or duplicate pair: {key}')
    seen.add(paths)
    for side in 'ab':
        row=m[r['path_'+side]]
        need(row['original_split']==r['split_'+side] and row['original_label']==r['class_'+side],f'Pair metadata conflict: {key}')
    verdict=r['review']
    need(verdict in (high,suspect,unknown),f'Unknown pair verdict {key}')
    review_counts[(key[0],verdict)]+=1
    if verdict==high or verdict==suspect:
        need(r['class_a']==r['class_b'],f'Reviewed near-duplicate label conflict: {key}')
        field='same_leaf_group' if verdict==high else 'precaution_group'
        group_id=r[field]
        need(group_id in g and paths<=set(g[group_id]['images'].split(' | ')) and all(m[path][field]==group_id for path in paths),
             f'Pair/group membership conflict: {key}')
        if r['split_a']!=r['split_b']:
            need(not paths<=keep,f'Reviewed cross-split pair survives: {key}')
    if key.startswith(('S','I')):
        need(verdict==high and (r['split_a']==r['split_b'])==key.startswith('I'),f'Inconsistent S or I classification: {key}')
need(review_counts=={('S',high):15,('I',high):18,('M',high):19,('M',suspect):10,('M',unknown):196},'Reviewed verdict totals differ')

group_members={}
for gid,r in g.items():
    members=r['images'].split(' | ')
    paths=set(members)
    group_members[gid]=paths
    need(len(members)==len(paths)==int(r['image_count']) and paths<=m.keys(),f'Invalid images in {gid}')
    kind='same_leaf_group' if gid.startswith('G') else 'precaution_group'
    need(r['group_type']==(high if kind=='same_leaf_group' else 'PRECAUTION_ONLY') and all(m[x][kind]==gid for x in paths),f'Group metadata conflict {gid}')
    splits={m[x]['original_split'] for x in paths}
    top=max(splits,key=priority.get)
    need(splits==set(r['splits'].split(' | ')) and top==r['preferred_split'],f'Group split conflict {gid}')
    links=r['pair_evidence'].split(' | ')
    need(len(set(links))==len(links) and all(x in p and p[x]['review'] in (high,suspect) for x in links),f'Invalid evidence for {gid}')
    need({p[x]['path_'+side] for x in links for side in 'ab'}==paths,f'Evidence/member mismatch {gid}')
    if gid.startswith('Q'):
        need(any(p[x]['review']==suspect for x in links),f'No suspect links for {gid}')
    else:
        need(all(p[x]['review']==high for x in links) and len({m[x]['original_label'] for x in paths})==1,f'Conflicting reviewed labels/evidence for {gid}')
    active=paths & keep
    surviving_folds={m[x]['original_split'] for x in active}
    all_train_quarantined=(kind=='same_leaf_group' and splits=={'train'} and not active
        and all(m[x]['proposed_action']=='QUARANTINE_PENDING_REVIEW' for x in paths))
    need(surviving_folds=={top} or all_train_quarantined,f'Cross-split group or lost preferred group: {gid}')
    if kind=='same_leaf_group' and top in ('valid','test'):
        need(len(active)==1,f'Multiple representatives in evaluation group {gid}')

for path,row in m.items():
    for field in ('same_leaf_group','precaution_group'):
        gid=row[field]
        if gid:
            need(gid in group_members and path in group_members[gid],f'Invalid reverse group membership: {path}')
    action=row['proposed_action']
    if action=='EXCLUDE_NEAR_DUPLICATE':
        group=g[row['same_leaf_group']]
        need(priority[group['preferred_split']]>priority[row['original_split']],f'Wrong exclusion type: {path}')
    if action=='EXCLUDE_EVAL_REPEAT':
        group=g[row['same_leaf_group']]
        need(group['preferred_split']==row['original_split'] and row['original_split']!='train',f'Wrong repeat type: {path}')
    if action=='QUARANTINE_PENDING_REVIEW':
        group=g[row['precaution_group']]
        need(row['original_split']=='train' and priority[group['preferred_split']]>0,f'Wrong quarantine: {path}')

original_bytes=sum(path.stat().st_size for rel,path in images.items() if rel in keep)
need(shutil.disk_usage(parent).free>original_bytes+16*1024*1024,'Insufficient free space')
remaining=[r for r in p.values() if r['path_a'] in keep and r['path_b'] in keep and r['split_a']!=r['split_b']]
need(all(r['review']==unknown for r in remaining),'An approved exclusion or quarantine pair remains across splits')
print('PREFLIGHT '+json.dumps(dict(source_images=len(m),actions=expected_actions,groups=len(g),kept_bytes=original_bytes,
     unresolved_pairs=len(remaining),unresolved_pairs_touching_test=sum(r['split_b']=='test' or r['split_a']=='test' for r in remaining),
     cross_label_screening_candidates=sum(r['class_a']!=r['class_b'] for r in p.values()),
     cross_label_review_conflicts=0,remaining_cross_label_candidates=sum(r['class_a']!=r['class_b'] for r in remaining))))

def stat_tuple(p,kind):
    s=safe_stat(p,kind)
    return (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_file_attributes)

def image_hashes(data, path):
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        need(im.format=='JPEG' and im.size==(224,224) and im.mode=='RGB',f'Invalid image format or dimensions: {path}')
        return hashlib.sha256(im.tobytes()).hexdigest()

def snapshot_source(decode=False):
    scanned=scan_layout(root,original_docs,baseline)
    r={}
    pixel_hashes={}
    original_hashes={}
    dirs=(root,*(root/f for f in folds),*(root/f/label for f in folds for label in labels))
    for d in dirs:
        r[d.relative_to(root).as_posix()]=stat_tuple(d,'dir')
    for rel,path in scanned.items():
        before=stat_tuple(path,'file')
        data=path.read_bytes()
        raw=hashlib.sha256(data).hexdigest()
        need(before==stat_tuple(path,'file'),f'Source image changed during read: {rel}')
        r[rel]=(*before, raw)
        if decode:
            pixel=image_hashes(data,rel)
            need(raw not in original_hashes,f'Unexpected byte duplicate: {rel} and {original_hashes.get(raw)}')
            need(pixel not in pixel_hashes,f'Unexpected RGB duplicate: {rel} and {pixel_hashes.get(pixel)}')
            original_hashes[raw]=rel
            pixel_hashes[pixel]=rel
    for filename in original_docs:
        doc=root/filename
        r[filename]=(*stat_tuple(doc,'file'),hashlib.sha256(doc.read_bytes()).hexdigest())
    return r

before=snapshot_source(decode=True)
print('SOURCE_OK '+json.dumps(dict(images=len(images),bad_images=0,exact_file_duplicates=0,exact_pixel_duplicates=0,
                                    fingerprint=hashlib.sha256(json.dumps(sorted(before.items())).encode('utf-8')).hexdigest())))

NOTES="# TeaSmart \u00b7 tea-leaf-clean-v1\n\nB\u1ea3n sao dataset cho b\u00e0i to\u00e1n ph\u00e2n lo\u1ea1i \u0111\u01a1n nh\u00e3n 7 l\u1edbp. C\u00e1c t\u1ec7p \u1ea3nh gi\u1eef nguy\u00ean byte, t\u00ean t\u1ec7p, t\u00ean nh\u00e3n v\u00e0 t\u1eadp theo b\u1ea3n xu\u1ea5t.\n\n## Ngu\u1ed3n v\u00e0 gi\u1ea5y ph\u00e9p\n\n- B\u1ea3n ngu\u1ed3n **My First Project v1**, xu\u1ea5t ng\u00e0y **30/05/2026**. Hai README \u0111i k\u00e8m ngu\u1ed3n ghi gi\u1ea5y ph\u00e9p **CC BY 4.0**, Auto-Orient, Stretch 224\u00d7224 v\u00e0 kh\u00f4ng \u00e1p d\u1ee5ng augmentation.\n- [Roboflow Universe \u00b7 My First Project](https://universe.roboflow.com/rohithdraj810-gmail-com/my-first-project-yklen) l\u00e0 d\u1ef1 \u00e1n ghi trong README c\u1ee7a b\u1ea3n xu\u1ea5t.\n- T\u00ean l\u1edbp, m\u1eabu t\u00ean t\u1ec7p v\u00e0 d\u00e3y ID ph\u00f9 h\u1ee3p v\u1edbi [teaLeafBD v4 c\u1ee7a Alam v\u00e0 c\u1ed9ng s\u1ef1](https://data.mendeley.com/datasets/744vznw5k2/4), DOI 10.17632/744vznw5k2.4, CC BY 4.0. \u0110\u00e2y l\u00e0 ngu\u1ed3n g\u1ed1c c\u00f3 kh\u1ea3 n\u0103ng cao theo \u0111\u1ed1i chi\u1ebfu giai \u0111o\u1ea1n 2B. Ch\u01b0a c\u00f3 l\u1ecbch s\u1eed xu\u1ea5t \u0111\u1ec3 x\u00e1c nh\u1eadn ngu\u1ed3n c\u1ee7a t\u1eebng \u1ea3nh.\n\n## Quy tr\u00ecnh sao ch\u00e9p\n\n1. Ki\u1ec3m tra danh s\u00e1ch c\u1eb7p, nh\u00f3m v\u00e0 quy\u1ebft \u0111\u1ecbnh theo \u1ea3nh c\u1ee7a giai \u0111o\u1ea1n 2B \u0111\u1ed1i v\u1edbi to\u00e0n b\u1ed9 5.275 t\u1ec7p ngu\u1ed3n.\n2. Gi\u1eef \u0111\u00fang 5.228 d\u00f2ng `KEEP`. B\u1ecf qua 34 \u1ea3nh g\u1ea7n tr\u00f9ng gi\u1eefa c\u00e1c t\u1eadp v\u00e0 2 \u1ea3nh l\u1eb7p n\u1ed9i b\u1ed9 trong t\u1eadp \u0111\u00e1nh gi\u00e1. Ghi ri\u00eang 11 d\u00f2ng c\u00e1ch ly Train trong `exclusions.csv`. M\u1ecdi \u1ea3nh b\u1ecb lo\u1ea1i hay c\u00e1ch ly v\u1eabn n\u1eb1m trong th\u01b0 m\u1ee5c ngu\u1ed3n.\n3. Gi\u1eef t\u1eadp theo th\u1ee9 t\u1ef1 \u01b0u ti\u00ean Test > Valid > Train cho nh\u00f3m li\u00ean t\u1eadp \u0111\u00e3 nh\u1eadn di\u1ec7n. M\u1ed7i nh\u00f3m trong Valid v\u00e0 Test c\u00f2n m\u1ed9t \u0111\u1ea1i di\u1ec7n c\u00f3 b\u1eb1ng ch\u1ee9ng m\u1ea1nh. Ki\u1ec3m tra l\u1ea1i 50 nh\u00f3m c\u00f9ng l\u00e1 c\u00f3 kh\u1ea3 n\u0103ng cao v\u00e0 9 nh\u00f3m c\u1ea7n ph\u00f2ng ng\u1eeba theo d\u1eef li\u1ec7u \u0111\u00e3 duy\u1ec7t.\n4. Gi\u1ea3i m\u00e3 to\u00e0n b\u1ed9 \u1ea3nh b\u1ea3n sao v\u00e0 so s\u00e1nh SHA-256 t\u1eebng t\u1ec7p v\u1edbi ngu\u1ed3n. So SHA-256 c\u1ee7a t\u1ec7p v\u00e0 c\u1ee7a pixel RGB gi\u1eefa c\u00e1c t\u1eadp. Ki\u1ec3m tra l\u1ea1i t\u1eadp t\u1ec7p, th\u01b0 m\u1ee5c, th\u1eddi gian s\u1eeda \u0111\u1ed5i v\u00e0 SHA-256 \u1edf ngu\u1ed3n tr\u01b0\u1edbc v\u00e0 sau thao t\u00e1c.\n5. Gi\u1eef nguy\u00ean 7 th\u01b0 m\u1ee5c l\u1edbp trong m\u1ed7i t\u1eadp. B\u1ea3n sao kh\u00f4ng c\u00f3 \u1ea3nh augmentation, t\u1ec7p c\u00e1ch ly ho\u1eb7c \u1ea3nh \u0111\u01b0\u1ee3c chuy\u1ec3n sang t\u1eadp kh\u00e1c.\n\n## Nh\u1eadt k\u00fd\n\n- [manifest.csv](manifest.csv) ghi \u0111\u01b0\u1eddng d\u1eabn ngu\u1ed3n v\u00e0 \u0111\u00edch, nh\u00e3n, t\u1eadp, nh\u00f3m n\u1ebfu c\u00f3 v\u00e0 SHA-256 t\u1eebng \u1ea3nh \u0111\u01b0\u1ee3c gi\u1eef.\n- [exclusions.csv](exclusions.csv) ghi tr\u1ea1ng th\u00e1i lo\u1ea1i ho\u1eb7c c\u00e1ch ly, m\u00e3 quy\u1ebft \u0111\u1ecbnh, l\u00fd do v\u00e0 nh\u00f3m t\u1eebng \u1ea3nh.\n- [dataset_summary.csv](dataset_summary.csv) ghi s\u1ed1 l\u01b0\u1ee3ng \u0111\u00e3 \u0111\u1ed1i chi\u1ebfu theo l\u1edbp v\u00e0 t\u1eadp.\n\n## Gi\u1edbi h\u1ea1n khi \u0111\u00e1nh gi\u00e1\n\nM\u01b0\u1eddi c\u1eb7p t\u1eebng x\u1ebfp nghi v\u1ea5n \u1edf giai \u0111o\u1ea1n 2B \u0111\u1ec1u \u0111\u00e3 \u0111\u01b0\u1ee3c t\u00e1ch kh\u1ecfi vi\u1ec7c c\u00f9ng xu\u1ea5t hi\u1ec7n \u1edf hai t\u1eadp. C\u00e1c c\u1eb7p x\u1ebfp \u201cch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9\u201d c\u00f2n trong b\u1ea3n sao v\u1eabn mang m\u1ee9c b\u1eb1ng ch\u1ee9ng \u0111\u00f3, k\u1ec3 c\u1ea3 khi \u0111i\u1ec3m t\u01b0\u01a1ng \u0111\u1ed3ng t\u1ed5ng th\u1ec3 cao. D\u1eef li\u1ec7u thi\u1ebfu \u0111\u1ecbnh danh theo l\u00e1, c\u00e2y, v\u01b0\u1eddn v\u00e0 \u0111\u1ee3t ch\u1ee5p, n\u00ean t\u00ednh \u0111\u1ed9c l\u1eadp ho\u00e0n to\u00e0n c\u1ee7a Test ch\u01b0a \u0111\u01b0\u1ee3c x\u00e1c minh.\n\n## D\u1ea5u v\u1ebft ki\u1ec3m tra"
ORIGINAL="Dataset g\u1ed1c"
APPROVED_AUDIT="B\u00e1o c\u00e1o \u0111\u00e3 duy\u1ec7t"
PHOTO_COUNTS="S\u1ed1 l\u01b0\u1ee3ng x\u00e1c minh"
REMAINING="C\u1eb7p s\u00e0ng l\u1ecdc 2B c\u00f2n c\u1ea3 hai \u1ea3nh \u1edf hai t\u1eadp v\u00e0 ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 k\u1ebft lu\u1eadn c\u00f9ng l\u00e1"
REMAINING_TEST="Trong \u0111\u00f3 s\u1ed1 c\u1eb7p c\u00f3 \u1ea3nh Test"
FINGERPRINT="SHA-256 t\u1ed5ng h\u1ee3p c\u1ee7a \u1ea3nh, hai README g\u1ed1c v\u00e0 c\u00e2y th\u01b0 m\u1ee5c ngu\u1ed3n tr\u01b0\u1edbc/sau"

if not write:
    raise ValueError('Copy command requires copy mode')
need(os.environ['TEASMART_2C_EXPECTED_FP']==hashlib.sha256(json.dumps(sorted(before.items())).encode('utf-8')).hexdigest(),
     'Original dataset changed between review and copy')
source_readme=(root/'README.dataset.txt').read_text(encoding='utf-8-sig')
source_roboflow=(root/'README.roboflow.txt').read_text(encoding='utf-8-sig')
need('CC BY 4.0' in source_readme and 'my-first-project-yklen' in source_readme
     and 'My First Project - v1' in source_roboflow and '5275' in source_roboflow,'Source provenance files differ')
require_absent(dest)
dest.mkdir(exist_ok=False)
safe_stat(dest,'dir')
for fold in folds:
    (dest/fold).mkdir()
    for label in labels:
        (dest/fold/label).mkdir()

for rel in sorted(keep):
    target=dest/rel
    data=images[rel].read_bytes()
    need(hashlib.sha256(data).hexdigest()==before[rel][-1],f'Source modified during copy: {rel}')
    with target.open('xb') as f:
        chars=f.write(data)
        need(chars==len(data),f'Incomplete write: {rel}')

copied=scan_layout(dest,(),approved)
need(set(copied)==keep,'Destination filenames do not match approved rows')
pixel_digests={}
raw_digests={}
for rel,target in copied.items():
    data=target.read_bytes()
    raw=hashlib.sha256(data).hexdigest()
    need(raw==before[rel][-1],f'Byte verification failed: {rel}')
    pixel=image_hashes(data,rel)
    need(raw not in raw_digests,f'Exact copy duplicate detected: {rel}')
    need(pixel not in pixel_digests,f'Decoded pixel duplicate detected: {rel}')
    raw_digests[raw]=rel
    pixel_digests[pixel]=rel
for gid,paths in group_members.items():
    present=paths & copied.keys()
    surviving_splits={m[x]['original_split'] for x in present}
    need(len(surviving_splits)<=1,f'Near-duplicate group unexpectedly crosses splits after copy: {gid}')
    if gid.startswith('G') and surviving_splits & {'valid','test'}:
        need(len(present)==1,f'Evaluation group unexpectedly has more than one representative: {gid}')
for row in p.values():
    if row['review'] in (high,suspect) and row['split_a']!=row['split_b']:
        need(not {row['path_a'],row['path_b']}<=copied.keys(),f'Cross-split reviewed pair unexpectedly remains: {row["id"]}')
after=snapshot_source()
need(before==after,'Source dataset changed during copy or verification')
fingerprint=hashlib.sha256(json.dumps(sorted(after.items())).encode('utf-8')).hexdigest()
print('COPIES_OK '+json.dumps(dict(count=len(copied),bytes=sum(p.stat().st_size for p in copied.values()),
     sha256_matches=len(raw_digests),bad_images=0,exact_file_duplicates=0,exact_pixel_duplicates=0,
     verified_group_counts=[50,9], reviewed_cross_split_pairs_remaining=0, source_unchanged=True,source_fingerprint=fingerprint)))

def output_csv(filename,rows):
    with (dest/filename).open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
def read_own_csv(filename):
    with (dest/filename).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

manifest=[]
exclusions=[]
for rel,row in sorted(m.items()):
    common=dict(split=row['original_split'],label=row['original_label'],relative_path=rel,
                source_path=str(images[rel]),sha256=before[rel][-1],same_leaf_group=row['same_leaf_group'],
                precaution_group=row['precaution_group'])
    if rel in copied:
        manifest.append(dict(relative_path=rel,**{k:v for k,v in common.items() if k!='relative_path'}))
    else:
        action=row['proposed_action']
        exclusions.append(dict(**common,status='QUARANTINED' if action=='QUARANTINE_PENDING_REVIEW' else 'EXCLUDED',
                        action=action,reason=row['reason']))
summary=[]
for label in (*labels,'TOTAL'):
    sizes=tuple(sum(PurePosixPath(x).parts[:2]==(f,label) if label!='TOTAL' else PurePosixPath(x).parts[0]==f
                    for x in copied) for f in folds)
    summary.append(dict(label=label,train=sizes[0],valid=sizes[1],test=sizes[2],total=sum(sizes)))
output_csv('manifest.csv',manifest)
output_csv('exclusions.csv',exclusions)
output_csv('dataset_summary.csv',summary)

with (dest/'README.md').open('x',encoding='utf-8') as file:
    file.write(NOTES+'\n\n')
    file.write('- '+ORIGINAL+f' \x60{root}\x60.\n')
    file.write('- '+APPROVED_AUDIT+f' \x60{audit}\x60.\n')
    file.write('- '+PHOTO_COUNTS+' 3.647 Train / 1.051 Valid / 530 Test / 5.228.\n')
    file.write('- '+REMAINING+f' {len(remaining)}. '+REMAINING_TEST+
               f' {sum(r["split_a"]=="test" or r["split_b"]=="test" for r in remaining)}.\n')
    file.write('- '+FINGERPRINT+f' \x60{fingerprint}\x60.\n')
    for name,hashvalue in sorted(evidence_hashes.items()):
        file.write(f'- SHA-256 giai doan 2B \x60{name}\x60 \x60{hashvalue}\x60.\n')

logged=index(read_own_csv('manifest.csv'),'relative_path','destination manifest')
skipped=index(read_own_csv('exclusions.csv'),'relative_path','destination exclusions')
need(set(logged)==set(copied) and set(skipped)==set(m)-keep,'Audit manifests differ from actual inclusions and exclusions')
need(all(logged[x]['sha256']==before[x][-1] and logged[x]['label']==m[x]['original_label']
         and logged[x]['split']==m[x]['original_split'] and logged[x]['source_path']==str(images[x]) for x in logged),'Written manifest validation failed')
need(Counter(r['status'] for r in skipped.values())=={'EXCLUDED':36,'QUARANTINED':11},'Written exclusions totals differ')
need(read_own_csv('dataset_summary.csv')[-1]==dict(label='TOTAL',train='3647',valid='1051',test='530',total='5228'),'Written summary totals differ')
scan_layout(dest,('manifest.csv','exclusions.csv','dataset_summary.csv','README.md'),approved)
print('OUTPUT_OK '+json.dumps(dict(summaries=summary,exclusions=len(exclusions),manifests=len(manifest),
     files=['manifest.csv','exclusions.csv','dataset_summary.csv','README.md'])))
