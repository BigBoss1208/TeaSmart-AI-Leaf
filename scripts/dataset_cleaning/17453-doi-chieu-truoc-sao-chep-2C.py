
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
