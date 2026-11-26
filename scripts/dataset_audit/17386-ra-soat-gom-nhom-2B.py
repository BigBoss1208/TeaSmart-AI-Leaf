
import csv, json, os
from collections import Counter,defaultdict
from pathlib import Path
root=Path(os.environ['TEASMART_DATASET_PATH']).resolve()
out=Path(os.environ['TEASMART_LEAKAGE_OUTPUT']).resolve()
assert not out.is_relative_to(root) and not root.is_relative_to(out)
HIGH=set([4,6,11,13,14,17,40,50,62,63,65,78,87,99,130,154,170,179,198])
HOLD=set([64,68,70,82,103,107,109,137,175,177])
NOTES=json.loads("{\"1\":\"N\u1ec1n v\u00e0 b\u1ed1 c\u1ee5c gi\u1ed1ng nhau nh\u01b0ng \u1ea3nh tr\u00e1i l\u00e0 l\u00e1 nguy\u00ean, \u1ea3nh ph\u1ea3i cu\u1ed9n m\u00e9p v\u00e0 c\u00f3 v\u1ebft. Nh\u00e3n kh\u00e1c nhau kh\u00f4ng \u0111\u1ee7 \u0111\u1ec3 t\u1ef1 s\u1eeda nh\u00e3n.\",\"3\":\"C\u00f9ng n\u1ec1n gi\u1ea5y v\u00e0 b\u1ed1 c\u1ee5c nh\u1ecf gi\u1eefa \u1ea3nh. D\u00e1ng vi\u1ec1n v\u00e0 n\u1ebfp l\u00e1 kh\u00e1c khi xem k\u00edch th\u01b0\u1edbc g\u1ed1c.\",\"4\":\"C\u00f9ng m\u1ea3ng n\u00e2u, khuy\u1ebft tr\u1eafng \u1edf m\u00e9p ph\u1ea3i v\u00e0 v\u1ecb tr\u00ed g\u00e2n.\",\"6\":\"C\u00f9ng h\u00ecnh l\u00e1 r\u1ed9ng, m\u00e9p r\u00e1ch v\u00e0 d\u00e3y v\u1ec7t nh\u1ea1t theo g\u00e2n.\",\"11\":\"C\u00f9ng ch\u00f3p cong, g\u00e2n, v\u1ebft ch\u1ea5m n\u00e2u v\u00e0 d\u1ea5u tr\u1eafng gi\u1eefa l\u00e1. Khung h\u00ecnh/\u0111\u1ed9 s\u00e1ng kh\u00e1c.\",\"13\":\"C\u00f9ng vi\u1ec1n l\u00f5m \u1edf \u0111\u1ec9nh v\u00e0 c\u1ee5m v\u1ebft n\u00e2u qua ph\u1ea7n tr\u00ean.\",\"14\":\"C\u00f9ng m\u00e9p x\u1ebb s\u00e2u v\u00e0 c\u1ee5m v\u1ebft quanh g\u00e2n v\u1edbi M040/I004.\",\"17\":\"C\u00f9ng m\u00e9p ph\u1ea7n l\u00e1 khuy\u1ebft, v\u1ec7t n\u00e2u c\u1ea1nh g\u00e2n v\u00e0 t\u01b0 th\u1ebf gi\u1eef l\u00e1.\",\"40\":\"C\u00f9ng h\u00ecnh m\u00e9p x\u1ebb, g\u00e2n v\u00e0 c\u1ee5m \u0111\u1ed1m v\u1edbi M014/I004.\",\"50\":\"C\u00f9ng v\u1ebft t\u1eadp trung \u1edf ph\u1ea7n d\u01b0\u1edbi v\u00e0 \u0111i\u1ec3m khuy\u1ebft \u1edf m\u00e9p.\",\"62\":\"C\u00f9ng m\u00e9p b\u1ecb x\u00e9, h\u00ecnh nh\u00e1nh g\u00e2n v\u00e0 m\u1ea3ng t\u1ed5n th\u01b0\u01a1ng.\",\"63\":\"C\u00f9ng h\u00ecnh vi\u1ec1n, g\u00e2n v\u00e0 m\u1ea3ng n\u00e2u khuy\u1ebft c\u1ea1nh d\u01b0\u1edbi.\",\"64\":\"D\u00e1ng l\u00e1 t\u01b0\u01a1ng t\u1ef1 nh\u01b0ng g\u00f3c v\u00e0 \u0111\u01b0\u1eddng g\u00e2n ch\u01b0a \u0111\u1ee7 \u0111\u1eb7c tr\u01b0ng.\",\"65\":\"C\u00f9ng \u0111\u01b0\u1eddng c\u1eaft m\u00e9p tr\u00ean v\u00e0 v\u00f9ng v\u1ebft d\u1ecdc g\u00e2n gi\u1eefa.\",\"68\":\"C\u00f9ng h\u01b0\u1edbng l\u00e1 v\u00e0 c\u1ee5m \u0111\u1ed1m, kh\u00e1c v\u1ecb tr\u00ed m\u00e9p/\u0111\u1ed9 d\u00e0i. C\u00f9ng \u1ea3nh Valid v\u1edbi M109.\",\"70\":\"Hai l\u00e1 \u0111\u1ec1u g\u1eadp, kh\u00e1c t\u01b0 th\u1ebf v\u00e0 \u00e1nh s\u00e1ng. Ch\u01b0a \u0111\u1ed1i chi\u1ebfu \u0111\u01b0\u1ee3c c\u00e1c m\u1ed1c t\u1ed5n th\u01b0\u01a1ng.\",\"78\":\"C\u00f9ng \u0111\u01b0\u1eddng g\u00e2n ch\u00ednh, h\u00ecnh m\u00e9p l\u00e1 v\u00e0 v\u1ecb tr\u00ed cu\u1ed1ng.\",\"82\":\"Hai l\u00e1 n\u1eb1m ng\u01b0\u1ee3c h\u01b0\u1edbng, c\u1ea7n xem l\u1ea1i cu\u1ed1ng v\u00e0 v\u1ecb tr\u00ed t\u1ed5n th\u01b0\u01a1ng.\",\"87\":\"C\u00f9ng c\u1ea1nh tr\u00ean b\u1ecb khuy\u1ebft, v\u00f9ng n\u00e2u v\u00e0 h\u00ecnh g\u00e2n.\",\"99\":\"C\u00f9ng m\u00e9p l\u01b0\u1ee3n nhi\u1ec1u n\u1ea5c v\u00e0 \u0111\u1ea7u d\u01b0\u1edbi qu\u1eb7p. Hai \u1ea3nh kh\u00e1c \u00e1nh s\u00e1ng.\",\"103\":\"C\u1ee5m v\u1ebft tr\u00ean l\u00e1 t\u01b0\u01a1ng t\u1ef1 nh\u01b0ng vi\u1ec1n v\u00e0 g\u00f3c kh\u00e1c. N\u1ed1i v\u1edbi nh\u00f3m \u0111\u00e3 x\u00e1c nh\u1eadn c\u1ee7a M130.\",\"107\":\"H\u00ecnh l\u00e1 r\u00e1ch v\u00e0 ph\u1ea7n s\u1eabm ph\u00eda d\u01b0\u1edbi t\u01b0\u01a1ng t\u1ef1. N\u1ed1i v\u1edbi nh\u00f3m \u0111\u00e3 x\u00e1c nh\u1eadn c\u1ee7a I013.\",\"109\":\"C\u00f9ng v\u00f9ng \u0111\u1ed1m d\u00e0y, h\u00ecnh l\u00e1 kh\u00e1c g\u00f3c. C\u00f9ng \u1ea3nh Valid v\u1edbi M068.\",\"130\":\"C\u00f9ng m\u00e9p r\u00e1ch \u1edf \u0111\u1ec9nh v\u00e0 c\u1ee5m ch\u1ea5m n\u00e2u quanh g\u00e2n.\",\"137\":\"L\u00e1 t\u1ed1i v\u00e0 qu\u0103n t\u01b0\u01a1ng t\u1ef1, kh\u00e1c g\u00f3c v\u00e0 ch\u01b0a th\u1ea5y m\u1ed1c g\u00e2n \u0111\u1ee7 r\u00f5.\",\"154\":\"C\u00f9ng \u0111\u1ea7u l\u00e1 cu\u1ed9n, g\u00e2n g\u1ea5p v\u00e0 kho\u1ea3ng khuy\u1ebft theo m\u00e9p.\",\"170\":\"C\u00f9ng m\u00e9p tr\u00f2n c\u00f3 n\u1ea5c, d\u00e3y \u0111\u1ed1m v\u00e0 ch\u00f3p nh\u1ecf b\u1ecb h\u1ea1i.\",\"175\":\"Nhi\u1ec1u n\u1ebfp g\u1eadp t\u01b0\u01a1ng t\u1ef1, ch\u01b0a x\u00e1c \u0111\u1ecbnh \u0111\u01b0\u1ee3c c\u00e1c v\u1ebft t\u1ed5n th\u01b0\u01a1ng t\u01b0\u01a1ng \u1ee9ng.\",\"177\":\"C\u00f9ng d\u1ea1ng l\u00e1 h\u1eb9p, thi\u1ebfu d\u1ea5u ri\u00eang bi\u1ec7t \u0111\u1ec3 x\u00e1c nh\u1eadn.\",\"179\":\"C\u00f9ng h\u00ecnh cu\u1ed1ng cong, ch\u00f3p v\u00e0 m\u1eabu \u0111\u1ed1m \u0111en quanh g\u00e2n.\",\"198\":\"Sau khi xoay \u1ea3nh ph\u1ea3i 90\u00b0, h\u00ecnh m\u00e9p, g\u00e2n v\u00e0 c\u1ee5m ch\u1ea5m s\u00e1ng t\u01b0\u01a1ng \u1ee9ng.\",\"220\":\"H\u00ecnh vi\u1ec1n v\u00e0 cu\u1ed1ng kh\u00e1c r\u00f5 khi xem k\u00edch th\u01b0\u1edbc g\u1ed1c.\"}")
COMMON_S="Kh\u1ea3 n\u0103ng r\u1ea5t cao c\u00f9ng l\u00e1. H\u00ecnh vi\u1ec1n, g\u00e2n v\u00e0 v\u1ecb tr\u00ed h\u01b0 t\u1ed5n/b\u1ed1 c\u1ee5c tr\u00f9ng khi \u0111\u1ed1i chi\u1ebfu to\u00e0n \u1ea3nh."
COMMON_I="Kh\u1ea3 n\u0103ng r\u1ea5t cao c\u00f9ng l\u00e1 trong m\u1ed9t t\u1eadp. D\u00e1ng l\u00e1, g\u00e2n, m\u00e9p v\u00e0 v\u1ebft t\u01b0\u01a1ng \u1ee9ng."
DEFAULT_M="Ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 x\u00e1c \u0111\u1ecbnh c\u00f9ng l\u00e1 sau r\u00e0 tr\u1ef1c quan. \u0110i\u1ec3m t\u01b0\u01a1ng \u0111\u1ed3ng to\u00e0n \u1ea3nh c\u00f3 th\u1ec3 ch\u1ecbu \u1ea3nh h\u01b0\u1edfng c\u1ee7a n\u1ec1n, \u00e1nh s\u00e1ng v\u00e0 b\u1ed1 c\u1ee5c."
CONFIRMED_REASON="\u1ea2nh g\u1ea7n tr\u00f9ng thu\u1ed9c nh\u00f3m c\u00f3 \u1ea3nh \u1edf t\u1eadp \u01b0u ti\u00ean cao h\u01a1n. Lo\u1ea1i \u1ea3nh n\u00e0y kh\u1ecfi b\u1ea3n sao d\u1ef1 ki\u1ebfn."
EVAL_REASON="Nh\u00f3m g\u1ea7n tr\u00f9ng trong t\u1eadp \u0111\u00e1nh gi\u00e1. Gi\u1eef m\u1ed9t \u1ea3nh \u0111\u1ea1i di\u1ec7n theo th\u1ee9 t\u1ef1 t\u00ean t\u1ec7p."
PRECAUTION_REASON="C\u1eb7p nghi v\u1ea5n b\u1eafc c\u1ea7u gi\u1eefa c\u00e1c t\u1eadp. T\u1ea1m gi\u1eef \u1ea3nh c\u1ee7a t\u1eadp \u01b0u ti\u00ean cao h\u01a1n v\u00e0 c\u00e1ch ly \u1ea3nh n\u00e0y kh\u1ecfi d\u1eef li\u1ec7u d\u00f9ng \u0111\u1ec3 hu\u1ea5n luy\u1ec7n/\u0111\u00e1nh gi\u00e1."
RETAIN_REASON="Kh\u00f4ng thu\u1ed9c nh\u00f3m c\u00f3 \u0111\u1ec1 xu\u1ea5t lo\u1ea1i ho\u1eb7c c\u00e1ch ly trong l\u1ea7n r\u00e0 so\u00e1t n\u00e0y."
PREFERRED_REASON="Gi\u1eef trong t\u1eadp hi\u1ec7n t\u1ea1i theo \u01b0u ti\u00ean Test > Valid > Train ho\u1eb7c gi\u1eef c\u00e1c \u1ea3nh c\u00f9ng nh\u00f3m n\u1ed9i b\u1ed9 Train."

def write_csv(filename, data):
    assert all(isinstance(row,dict) for row in data)
    with (out/filename).open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def parse_pair(item):
    return (item['path_a'],item['path_b'])

def split(path):
    return path.split('/',1)[0]

def components(pair_list):
    parent={}
    def find(node):
        while parent.get(node,node)!=node:
            node=parent[node]
        return node
    for edge in pair_list:
        a,b=parse_pair(edge)
        if find(a)!=find(b):
            parent[find(b)]=find(a)
    graph=defaultdict(set)
    aliases=defaultdict(list)
    for edge in pair_list:
        key=find(edge['path_a'])
        graph[key].update(parse_pair(edge))
        aliases[key].append(edge['id'])
    return [(graph[k],sorted(aliases[k])) for k in graph]

def abbrev(path):
    return split(path)+'/'+Path(path).name.split('_jpg.rf.')[0]

with (out/'cap-anh.csv').open(encoding='utf-8-sig',newline='') as f:
    pairs=list(csv.DictReader(f))
required={'S':15,'M':225,'I':18}
assert Counter(p['id'][0] for p in pairs)==required and len({p['id'] for p in pairs})==258
assert all((out/p['sheet']).is_file() and all((root/x).is_file() for x in parse_pair(p)) for p in pairs)
high_links, hold_links=[],[]
for p in pairs:
    prefix=p['id'][0]
    n=int(p['id'][1:])
    if prefix in ('S','I') or n in HIGH:
        p['review']='VERY_LIKELY_SAME_LEAF'
        p['review_note']=COMMON_S if prefix=='S' else COMMON_I if prefix=='I' else NOTES[str(n)]
        high_links.append(p)
    elif n in HOLD:
        p['review']='SUSPECT_REVIEW'
        p['review_note']=NOTES[str(n)]
        hold_links.append(p)
    else:
        p['review']='INSUFFICIENT_EVIDENCE'
        p['review_note']=NOTES.get(str(n),DEFAULT_M)

high_clusters=sorted(components(high_links),key=lambda c:(len({split(x) for x in c[0]})==1,c[1]))
high_membership={}
hold_membership={}
priority={'train':0,'valid':1,'test':2}
override={}
group_info=[]

def get_splits(members):
    return {split(x) for x in members}

for index,(members,proof) in enumerate(high_clusters,1):
    name=f'G{index:03}'
    top=max(get_splits(members),key=priority.get)
    options=sorted(x for x in members if split(x)==top)
    representative=options[0]
    for x in members:
        high_membership[x]=name
        if split(x)!=top:
            override[x]=('EXCLUDE_NEAR_DUPLICATE',CONFIRMED_REASON)
        elif top in ('valid','test') and x!=representative:
            override[x]=('EXCLUDE_EVAL_REPEAT',EVAL_REASON)
    group_info.append(dict(group_id=name,group_type='VERY_LIKELY_SAME_LEAF',splits=' | '.join(sorted(get_splits(members),key=priority.get)),
                    preferred_split=top,images=' | '.join(sorted(members)),image_count=len(members),pair_evidence=' | '.join(proof),associations='',
                    proposed_impact=' | '.join(abbrev(x) for x in sorted(members) if x in override)))

q_ids={p['id'] for p in hold_links}
safety_groups=[(members,evidence) for members,evidence in components(high_links+hold_links) if q_ids.intersection(evidence)]
for index,(members,proof) in enumerate(sorted(safety_groups,key=lambda c: sorted(q_ids.intersection(c[1]))),1):
    name=f'Q{index:03}'
    top=max(get_splits(members),key=priority.get)
    for x in members:
        hold_membership[x]=name
        if split(x)!=top and x not in override:
            override[x]=('QUARANTINE_PENDING_REVIEW',PRECAUTION_REASON)
    group_info.append(dict(group_id=name,group_type='PRECAUTION_ONLY',splits=' | '.join(sorted(get_splits(members),key=priority.get)),
                    preferred_split=top,images=' | '.join(sorted(members)),image_count=len(members),pair_evidence=' | '.join(proof),
                    associations=' | '.join(sorted({high_membership[x] for x in members if x in high_membership})),
                    proposed_impact=' | '.join(abbrev(x) for x in sorted(members) if x in override)))
for p in pairs:
    p['same_leaf_group']=high_membership.get(p['path_a'],'') if p['review']=='VERY_LIKELY_SAME_LEAF' else ''
    p['precaution_group']=hold_membership.get(p['path_a'],'') if p['review']=='SUSPECT_REVIEW' else ''

inventory=sorted(p.relative_to(root).as_posix() for fold in priority for p in (root/fold).rglob('*.jpg') if p.is_file())
assert len(inventory)==5275 and set(override).issubset(inventory)
all_counts=Counter()
keep_counts=Counter()
actions=Counter()
manifest=[]
impact=defaultdict(dict)
for x in inventory:
    fold,label,_=x.split('/')
    all_counts[(fold,label)]+=1
    choice, reason=override.get(x,('KEEP',RETAIN_REASON))
    if choice=='KEEP':
        keep_counts[(fold,label)]+=1
        if x in high_membership or x in hold_membership:
            reason=PREFERRED_REASON
    else:
        impact[choice][abbrev(x)]=f'{high_membership.get(x,"")} {hold_membership.get(x,"")}'.strip()
    actions[(choice,fold)]+=1
    manifest.append(dict(path=x,original_split=fold,original_label=label,proposed_split=fold if choice=='KEEP' else '',
            proposed_action=choice,reason=reason,same_leaf_group=high_membership.get(x,''),precaution_group=hold_membership.get(x,'')))
kept={r['path'] for r in manifest if r['proposed_action']=='KEEP'}
assert all(len(get_splits(members & kept))<=1 for members,_ in high_clusters+safety_groups)
for name,members,_ in (('group',m,p) for m,p in high_clusters):
    for fold in ('valid','test'):
        assert sum(split(x)==fold for x in members & kept)<=1
assert all(p['class_a']==p['class_b'] for p in high_links)
write_csv('cap-anh.csv',pairs)
write_csv('nhom-anh.csv',group_info)
write_csv('de-xuat-theo-anh.csv',manifest)

split_order={'train':0,'valid':1,'test':2}
def totals(counts):
    labels=sorted({label for fold,label in counts})
    return {lbl:{fold:counts[(fold,lbl)] for fold in ('train','valid','test')} for lbl in labels}
def by_split(prefix):
    x=defaultdict(Counter)
    for p in pairs:
        if p['id'].startswith(prefix):
            x[('/'.join(sorted((p['split_a'],p['split_b']),key=split_order.get)))][p['review']]+=1
    return x
results=dict(all_counts=totals(all_counts),keep=totals(keep_counts),moderate=by_split('M'),strong=by_split('S'),internal=by_split('I'),
      groups_cross=sum(len(get_splits(m))>1 for m,_ in high_clusters),groups_intra=sum(len(get_splits(m))==1 for m,_ in high_clusters),
      multi_cross=[([abbrev(x) for x in sorted(m)],p,high_membership[next(iter(m))]) for m,p in high_clusters if len(m)>2 and len(get_splits(m))>1],
      holds=[(q['group_id'],q['pair_evidence'],[abbrev(x) for x in q['images'].split(' | ')],q['proposed_impact']) for q in group_info if q['group_type']=='PRECAUTION_ONLY'],
      actions={choice:{fold:actions[(choice,fold)] for fold in priority} for choice,_ in actions},
      test_rows={a[0]:[abbrev(a[1]),abbrev(a[2]),a[3]] for a in [(p['id'],*parse_pair(p),p['review']) for p in pairs if p['id'].startswith('M') and p['split_b']=='test' and p['review']=='VERY_LIKELY_SAME_LEAF']},
      moderate_flags=dict(transforms=sum(bool(int(r['rotation_b'])) or r['flip_b']=='True' for r in pairs if r['id'].startswith('M')),
                          differing_labels=sum(r['class_a']!=r['class_b'] for r in pairs if r['id'].startswith('M')),
                          confirmed_transforms=[r['id'] for r in pairs if r['id'].startswith('M') and (bool(int(r['rotation_b'])) or r['flip_b']=='True') and r['review']=='VERY_LIKELY_SAME_LEAF']),
      impact=impact)
print(json.dumps(results))
