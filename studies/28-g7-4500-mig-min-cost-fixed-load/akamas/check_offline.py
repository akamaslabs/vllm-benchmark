"""Offline checks of study 28's Akamas YAML against the repo rules and the local pack checkouts.

Usage: python3 akamas/check_offline.py [--packs ~/akamas/offline/optimization-packs]
Exit 0 if every check passes; prints one line per failure otherwise.
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
PACKS = os.path.expanduser(sys.argv[sys.argv.index('--packs') + 1] if '--packs' in sys.argv
                           else '~/akamas/offline/optimization-packs')
fails = []


def load(p):
    with open(p) as f:
        return yaml.safe_load(f)


components = {d['name']: d for d in map(load, glob.glob(os.path.join(HERE, 'components', '*.yaml')))}
for name in components:
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        fails.append('component name %s' % name)

# Pack component types: name -> {parameter: domain}; (type, parameter) -> FileConfigurator confTemplate
ctypes, conftpl = {}, {}
for p in glob.glob(os.path.join(PACKS, '*', 'component-types', '*.yaml')):
    d = load(p)
    ctypes[d['name']] = {x['name']: x.get('domain', {}) for x in d.get('parameters', [])}
    for x in d.get('parameters', []):
        t = ((x.get('operators') or {}).get('FileConfigurator') or {}).get('confTemplate')
        if t:
            conftpl[(d['name'], x['name'])] = t

TOKEN = r'\$\{([a-z0-9_]+\.[a-z0-9_]+)\}'
env_template = open(os.path.join(STUDY, 'k8s', 'params.env.template')).read()
tokens = set(re.findall(TOKEN, env_template))


def render_step(study_name, st):
    """What the FileConfigurator writes for this step (values through the pack's confTemplates),
    fed to the real render_statefulset.sh: every baseline/preset must render."""
    def sub(m):
        comp, par = m.group(1).split('.')
        v = str(st['values'][m.group(1)])
        t = conftpl.get((components[comp]['componentType'], par))
        return t.replace('${value}', v) if t else v
    try:
        env = re.sub(TOKEN, sub, env_template)
    except KeyError:
        return  # reported by the "renders every parameter" check
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, 'params.env'), 'w').write(env)
        r = subprocess.run(['bash', os.path.join(STUDY, 'k8s', 'render_statefulset.sh'), os.path.join(td, 'params.env'),
                            os.path.join(STUDY, 'k8s', '01-statefulset_template.yaml'), os.path.join(td, 'sts.yaml'), '1'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails.append('%s: step %r does not render as the FileConfigurator writes it: %s'
                         % (study_name, st['name'], r.stderr.strip()))
for sp in glob.glob(os.path.join(HERE, '28-*.yaml')):
    s = load(sp)
    if s.get('kind') != 'study':
        continue
    sel = {x['name']: x for x in s['parametersSelection']}
    for t in sorted(tokens - set(sel)):
        fails.append('%s: template token %s not in parametersSelection' % (s['name'], t))
    for pname, x in sel.items():
        comp, par = pname.split('.')
        ctype = components[comp]['componentType']
        dom = ctypes.get(ctype, {}).get(par)
        if dom is None:
            fails.append('%s: %s not a parameter of %s' % (s['name'], pname, ctype))
            continue
        if 'categories' in x:
            extra = set(map(str, x['categories'])) - set(map(str, dom.get('categories', [])))
            if extra:
                fails.append('%s: %s categories %s not in the pack' % (s['name'], pname, sorted(extra)))
        elif 'domain' in x:
            lo, hi = dom['domain']
            if not (lo <= x['domain'][0] <= x['domain'][1] <= hi):
                fails.append('%s: %s domain %s outside the pack %s' % (s['name'], pname, x['domain'], dom['domain']))
    for st in s.get('steps', []):
        if not re.match(r'^[a-zA-Z\s][a-zA-Z0-9_\s]*$', st['name']):
            fails.append('%s: step name %r' % (s['name'], st['name']))
        if st.get('type') in ('baseline', 'preset') and set(st.get('values', {})) != set(sel):
            fails.append('%s: step %r does not render every parameter' % (s['name'], st['name']))
        if st.get('type') in ('baseline', 'preset'):
            render_step(s['name'], st)
    if len(s.get('kpis', [])) > 8:
        fails.append('%s: %d KPIs (max 8)' % (s['name'], len(s['kpis'])))
    fa = 'FLASH_ATTN' in map(str, sel.get('vllm.attention_backend', {}).get('categories', []))
    has_c = any('FLASH_ATTN' in c['formula'] for c in s.get('parameterConstraints', []))
    if fa != has_c:
        fails.append('%s: FLASH_ATTN in the domain (%s) but FLASH_ATTN/fp8 constraint present (%s)' % (s['name'], fa, has_c))

tel = open(os.path.join(HERE, 'telemetry', 'prometheus.yaml')).read()
keys = set(re.findall(r'\$([A-Za-z_]+)\$', tel)) - {'DURATION'}
for k in sorted(keys):
    if '_' in k:
        fails.append('telemetry placeholder $%s$ has an underscore' % k)
for line in fails:
    print('FAIL ' + line)
print('%d failure(s)' % len(fails))
sys.exit(1 if fails else 0)
