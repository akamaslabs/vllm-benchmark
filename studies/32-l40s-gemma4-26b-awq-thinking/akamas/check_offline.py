"""Offline checks of study 32's Akamas YAML (study 31's checks, adapted from studies 30 and 28) against the repo rules and the local pack checkouts.

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
ctypes, conftpl, cmetrics = {}, {}, {}
for p in glob.glob(os.path.join(PACKS, '*', 'component-types', '*.yaml')):
    d = load(p)
    ctypes[d['name']] = {x['name']: x.get('domain', {}) for x in d.get('parameters', [])}
    cmetrics[d['name']] = {x['name'] for x in d.get('metrics', [])}
    for x in d.get('parameters', []):
        t = ((x.get('operators') or {}).get('FileConfigurator') or {}).get('confTemplate')
        if t:
            conftpl[(d['name'], x['name'])] = t

tel = open(os.path.join(HERE, 'telemetry', 'prometheus.yaml')).read()
TOKEN = r'\$\{([a-z0-9_]+\.[a-z0-9_]+)\}'
env_template = open(os.path.join(STUDY, 'k8s', 'params.env.template')).read()
tokens = set(re.findall(TOKEN, '\n'.join(l for l in env_template.splitlines() if not l.lstrip().startswith('#'))))


def render_step(study_name, st):
    """What the FileConfigurator writes for this step (values through the pack's confTemplates;
    a parameter in doNotRenderParameters as an empty string, as Akamas renders it), fed to the
    real render_statefulset.sh: every baseline/preset must render."""
    dnr = set(st.get('doNotRenderParameters', []))
    def sub(m):
        comp, par = m.group(1).split('.')
        if m.group(1) in dnr:
            return ''
        v = str(st['values'][m.group(1)])
        t = conftpl.get((components[comp]['componentType'], par))
        return t.replace('${value}', v) if t else v
    try:
        env = re.sub(TOKEN, sub, '\n'.join(l for l in env_template.splitlines() if not l.lstrip().startswith('#')))
    except KeyError:
        return  # reported by the "renders every parameter" check
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, 'params.env'), 'w').write(env)
        r = subprocess.run(['bash', os.path.join(STUDY, 'k8s', 'render_statefulset.sh'), os.path.join(td, 'params.env'),
                            os.path.join(STUDY, 'k8s', '01-statefulset_template.yaml'), os.path.join(td, 'sts.yaml')],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails.append('%s: step %r does not render as the FileConfigurator writes it: %s'
                         % (study_name, st['name'], r.stderr.strip()))
# Study 32 is a copy of study 31 with every name changed: check that the names agree with each
# other, that the workflow runs this study's scripts and that the telemetry filters on the model
# name the StatefulSet serves.
system_name = load(os.path.join(HERE, 'system.yaml'))['name']
for name, c in components.items():
    if c.get('system') != system_name:
        fails.append('component %s: system %s, not %s' % (name, c.get('system'), system_name))
tel_doc = load(os.path.join(HERE, 'telemetry', 'prometheus.yaml'))
if tel_doc.get('system') != system_name:
    fails.append('telemetry instance: system %s, not %s' % (tel_doc.get('system'), system_name))
workflows = {}
for wp in glob.glob(os.path.join(HERE, '*.yaml')):
    w = load(wp)
    if w.get('kind') == 'workflow':
        workflows[w['name']] = w
        for t in w['tasks']:
            a = t['arguments']
            paths = [a.get('command', '')] + [a.get(k, {}).get('path', '') for k in ('source', 'target')]
            for x in paths:
                if '/studies/' in x and '/studies/%s/' % os.path.basename(STUDY) not in x:
                    fails.append('workflow %s, task %s: %s is not this study folder' % (w['name'], t['name'], x))
sts = open(os.path.join(STUDY, 'k8s', '01-statefulset_template.yaml')).read()
served = re.search(r'--served-model-name=([^"\s]+)', sts).group(1)
if components['vllm']['properties']['prometheus'].get('model') != served:
    fails.append('component vllm: model %s, but the StatefulSet serves %s'
                 % (components['vllm']['properties']['prometheus'].get('model'), served))
for sp in glob.glob(os.path.join(HERE, '32-*.yaml')):
    s = load(sp)
    if s.get('kind') != 'study':
        continue
    if s.get('system') != system_name:
        fails.append('%s: system %s, not %s' % (s['name'], s.get('system'), system_name))
    if s.get('workflow') not in workflows:
        fails.append('%s: workflow %s not in this folder' % (s['name'], s.get('workflow')))
    sel = {x['name']: x for x in s['parametersSelection']}
    for t in sorted(tokens - set(sel)):
        fails.append('%s: template token %s not in parametersSelection' % (s['name'], t))
    # The other direction: a selected parameter with no token is never applied, so the
    # optimizer would spend experiments on a no-op dimension.
    for t in sorted(set(sel) - tokens):
        fails.append('%s: %s selected but not rendered by params.env.template' % (s['name'], t))
    # Every metric the study reads: bound to the component's type and produced by the telemetry.
    cons = s['goal'].get('constraints', {})
    refs = [s['goal']['function']['formula']] + [c['formula'] for c in cons.get('absolute', []) + cons.get('relativeToBaseline', [])]
    # A relative constraint needs a step of type baseline to compare against, and a signed percentage.
    if cons.get('relativeToBaseline') and not any(st.get('type') == 'baseline' for st in s.get('steps', [])):
        fails.append('%s: relativeToBaseline constraints without a baseline step' % s['name'])
    for c in cons.get('relativeToBaseline', []):
        if not re.search(r'(<=|>=|<|>)\s*[+-]\d+(\.\d+)?%\s*$', c['formula']):
            fails.append('%s: relative constraint %r is not "<metric> <op> +/-N%%"' % (s['name'], c['formula']))
    refs += [k['formula'] for k in s.get('kpis', [])]
    refs += [s['windowing']['stability']['metric'], s['windowing']['stability']['when']['metric']]
    for ref in refs:
        for comp, met in re.findall(r'\b([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)', ref):
            if comp not in components:
                fails.append('%s: %s.%s: no component %s' % (s['name'], comp, met, comp)); continue
            ct = components[comp]['componentType']
            if met not in cmetrics.get(ct, set()):
                fails.append('%s: %s.%s not a metric of %s' % (s['name'], comp, met, ct))
            if ("- metric: %s\n" % met) not in tel:
                fails.append('%s: %s.%s not produced by the telemetry instance' % (s['name'], comp, met))
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
        dnr = set(st.get('doNotRenderParameters', []))
        # Akamas 4.1 (create, 2026-10-09): "doNotRender rules are excluding parameters that are
        # present in 'values'" is an error, so a not-rendered parameter has no value.
        if dnr & set(st.get('values', {})):
            fails.append('%s: step %r: %s both in values and doNotRenderParameters'
                         % (s['name'], st['name'], sorted(dnr & set(st.get('values', {})))))
        if st.get('type') in ('baseline', 'preset') and set(st.get('values', {})) | dnr != set(sel):
            fails.append('%s: step %r does not render every parameter' % (s['name'], st['name']))
        if st.get('type') in ('baseline', 'preset'):
            render_step(s['name'], st)
        for d in dnr:
            if d not in sel:
                fails.append('%s: step %r: doNotRenderParameters %s not a selected parameter' % (s['name'], st['name'], d))
        if st.get('from') and (dnr or st.get('renderParameters')):
            fails.append('%s: step %r: (do)NotRenderParameters with from' % (s['name'], st['name']))
    # The repeat must be the baseline again: the baseline's values, plus a value for (some of) the
    # parameters the baseline leaves unrendered (the user's choice, 2026-10-09: the repeat writes
    # every value out, the defaults vLLM resolves, so it reaches the optimizer engine).
    by_name = {st['name']: st for st in s.get('steps', [])}
    b, r = by_name.get('baseline'), by_name.get('baseline repeat')
    if b and r:
        bv, rv = b.get('values', {}), r.get('values', {})
        bd = set(b.get('doNotRenderParameters', []))
        if ({k: rv.get(k) for k in bv} != bv or not set(rv) - set(bv) <= bd
                or not set(r.get('doNotRenderParameters', [])) <= bd):
            fails.append('%s: baseline repeat is not the baseline (values differ, or it leaves out more)' % s['name'])
    if len(s.get('kpis', [])) > 8:
        fails.append('%s: %d KPIs (max 8)' % (s['name'], len(s['kpis'])))
    fa = 'FLASH_ATTN' in map(str, sel.get('vllm.attention_backend', {}).get('categories', []))
    has_c = any('FLASH_ATTN' in c['formula'] for c in s.get('parameterConstraints', []))
    if fa != has_c:
        fails.append('%s: FLASH_ATTN in the domain (%s) but FLASH_ATTN/fp8 constraint present (%s)' % (s['name'], fa, has_c))

keys = set(re.findall(r'\$([A-Za-z_]+)\$', tel)) - {'DURATION'}
for k in sorted(keys):
    if '_' in k:
        fails.append('telemetry placeholder $%s$ has an underscore' % k)
for line in fails:
    print('FAIL ' + line)
print('%d failure(s)' % len(fails))
sys.exit(1 if fails else 0)
