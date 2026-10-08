from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import quote
import gradio as gr

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'config/notebooks.json').read_text(encoding='utf-8'))
CSS=(ROOT/'assets/style.css').read_text(encoding='utf-8')

def configured(url): return bool(url and 'YOUR-' not in url)
def colab_url(project,file):
 repo=project['github_repository'].rstrip('/')
 if not repo.startswith('https://github.com/'): return ''
 path=repo.removeprefix('https://github.com/')
 return f"https://colab.research.google.com/github/{path}/blob/{quote(project.get('github_branch','main'),safe='')}/{quote(file,safe='/')}"
def card(item,project):
 url=colab_url(project,item['file'])
 launch=f'<a class="launch" href="{url}" target="_blank" rel="noopener">Open notebook in Colab →</a>' if configured(project['github_repository']) else '<span class="launch disabled">Configure GitHub URL first</span>'
 req=''.join(f'<li>{x}</li>' for x in item.get('requires',[])); out=''.join(f'<li>{x}</li>' for x in item.get('produces',[]))
 gpu='<div class="warning">Before running: choose a GPU runtime in Colab.</div>' if item['runtime']=='GPU' else ''
 write='WRITES OUTPUTS' if item.get('modifies_data') else 'READ-ONLY'
 hosted=''
 if item['id']=='hitl-review' and configured(project.get('reviewer_url','')): hosted=f'<a class="secondary-link" href="{project["reviewer_url"]}" target="_blank">Open hosted reviewer</a>'
 return f'''<article class="task-card {item['runtime'].lower()}"><div class="stage">{item['stage']}</div><h3>{item['title']}</h3><div class="badges"><span class="badge {item['runtime'].lower()}">{item['runtime']}</span><span class="badge neutral">COLAB</span><span class="badge neutral">{write}</span></div><p>{item['description']}</p>{gpu}<details><summary>Requirements and outputs</summary><div class="detail-columns"><div><strong>Requires</strong><ul>{req}</ul></div><div><strong>Produces</strong><ul>{out}</ul></div></div></details><div class="card-actions">{launch}{hosted}</div></article>'''
def render():
 p=DATA['project']; items=DATA['notebooks']
 workflow=''.join(card(x,p) for x in items if x['category']=='workflow'); analysis=''.join(card(x,p) for x in items if x['category']=='analysis')
 repo=f'<a class="tool-link" href="{p["github_repository"]}" target="_blank">Open GitHub repository</a>' if configured(p['github_repository']) else '<span class="tool-link disabled">Set GitHub repository</span>'
 hf=f'<a class="tool-link" href="{p["huggingface_space"]}" target="_blank">Open Hugging Face Space</a>' if configured(p.get('huggingface_space','')) else '<span class="tool-link disabled">Set Hugging Face Space</span>'
 status='Configured' if configured(p['github_repository']) else 'Configuration needed'
 return f'''<main class="control-centre"><header class="hero"><div><p class="eyebrow">VERSION {p['version']}</p><h1>{p['title']}</h1><p class="subtitle">Prepare → Review → Build → Train → Evaluate</p></div><div class="hero-note">The Control Centre launches work in Colab. Project data remains in Google Drive.</div></header><section class="status-grid"><div class="status"><span>Repository</span><strong>{status}</strong></div><div class="status"><span>Google Drive</span><strong>Checked inside Colab</strong></div><div class="status"><span>Current model</span><strong>Selected inside notebook</strong></div><div class="status"><span>Recommended start</span><strong>Prepare Existing Pages</strong></div></section><section><div class="section-heading"><p class="eyebrow">START HERE</p><h2>Main workflow</h2></div><div class="cards">{workflow}</div></section><section><div class="section-heading"><p class="eyebrow">OPTIONAL</p><h2>Analysis and diagnostics</h2></div><div class="cards analysis-cards">{analysis}</div></section><section class="workflow-panel"><h2>Workflow</h2><div class="workflow-line"><span>Prepare</span><b>→</b><span>Generate drafts</span><b>→</b><span>Review</span><b>→</b><span>Build</span><b>→</b><span>Forensics</span><b>→</b><span>Train</span><b>→</b><span>Evaluate</span></div></section><section class="tools"><h2>Project tools</h2><div class="tool-row">{repo}{hf}</div><p class="small">Edit <code>config/notebooks.json</code> after uploading. Do not store credentials there.</p></section><footer>Music Annotation Control Centre · Version {p['version']}</footer></main>'''
with gr.Blocks(title=DATA['project']['title'],css=CSS) as demo: gr.HTML(render())
if __name__=='__main__': demo.launch()
