"""Public pretrained CPU image classification; separate from gated Llama vision."""
import argparse, hashlib, html, io, json, time
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL='WinKawaks/vit-tiny-patch16-224'
REVISION='77d1485af66b34d4ed0fe95dbb0c60c7496f950b'
SAMPLE='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png'

def classify(model,processor,image,top_k=5):
    inputs=processor(images=image,return_tensors='pt')
    with torch.inference_mode():
        logits=model(**inputs).logits
        probability=logits.softmax(dim=-1)[0]
    if not torch.isfinite(probability).all() or not torch.allclose(probability.sum(),torch.tensor(1.),atol=1e-6):
        raise RuntimeError('Invalid predicted probability distribution')
    values,indices=probability.topk(top_k)
    labels=[{'label':model.config.id2label[int(i)],'probability':float(p)} for p,i in zip(values,indices)]
    entropy=float(-(probability*probability.clamp_min(1e-12).log()).sum())
    return labels,entropy

def run(image_path,output):
    torch.set_num_threads(1);torch.manual_seed(42)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if image_path:
        payload=Path(image_path).read_bytes();source=str(image_path)
    else:
        response=requests.get(SAMPLE,timeout=30);response.raise_for_status();payload=response.content;source=SAMPLE
    image=Image.open(io.BytesIO(payload)).convert('RGB')
    if image.width*image.height>25_000_000: raise ValueError('Image must contain at most 25 million pixels')
    image.save(out/'input.png')
    started=time.perf_counter()
    processor=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForImageClassification.from_pretrained(MODEL,revision=REVISION).eval().cpu()
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    labels,entropy=classify(model,processor,image)
    repeat,_=classify(model,processor,image)
    if labels!=repeat: raise RuntimeError('Evaluation should be deterministic')
    if any(not torch.equal(p,before[n]) for n,p in model.named_parameters()): raise RuntimeError('Inference changed model parameters')
    result={'model':MODEL,'revision':REVISION,'parameters':sum(p.numel() for p in model.parameters()),'device':'cpu','input':source,'image_sha256':hashlib.sha256(payload).hexdigest(),'top_classes':labels,'entropy_nats':entropy,'elapsed_seconds':time.perf_counter()-started,'deterministic_repeat':True,'scope':'Pretrained ImageNet classification only; not gated Llama image reasoning, autonomous agents, or calibrated confidence.'}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    bars=''.join('<div class="row"><b>'+html.escape(x['label'])+'</b><span>'+f'{x["probability"]:.1%}'+'</span><i style="width:'+str(100*x['probability'])+'%"></i></div>' for x in labels)
    report='<!doctype html><meta charset="utf-8"><title>CPU vision evidence</title><style>body{background:#f5f2e9;color:#22352f;font:17px system-ui;max-width:950px;margin:60px auto;padding:24px}h1{font-size:44px}main{display:grid;grid-template-columns:1fr 1fr;gap:35px}img{width:100%;border-radius:18px}.row{margin:22px 0;display:flex;justify-content:space-between;position:relative;padding-bottom:16px}.row i{position:absolute;bottom:0;height:5px;background:#438774;border-radius:4px}small{display:block;margin-top:35px;line-height:1.6}@media(max-width:600px){main{display:block}}</style><h1>What the model sees</h1><p>Actual CPU inference · public pretrained ViT-tiny · fixed checkpoint</p><main><img src="input.png" alt="Analyzed input"><section>'+bars+'</section></main><small>'+html.escape(result['scope'])+' Entropy: '+f'{entropy:.3f}'+' nats. Source and checkpoint hashes are recorded in result.json.</small>'
    (out/'index.html').write_text(report)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--image');parser.add_argument('--output',default='cpu-vision-output');args=parser.parse_args()
    print(json.dumps(run(args.image,args.output),indent=2))
