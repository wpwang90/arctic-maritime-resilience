"""Relabel existing vector figures without rerunning or changing their data."""
from collections import defaultdict
from io import BytesIO
import hashlib
import json
import logging
import math
from pathlib import Path
import re

from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
import pypdfium2 as pdfium
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

logging.getLogger('pypdf').setLevel(logging.ERROR)
HERE=Path(__file__).resolve().parent
BEFORE=HERE.parent/'63_ReaderClarity_Revision_20261101'
QA=HERE/'qa'/'figure_labels'
QA.mkdir(parents=True,exist_ok=True)
NETWORKS=[('Conventional network','#0077b6','patch'),
          ('Arctic network','#e87519','patch'),
          ('Unserved network','#a5abb0','patch')]
OTHER=[('No comparable mapped OD','#d9dddf','patch'),
       ('Restricted corridor','#c9553e','circle'),
       ('Excluded coast: country locator','#c9553e','triangle'),
       ('Closed port service','#c9553e','patch')]


def normalized(text):
    return ' '.join(text.split())


def text_groups(page):
    state={'index':-1}
    groups=defaultdict(lambda:dict(parts=[],events=[]))
    def before(op,args,cm,tm):
        if op==b'BT':state['index']+=1
    def visit(text,cm,tm,font,size):
        if not text.strip():return
        item=groups[state['index']]
        item['parts'].append(text)
        item['events'].append(dict(text=text,cm=list(cm),tm=list(tm),size=float(size)))
    page.extract_text(visitor_operand_before=before,visitor_text=visit)
    for item in groups.values():item['text']=normalized(''.join(item['parts']))
    return dict(groups),state['index']+1


def mark(canvas_,label,color,kind,x,y,size):
    canvas_.setStrokeColor(HexColor(color));canvas_.setFillColor(HexColor(color))
    if kind=='patch':canvas_.rect(x,y-1,12,6,stroke=0,fill=1)
    elif kind in ['line','dashed']:
        canvas_.setLineWidth(2)
        if kind=='dashed':canvas_.setDash(3,2)
        canvas_.line(x,y+2, x+14,y+2);canvas_.setDash()
    elif kind=='circle':canvas_.circle(x+6,y+2,3,stroke=0,fill=1)
    elif kind=='triangle':
        path=canvas_.beginPath();path.moveTo(x+2,y-1);path.lineTo(x+10,y-1)
        path.lineTo(x+6,y+6);path.close();canvas_.drawPath(path,fill=1,stroke=0)
    elif kind=='star':
        path=canvas_.beginPath()
        for k in range(10):
            angle=math.pi/2+k*math.pi/5;r=4 if k%2==0 else 1.8
            xx=x+6+r*math.cos(angle);yy=y+2+r*math.sin(angle)
            if k==0:path.moveTo(xx,yy)
            else:path.lineTo(xx,yy)
        path.close();canvas_.drawPath(path,fill=1,stroke=0)
    elif kind=='outline':
        canvas_.setLineWidth(1);canvas_.rect(x,y-1,12,6,stroke=1,fill=0)
    elif kind=='cross':
        canvas_.setLineWidth(1.1)
        canvas_.line(x+2,y-1,x+10,y+6);canvas_.line(x+2,y+6,x+10,y-1)
    canvas_.setFillColor(HexColor('#242b31'));canvas_.setFont('Helvetica',size)
    canvas_.drawString(x+20,y,label)


def footer(canvas_,width,kind):
    if kind=='main':
        entries=NETWORKS+OTHER+[
            ('Arctic network > Conventional network','#e87519','star'),
            ('Mapped stage','#303f48','outline')]
        columns=3;xs=[width*.18,width*.405,width*.63];ys=[33,21,9];size=8
        strip=43
    elif kind=='maps':
        entries=NETWORKS+OTHER
        columns=4;xs=[width*.13,width*.32,width*.51,width*.70];ys=[30,17];size=8
        strip=41
    elif kind=='curves':
        entries=[(x,c,'dashed' if x=='Unserved network' else 'line') for x,c,_ in NETWORKS]
        columns=3;xs=[width*.20,width*.43,width*.66];ys=[15];size=8
        strip=28
    elif kind=='legacy':
        entries=[(x,c,'line' if i<2 else 'patch') for i,(x,c,_) in enumerate(NETWORKS)]
        entries.append(('Closed chokepoint','#b52d42','cross'))
        columns=4;xs=[width*.12,width*.32,width*.52,width*.72];ys=[27];size=10
        strip=42
    else:raise ValueError(kind)
    canvas_.setFillColorRGB(1,1,1);canvas_.rect(0,0,width,strip,fill=1,stroke=0)
    for i,(label,color,shape) in enumerate(entries):
        mark(canvas_,label,color,shape,xs[i%columns],ys[i//columns],size)
    return (0,0,width,strip)


def relabel(source,destination,footer_kind=None,replacements=None):
    reader=PdfReader(source)
    old_numbers=re.findall(r'\d+(?:\.\d+)?',reader.pages[0].extract_text())
    writer=PdfWriter();writer.add_page(reader.pages[0]);page=writer.pages[0]
    width,height=float(page.mediabox.width),float(page.mediabox.height)
    groups,count=text_groups(page)
    chosen=set();individual=[]
    if footer_kind:
        expected={'main':9,'maps':7,'curves':3,'legacy':4}[footer_kind]
        for key,item in groups.items():
            text=item['text']
            compact=text.replace(' ','')
            if footer_kind=='legacy':
                hit=(any(term in compact for term in ['selectedconventionalconnections',
                    'selectedArcticconnections']) or compact in ['Noadmissibleroute','Closedchokepoint'])
            else:
                labels=['Traditional A','Arctic B','Unserved U']
                if footer_kind in ['main','maps']:labels+=[x[0] for x in OTHER]
                if footer_kind=='main':labels+=['B > A','Mapped stage']
                hit=text in labels
            if hit:chosen.add(key)
        assert len(chosen)==expected,(source,expected,[groups[k]['text'] for k in chosen])
    for old,new in (replacements or {}).items():
        matches=[(key,item) for key,item in groups.items() if item['text']==old]
        assert matches,(source,old)
        for key,item in matches:
            chosen.add(key);individual.append((old,new,item['events'][0]))
    assert chosen,source
    stream=ContentStream(page.get_contents(),writer)
    direct_count=sum(op==b'BT' for _,op in stream.operations)
    if direct_count!=count:
        # The archived polar figure appends four annotation Form XObjects after
        # its legend. Verify direct text-object identity before changing it.
        assert footer_kind=='legacy',(source,'unexpected nested text objects')
        direct=defaultdict(list);index=-1
        for operands,op in stream.operations:
            if op==b'BT':index+=1
            if op==b'Tj':direct[index].append(str(operands[0]))
            elif op==b'TJ':direct[index].extend(str(v) for v in operands[0] if isinstance(v,str))
        for key in chosen:
            assert normalized(''.join(direct[key])).replace(' ','')==groups[key]['text'].replace(' ',''),(source,key)
    current=-1;ops=[]
    for operands,op in stream.operations:
        if op==b'BT':current+=1
        if current in chosen and op in [b'Tj',b'TJ',b"'",b'"']:
            continue
        ops.append((operands,op))
    stream.operations=ops;page.replace_contents(stream)
    buf=BytesIO();overlay=canvas.Canvas(buf,pagesize=(width,height))
    regions=[]
    if footer_kind:regions.append(footer(overlay,width,footer_kind))
    for old,new,event in individual:
        cm,tm=event['cm'],event['tm'];size=event['size']
        x=cm[0]*tm[4]+cm[2]*tm[5]+cm[4]
        y=cm[1]*tm[4]+cm[3]*tm[5]+cm[5]
        angle=math.degrees(math.atan2(cm[1]*tm[0]+cm[3]*tm[1],cm[0]*tm[0]+cm[2]*tm[1]))
        old_width=stringWidth(old,'Helvetica',size)
        if abs(angle)>45:
            size=min(size,8)
            limit=height*.37
        else:limit=width*.92
        while stringWidth(new,'Helvetica',size)>limit:size-=.1
        assert size>=6.5,(source,new,size)
        new_width=stringWidth(new,'Helvetica',size)
        shift=(old_width-new_width)/2
        if abs(angle)<45:
            xx=max(12,min(x+shift,width-new_width-12));yy=y
        else:
            xx=x+shift*math.cos(math.radians(angle));yy=y+shift*math.sin(math.radians(angle))
        overlay.saveState();overlay.translate(xx,yy);overlay.rotate(angle)
        overlay.setFillColorRGB(.15,.15,.15);overlay.setFont('Helvetica',size)
        overlay.drawString(0,0,new);overlay.restoreState()
        if abs(angle)<45:
            regions.append((0,y-size-4,width,y+size+5))
        else:
            regions.append((min(x,xx)-size-5,min(y,yy)-30,
                            max(x,xx)+size+5,max(y+old_width*1.25,yy+new_width)+30))
    overlay.save();buf.seek(0);page.merge_page(PdfReader(buf).pages[0])
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('wb') as handle:writer.write(handle)
    # Re-serialization can change a few antialiased edge pixels by sub-visible amounts.
    def render(path):
        doc=pdfium.PdfDocument(str(path));im=doc[0].render(scale=1.5).to_pil().convert('RGB');doc.close();return im
    before,after=render(source),render(destination)
    preview=QA/(destination.stem+'.png');after.save(preview)
    diff=ImageChops.difference(before,after);outside=diff.copy();draw=ImageDraw.Draw(outside)
    for x0,y0,x1,y1 in regions:draw.rectangle((x0*1.5,(height-y1)*1.5,x1*1.5,(height-y0)*1.5),fill=(0,0,0))
    maximum=max(high for low,high in outside.getextrema())
    changed=sum(outside.convert('L').histogram()[1:])
    fraction=changed/(outside.width*outside.height)
    if maximum>16 or fraction>1e-4:outside.save(QA/(destination.stem+'_unexpected_diff.png'))
    assert maximum<=16 and fraction<=1e-4,(source,'unexpected change beyond labels',maximum,fraction)
    new_numbers=re.findall(r'\d+(?:\.\d+)?',PdfReader(destination).pages[0].extract_text())
    assert sorted(old_numbers)==sorted(new_numbers),(source,'numeric text changed')
    if destination.parent.name=='figures':after.save(destination.with_suffix('.png'))
    return dict(file=str(destination.relative_to(HERE)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                changed_text=[groups[k]['text'] for k in sorted(chosen)],numeric_text_unchanged=True,
                all_original_nontext_draw_operations_retained=True,
                pixels_outside_labels_maximum_channel_difference=maximum,
                pixels_outside_labels_changed_fraction=fraction,geometry_preserved_with_antialias_tolerance=True,
                preview=str(preview.relative_to(HERE)))


def main():
    report=[]
    report.append(relabel(BEFORE/'figures/MainFig3.pdf',HERE/'figures/MainFig3.pdf',footer_kind='main'))
    for n in range(12,21):
        source=BEFORE/'supplementary_figures'/('SupplementaryFig%02d.pdf'%n)
        dest=HERE/'supplementary_inputs'/source.name
        kind=None;replacements={}
        if n==12:replacements={'Arctic B minus traditional A (percentage points); outline = main-figure extreme case':
            'Arctic network share minus Conventional network share (percentage points); outline = selected case'}
        elif n==13:kind='curves'
        elif n==14:replacements={'First nominal scope stage with B > A':
            'First stage with Arctic network share > Conventional network share'}
        elif n in [15,16,17,18]:kind='maps'
        elif n==19:kind='legacy'
        elif n==20:replacements={
            'B minus A (percentage points)':'Arctic network share minus Conventional network share (pp)',
            'Minimum exposure at B > A (%)':'Minimum exposure for Arctic network dominance (%)',
            'ND: no qualifying trade-share crossover in the tested set. Numbers above boxes: model-years with B > A.':
            'ND: no trade-share crossover detected. Numbers above boxes count model-years with Arctic network share > Conventional network share.'}
        report.append(relabel(source,dest,footer_kind=kind,replacements=replacements))
    (HERE/'figure_label_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
