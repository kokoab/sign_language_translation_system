from pathlib import Path
D=Path(__file__).resolve().parent
s=(D/'export_actual.py').read_text();s=s.replace('self.model=model;self.original_word_head=original.fusion_head;self.letters=letters','''self.model=model;self.original_word_head=original.fusion_head;self.letters=letters
  import json
  labels=json.loads((O/'resources/live_reel_labels.json').read_text())
  if isinstance(labels,dict): labels=labels.get('labels',labels.get('word_labels'))
  self.my=labels.index('MY');self.like=labels.index('LIKE')
  self.floor=float(os.environ.get('REPAIR_FLOOR','0.9'))
  self.policy=os.environ.get('REPAIR_POLICY','swap')
  my_mask=torch.zeros(100);my_mask[self.my]=1;like_mask=torch.zeros(100);like_mask[self.like]=1
  self.register_buffer('my_mask',my_mask);self.register_buffer('like_mask',like_mask)''')
s=s.replace('return words,self.letters(lf,hf,original_words)','''prob=torch.softmax(words,dim=1)
  rank=((original_words>original_words[:,self.my:self.my+1]).to(torch.int32)).sum(dim=1)
  gate=(original_words.argmax(dim=1)==self.like)&(rank<3)&(words.argmax(dim=1)==self.my)&(prob[:,self.my]>=self.floor)
  if self.policy=='swap':
   delta=original_words[:,self.like]-original_words[:,self.my]
   corrected=original_words+delta[:,None]*(self.my_mask-self.like_mask)[None,:]
  else:
   original_lp=torch.log_softmax(original_words,dim=1)
   pair_lp=torch.log_softmax(torch.stack([words[:,self.my],words[:,self.like]],dim=1),dim=1)
   mass=torch.logsumexp(torch.stack([original_lp[:,self.my],original_lp[:,self.like]],dim=1),dim=1)
   corrected=original_words+(mass+pair_lp[:,0]-original_lp[:,self.my])[:,None]*self.my_mask[None,:]+(mass+pair_lp[:,1]-original_lp[:,self.like])[:,None]*self.like_mask[None,:]
  repaired=torch.where(gate[:,None],corrected,original_words)
  return repaired,self.letters(lf,hf,original_words)''')
s=s.replace("package=O/('SpanRecognizerV17'+''.join(w.title() for w in name.split('_'))+'B8FP16.mlpackage')", "package=O/('SpanRecognizerV17Gated'+os.environ.get('REPAIR_POLICY','swap').title()+'B8FP16.mlpackage')")
s=s.replace("O/f'{name}_export.json'","O/f'gated_{os.environ.get(\"REPAIR_POLICY\",\"swap\")}_export.json'")
(D/'export_gated.py').write_text(s)
print('Prepared one-way confidence-gated currentLIKE→MY export')
