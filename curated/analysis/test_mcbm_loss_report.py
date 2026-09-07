"""Synthetic runtime tests. These never create scientific result manifests."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import nbformat
from IPython.display import Markdown

from build_mcbm_parity import TAGS, SHARED, find, adapt, INFORMATION, ERASURE
from build_standard_cbm_reports import build_funnybird
from mcbm_loss_report import task_head, replacement_use, score_reference, off_target_erasure, softmax, loss_gradient_audit
from funnybird_followup_diagnostics import (conditional_information, conflict_response_table,
    ordinary_value_recognition, add_descriptors, prediction_audit, value_holdout_audit)


def fixture(directory):
    rng = np.random.default_rng(241)
    spans = dict(beak=(0,4),eye=(4,7),foot=(7,11),tail=(11,20),wing=(20,26))
    names = [f'{part}_{j}' for part,(lo,hi) in spans.items() for j in range(hi-lo)]
    y = np.repeat(np.arange(50),10)
    c = np.zeros((len(y),26),int)
    for lo,hi in spans.values():
        c[np.arange(len(y)),lo+rng.integers(hi-lo,size=len(y))]=1
    h = (6*c-3+rng.normal(size=c.shape)).astype('float32')
    z = h * 1.2
    torch.manual_seed(3)
    network = torch.nn.Sequential(torch.nn.Linear(26,256),torch.nn.ReLU(),torch.nn.Linear(256,50))
    checkpoint = directory/'head.pt'
    torch.save({'model':{f'mlp_y.{k}':v for k,v in network.state_dict().items()}},checkpoint)
    forward=task_head(checkpoint)
    np.testing.assert_allclose(forward(h),network(torch.tensor(h)).detach().numpy(),rtol=1e-6,atol=1e-6)
    rows=[]
    for part,(lo,hi) in spans.items():
        for i in range(1000):
            source=i%50; donor=(source+1+i//50)%50
            if donor==source: donor=(donor+1)%50
            old=i%(hi-lo); new=(old+1+i//(hi-lo))%(hi-lo)
            if new==old: new=(new+1)%(hi-lo)
            start=-abs(rng.normal(5,2)); delta=rng.normal(5,3); margin=start+delta
            record=dict(part=part,var_src=old,var_donor=new,sid_src=source,sid_donor=donor,
                orig_render_id=f'orig-{i%250:03d}',render_id=f'{part}-{i}',direction=['fwd','bwd'][i%2],
                z_old_orig=-start/2,z_new_orig=start/2,z_old=-margin/2,z_new=margin/2,
                margin=margin,m_cf=margin,m_orig=start,response_delta=delta,
                donor_gain=delta/2,source_decrease=delta/2,pixel_count_cf=int(rng.integers(1,1000)),
                responded_but_source_wins=bool(delta>0 and margin<0),
                controlled_event=int(delta>0 and margin<0),p_cf_donor=float(rng.uniform(0,.2)))
            for j in range(hi-lo):record[f'z_cf_{part}_{j}']=float(rng.normal())
            rows.append(record)
    s=pd.DataFrame(rows)
    conflict=pd.DataFrame([dict(concept_index=j,concept=name,part=name.rsplit('_',1)[0],
        value=int(name.rsplit('_',1)[1]),positive_images=100,hidden_positive_images=j,
        conflict_rate=j/100,species_support=2+j%18) for j,name in enumerate(names)])
    return dict(S=s,z_saved=z,h_saved=h,c_saved=c,y_saved=y,y_pred_saved=forward(h).argmax(1),
        CONCEPT_NAMES=names,SPANS=spans,CONCEPT_PART={n:n.rsplit('_',1)[0] for n in names},
        ORDER=['tail','wing','beak','foot','eye'],
        COLORS=dict(tail='#6A0DAD',wing='#0072B2',beak='#E69F00',foot='#009E73',eye='#CC79A7'),
        saved_head=forward,ordinary={'y_probability':softmax(forward(h))},CONFLICT=conflict,GAMMA=0.,
        PART_CONFLICT=conflict.groupby('part').conflict_rate.mean().to_frame())


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--slow',action='store_true')
    parser.add_argument('--mechanics-only',action='store_true')
    args=parser.parse_args()
    print('PREFLIGHT ONLY — SYNTHETIC DATA, NO SCIENTIFIC TRAINING',flush=True)
    nbpath=Path(__file__).resolve().parents[1]/'notebooks/03_funnybirds_mcbm.ipynb'
    notebook=nbformat.read(nbpath,as_version=4); nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code': compile(cell.source,cell.id,'exec')
    with tempfile.TemporaryDirectory(prefix='mcbm-report-test-') as temp:
        env=fixture(Path(temp)); env.update(dict(np=np,pd=pd,plt=plt,torch=torch,
            display=lambda *args,**kwargs:None,Markdown=Markdown,
            conditional_information=conditional_information,replacement_use=replacement_use,
            score_reference=score_reference,conflict_response_table=conflict_response_table,
            ordinary_value_recognition=ordinary_value_recognition,add_descriptors=add_descriptors,
            prediction_audit=prediction_audit,value_holdout_audit=value_holdout_audit))
        reference=score_reference(env['h_saved'],env['z_saved'],env['c_saved'],env['CONCEPT_NAMES'])
        assert len(reference)==104 and reference.N.min()>1
        use=replacement_use(env['h_saved'],env['c_saved'],env['y_saved'],
             env['ordinary']['y_probability'],env['saved_head'],env['SPANS'])
        assert len(use)==6 and use.mean_probability_mass_moved.between(0,1).all()
        hcf=np.tile(env['h_saved'],(10,1))
        evidence=off_target_erasure(env['S'],hcf,env['h_saved'],env['c_saved'],env['saved_head'],env['SPANS'])
        assert set(evidence.off_target_coordinates)=={1,2,4,7}
        assert evidence.mean_probability_mass_moved.between(0,1).all()
        # If every erased slot already equals its baseline, the effect must vanish.
        means=np.array([env['h_saved'][env['c_saved'][:,j]==0,j].mean() for j in range(26)])
        zero=off_target_erasure(env['S'],np.tile(means,(5000,1)),env['h_saved'],env['c_saved'],env['saved_head'],env['SPANS'])
        np.testing.assert_allclose(zero.off_target_source_evidence,0,atol=1e-6)
        env['EVIDENCE_ROWS']=evidence
        # Use the actual upstream loss with a tiny encoder: no download/training.
        from unittest.mock import patch
        import grounding_deletion
        from src.models.mcbm import MinimalConceptBottleneckModel
        tiny=MinimalConceptBottleneckModel(n_concepts=26,dim_y=50,dim_c=1,
            continuous_y=False,continuous_c=False,encoder=dict(arch='mlp',hidden_dims=[4],latent_dim=8),
            hidden_dims_y=256,hidden_dims_c=[3],hidden_dims_z=None,beta=1,gamma=1,var_z=1)
        before={name:value.detach().clone() for name,value in tiny.named_parameters()}
        with patch.object(grounding_deletion,'load_model',return_value=(tiny,26)):
            gradient=loss_gradient_audit(env['h_saved'][:67],env['c_saved'][:67],env['y_saved'][:67],1,env['SPANS'])
        for part,(lo,hi) in env['SPANS'].items():
            expected=.4*(env['h_saved'][:67,lo:hi]-(6*env['c_saved'][:67,lo:hi]-3))
            np.testing.assert_allclose(gradient.set_index('part').loc[part,'weighted_compression_gradient_RMS'],
                np.sqrt(np.mean(expected**2)),rtol=1e-5)
        for name,value in tiny.named_parameters():
            assert torch.equal(value,before[name]) and value.grad is None
        # Render the new nonlinear-erasure panel with synthetic h; the real CUDA
        # image-loader is deliberately not claimed as tested by this fixture.
        env.update(dict(h_cf=hcf,off_target_erasure=off_target_erasure))
        exec(compile(ERASURE[ERASURE.index('EVIDENCE_ROWS='):],'erasure-plot','exec'),env)
        preview=Path(tempfile.gettempdir())/'mcbm_report_erasure_SYNTHETIC.png'
        plt.gcf().savefig(preview,dpi=100);plt.close('all')
        print('Synthetic layout preview:',preview,flush=True)
        from funnybird_followup_diagnostics import grouped_predictions, source_stratified_folds
        constant=env['S'].copy();constant['controlled_event']=0
        predicted=grouped_predictions(constant,[],['part'],'controlled_event',True,source_stratified_folds(constant))
        assert (predicted==0).all()
        print('PASS: official loss gradient (short batch included), immutable weights, constant-event folds',flush=True)
        if args.mechanics_only:return
        standard=build_funnybird()['cells']
        for tag in TAGS:
            if tag in SHARED or tag=='f8d-source':continue
            if not args.slow and tag in {'f8c-source','f9-new'}:continue
            print('Runtime test:',tag,flush=True)
            exec(compile(adapt(''.join(find(standard,tag)['source']),tag),tag,'exec'),env)
            plt.close('all')
    print('SYNTHETIC RUNTIME PASS; CUDA image replay still requires real artifacts',flush=True)


if __name__=='__main__':main()
