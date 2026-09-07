"""Procedural manual motion from parsed Formal SignWriting symbols."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

from .avatar_rig_v17 import (RetargetedAvatar, _fallback_rest_hand, _solve_elbow,
                             interpolate_world_hand)


def load_hand_templates(path: Path):
    with np.load(path) as data:
        return ({str(key): value for key,value in zip(data['views'],data['oriented_templates'])},
                {str(key): value for key,value in zip(data['bases'],data['templates'])})


def _rotate_z(points, degrees):
    angle=np.deg2rad(degrees);c,s=np.cos(angle),np.sin(angle)
    return np.asarray(points)@np.array([[c,s,0],[-s,c,0],[0,0,1]],np.float32)


def oriented_hand(symbol, side, views, bases):
    """Apply the FSW in-plane rotation/chirality to an official-photo template."""
    view=symbol[:5];base=symbol[:4]
    if view in views:
        hand=np.asarray(views[view],np.float32).copy()
    elif base in bases:
        hand=np.asarray(bases[base],np.float32).copy()
    else:
        raise ValueError(f'no official-photo hand template for {symbol}')
    axis=hand[9,:2]-hand[0,:2]
    if np.linalg.norm(axis)<1e-8:
        raise ValueError(f'hand template has no palm direction for {symbol}')
    hand=_rotate_z(hand,np.degrees(np.arctan2(axis[0],axis[1])))
    rotation=int(symbol[5],16)
    if side==0:
        longitudinal=hand[9]-hand[0];longitudinal/=np.linalg.norm(longitudinal)
        normal=np.cross(hand[5]-hand[17],longitudinal);normal/=np.linalg.norm(normal)
        lateral=np.cross(longitudinal,normal);lateral/=np.linalg.norm(lateral)
        relative=hand-hand[0]
        hand=relative-2*np.sum(relative*lateral,axis=1,keepdims=True)*lateral+hand[0]
    hand=_rotate_z(hand,-45*(rotation%8))
    return hand


def _minimum_jerk(value):
    return value**3*(10-15*value+6*value*value)


def stroke_progress(frames, repeated):
    t=np.linspace(0,1,frames,dtype=np.float32)
    if repeated:
        t=np.where(t<1/3,3*t,np.where(t<2/3,2-3*t,3*t-2))
    return _minimum_jerk(t)


def _symbol_name(symbol):
    return unicodedata.name(chr(0x1d800+int(symbol[1:4],16)-0x100))


def motion_transform(symbol, frames, repeated):
    """Return wrist displacement and hand rotations for one ISWA motion symbol."""
    displacement=np.zeros((frames,3),np.float32)
    rotations=np.repeat(np.eye(3,dtype=np.float32)[None],frames,axis=0)
    if symbol is None:
        return displacement,rotations,stroke_progress(frames,repeated)
    name=_symbol_name(symbol);rotation=int(symbol[5],16)%8
    progress=stroke_progress(frames,repeated)
    amplitude=.06 if 'SMALL' in name else .10 if 'MEDIUM' in name else .14 if 'LARGE' in name else .08
    theta=np.deg2rad(rotation*45);page=np.array([np.sin(theta),np.cos(theta)],np.float32)
    if 'WRIST FLEX' in name or 'HINGE' in name:
        angle=np.deg2rad(28)*progress
        axis=2 if 'FLOORPLANE' in name else 0
        for i,a in enumerate(angle):
            c,s=np.cos(a),np.sin(a)
            rotations[i]=([[c,-s,0],[s,c,0],[0,0,1]] if axis==2 else
                          [[1,0,0],[0,c,-s],[0,s,c]])
    elif 'CIRCLE' in name or name.startswith('SIGNWRITING ROTATION'):
        turns=2 if ('DOUBLE' in name or repeated) else 1
        angle=2*np.pi*turns*np.linspace(0,1,frames)
        if 'ARM CIRCLE' in name:
            displacement[:,0]=amplitude*np.sin(angle)
            displacement[:,1]=amplitude*(1-np.cos(angle))
        else:
            for i,a in enumerate(angle):
                c,s=np.cos(a),np.sin(a);rotations[i]=[[c,-s,0],[s,c,0],[0,0,1]]
    elif 'CURVE' in name or 'PEAKS' in name:
        if 'FLOORPLANE' in name:
            direction=np.array([page[0],0,page[1]])
            perpendicular=np.array([page[1],0,-page[0]])
        else:
            direction=np.array([page[0],page[1],0])
            perpendicular=np.array([page[1],-page[0],0])
        displacement=amplitude*(progress[:,None]*direction+
            .35*np.sin(np.pi*progress)[:,None]*perpendicular)
    elif 'DIAGONAL' in name:
        sign=-1 if 'TOWARDS' in name else 1
        displacement=amplitude*progress[:,None]*np.array([page[0],page[1]/np.sqrt(2),
                                                          sign*page[1]/np.sqrt(2)])
    elif 'WALLPLANE' in name:
        displacement=amplitude*progress[:,None]*np.array([page[0],page[1],0])
    elif 'FLOORPLANE' in name:
        displacement=amplitude*progress[:,None]*np.array([page[0],0,page[1]])
    return displacement.astype(np.float32),rotations,progress


def _finger_action(hand, symbol, side, progress):
    if symbol is None:
        return hand
    name=_symbol_name(symbol)
    if not any(action in name for action in ('SQUEEZE','FLICK','HINGE')):
        return hand
    curled=np.asarray(hand,np.float32).copy()
    across=curled[5]-curled[17];along=curled[9]-curled[0]
    normal=np.cross(across,along)*(-1 if side==0 else 1)
    normal/=max(np.linalg.norm(normal),1e-8)
    for first in (5,9,13,17):
        chain=curled[first:first+4];lengths=np.linalg.norm(np.diff(chain,axis=0),axis=1)
        if np.linalg.norm(chain[-1]-chain[0])/lengths.sum()<.72:
            continue
        direction=chain[1]-chain[0]
        direction-=normal*np.dot(direction,normal)
        direction/=max(np.linalg.norm(direction),1e-8)
        for joint,angle in enumerate(np.deg2rad((20,85,145)),1):
            vector=direction*np.cos(angle)+normal*np.sin(angle)
            curled[first+joint]=curled[first+joint-1]+vector*lengths[joint-1]
    return interpolate_world_hand(curled,hand,float(progress)) if 'FLICK' in name else \
           interpolate_world_hand(hand,curled,float(progress))


_ANCHORS={
    # Wrist medians from the source-comparison corpus, grouped only by the locked
    # phonological location. The written contact location belongs to the fingers.
    'Neutral': (.18,1.13,.235), 'BodyAway': (.28,1.29,.32),
    'TorsoTop': (.12,1.14,.235), 'Clavicle': (.16,1.20,.235),
    'Forehead': (.18,1.33,.235), 'Eye': (.14,1.22,.235),
    'CheekNose': (.18,1.23,.235), 'Mouth': (.08,1.21,.235),
    'Chin': (.10,1.23,.235), 'UnderChin': (.14,1.22,.235),
    'HeadAway': (.14,1.20,.30), 'UpperArm': (.125,1.33,.235),
    'HandAway': (.14,1.13,.30), 'Palm': (.10,1.17,.235),
    'PalmBack': (.065,1.16,.235), 'FingerBack': (.19,1.09,.235),
    'FingerRadial': (.11,1.12,.235), 'WristBack': (.14,1.15,.235),
    'WristFront': (.17,1.15,.235), 'Other': (.18,1.315,.235),
}


def _wrist_anchor(phonology,side,two_handed,contact=False):
    minor=phonology.get('minor_location') or 'Neutral';x,y,z=_ANCHORS.get(minor,_ANCHORS['Neutral'])
    if phonology.get('major_location')=='Hand':
        x=.055 if side==0 else -.055
    elif two_handed and contact:
        x=.04 if side==0 else -.04
    else:
        x=x if side==0 else -x
    return np.array([x,y,z],np.float32)


def generate_manual_rig(parsed,phonology,views,bases,frames=30):
    """Generate one manual sign core; source-video landmarks are never accepted."""
    symbols=parsed['symbols'];hand_symbols=[s for s in symbols if s['category']=='hand']
    if not hand_symbols:
        raise ValueError('notation has no executable hand symbol')
    written_sides={1 if int(symbol['key'][5],16)<8 else 0 for symbol in hand_symbols}
    two_handed=(len(written_sides)==2 or phonology.get('sign_type')!='OneHanded' or
                (phonology.get('major_location')=='Arm' and len(hand_symbols)>1))
    grouped={0:[],1:[]}
    if two_handed:
        for symbol in hand_symbols:
            grouped[1 if int(symbol['key'][5],16)<8 else 0].append(symbol)
        if not grouped[0] or not grouped[1]:
            if phonology.get('major_location')=='Arm' and len(hand_symbols)==2:
                passive=next((s for s in hand_symbols if s['base']=='S15a'),hand_symbols[0])
                active=next(s for s in hand_symbols if s is not passive)
                grouped={0:[passive],1:[active]}
            else:
                ordered=sorted(hand_symbols,key=lambda s:s['x'])
                grouped={0:ordered[:len(ordered)//2],1:ordered[len(ordered)//2:]}
    else:
        grouped[1]=hand_symbols
    motions=[s for s in symbols if s['category']=='movement']
    finger_motions=[s for s in symbols if s['category']=='finger_movement']
    contact=any(s['category']=='contact' for s in symbols)
    written_center=np.mean([[s['x'],s['y']] for s in hand_symbols],axis=0)
    origins={side:_wrist_anchor(phonology,side,two_handed,contact)+.006*np.array(
        [written_center[0]-np.mean([s['x'] for s in shapes]),
         written_center[1]-np.mean([s['y'] for s in shapes]),0],np.float32)
        for side,shapes in grouped.items() if shapes}
    visible=set(origins)
    if len(visible)==1 or phonology.get('sign_type')=='SymmetricalOrAlternating':
        contact_sides=visible
    else:
        contact_sides={1 if 1 in visible else next(iter(visible))}
    repeated=phonology.get('repeated_movement')=='1'
    hands=np.repeat(np.stack([_fallback_rest_hand(0),_fallback_rest_hand(1)])[None],frames,axis=0)
    states=np.tile(np.array(['rest-uncertain','rest-uncertain']), (frames,1))
    for side,shapes in grouped.items():
        if not shapes:continue
        center=np.mean([[s['x'],s['y']] for s in shapes],axis=0)
        movement=min(motions,key=lambda s:np.linalg.norm(np.array([s['x'],s['y']])-center)) if motions else None
        finger_movement=min(finger_motions,key=lambda s:np.linalg.norm(
            np.array([s['x'],s['y']])-center)) if finger_motions else None
        displacement,rotations,progress=motion_transform(movement['key'] if movement else None,
                                                         frames,repeated)
        if contact and not motions and side in contact_sides:
            kind=phonology.get('movement')
            if kind=='Circular':
                turns=2 if repeated else 1;angle=2*np.pi*turns*np.linspace(0,1,frames)
                displacement[:,0]+=.03*np.sin(angle)
                displacement[:,1]+=.03*(1-np.cos(angle))
            elif kind in ('Straight','BackAndForth') or any(
                    s['base'] in ('S206','S20b','S211') for s in symbols):
                amount=.05 if any(s['base']=='S20b' for s in symbols) else .03
                if phonology.get('major_location') in ('Hand','Arm') and len(visible)==2:
                    other=next(iter(visible-{side}));toward=origins[other]-origins[side]
                    toward/=max(np.linalg.norm(toward),1e-8)
                    displacement-=amount*(1-progress[:,None])*toward
                else:
                    displacement[:,2]+=amount*(1-progress)
        direction=None
        if movement:
            theta=np.deg2rad((int(movement['key'][5],16)%8)*45)
            direction=np.array([np.sin(theta),-np.cos(theta)])
        if len(shapes)>1 and direction is not None:
            shapes=sorted(shapes,key=lambda s:np.dot([s['x'],s['y']],direction))
        first=oriented_hand(shapes[0]['key'],side,views,bases)
        last=oriented_hand(shapes[-1]['key'],side,views,bases)
        anchor=_wrist_anchor(phonology,side,two_handed,contact)
        offsets=[.006*np.array([written_center[0]-s['x'],written_center[1]-s['y'],0],np.float32)
                 for s in (shapes[0],shapes[-1])]
        for i,p in enumerate(progress):
            shape=interpolate_world_hand(first,last,float(p)) if len(shapes)>1 else first.copy()
            shape=_finger_action(shape,finger_movement['key'] if finger_movement else None,side,p)
            shape=shape@rotations[i].T
            position=anchor+offsets[0]*(1-p)+offsets[1]*p+displacement[i]
            hands[i,side]=shape-shape[0]+position
        states[:,side]='symbolic'
    if len(visible)==1 and any(s['category']=='body' and 'LIMB' in _symbol_name(s['key'])
                               for s in symbols):
        passive=1-next(iter(visible));shape=_fallback_rest_hand(passive)
        shape-=shape[0];axis=shape[9,:2]
        shape=_rotate_z(shape,np.degrees(np.arctan2(axis[0],axis[1])))
        shape=_rotate_z(shape,90 if passive==0 else -90)
        wrist=np.array([.24 if passive==0 else -.24,1.03,.235],np.float32)
        hands[:,passive]=shape+wrist;states[:,passive]='symbolic'
    shoulders=np.repeat(np.array([[[.1677125,1.285,0],[-.1677125,1.285,0]]],np.float32),frames,axis=0)
    elbows=np.empty_like(shoulders)
    for i in range(frames):
        for side in range(2):
            elbows[i,side],wrist=_solve_elbow(shoulders[i,side],hands[i,side,0],side)
            hands[i,side]+=wrist-hands[i,side,0]
    return RetargetedAvatar(shoulders,elbows,hands,states,np.zeros((frames,2),bool))
