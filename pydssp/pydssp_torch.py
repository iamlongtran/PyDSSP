#! /usr/bin/env python

import torch
from einops import repeat, rearrange

CONST_Q1Q2 = 0.084 #q1=0.42e; q2=0.20e. e=1.6e-19 C
CONST_F = 332 # dimenional factor
DEFAULT_CUTOFF = -0.5 # H-bond energy cutoff // pretty liberal cutoff, H bond can form if energy is <= -0.5 kcal/mol
DEFAULT_MARGIN = 1.0 # H bond E cutoff margin, at the marginal value, often seen in week H bond, bifurcated h bond? ~50-60% E of canonical H bonds 
# https://www.pnas.org/doi/full/10.1073/pnas.1319827111
from pydssp import util


def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape)==3) or (len(org_shape)==4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = repeat(coord, '... -> b ...', b=1) if len(org_shape)==3 else coord
    return coord, org_shape


def _get_hydrogen_atom_position(coord: torch.Tensor) -> torch.Tensor:
    # A little bit lazy (but should be OK) definition of H position here.
    vec_cn = coord[:,1:,0] - coord[:,:-1,2]
    vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
    vec_can = coord[:,1:,0] - coord[:,1:,1]
    vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
    return coord[:,1:,0] + 1.01 * vec_nh


def get_hbond_map(
    coord: torch.Tensor,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    return_e: bool=False
    ) -> torch.Tensor:
    # check input
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a==4) or (a==5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:,1:,4] if a == 5 else _get_hydrogen_atom_position(coord)
    # distance matrix
    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)
    d_on = torch.linalg.norm(omap - nmap, dim=-1)
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)
    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [0,1,1,0])
    if return_e: return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool)
    local_mask *= ~torch.diag(torch.ones(l-1, dtype=bool), diagonal=-1)
    local_mask *= ~torch.diag(torch.ones(l-2, dtype=bool), diagonal=-2)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map/margin*torch.pi/2)+1.)/2
    hbond_map = hbond_map * repeat(local_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
    # return h-bond map
    hbond_map = hbond_map.squeeze(0) if len(org_shape)==3 else hbond_map
    return hbond_map


def assign(coord: torch.Tensor) -> torch.Tensor:
    # check input
    coord, org_shape = _check_input(coord)
    # get hydrogen bond map
    hbmap = get_hbond_map(coord)
    hbmap = rearrange(hbmap, '... l1 l2 -> ... l2 l1') # convert into "i:C=O, j:N-H" form
    # identify turn 3, 4, 5
    turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3) > 0.
    turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4) > 0.
    turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = torch.nn.functional.pad(turn3[:,:-1] * turn3[:,1:], [1,3])
    h4 = torch.nn.functional.pad(turn4[:,:-1] * turn4[:,1:], [1,4])
    h5 = torch.nn.functional.pad(turn5[:,:-1] * turn5[:,1:], [1,5])
    # helix4 first
    helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
    h3 = h3 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    h5 = h5 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
    helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1) > 0.
    unfoldmap_rev = unfoldmap.transpose(-4,-3)
    p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
    p_bridge = torch.nn.functional.pad(p_bridge, [1,1,1,1])
    a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
    a_bridge = torch.nn.functional.pad(a_bridge, [1,1,1,1])
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # H, E, L of C3 #TODO add h as left handed helix

    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    ### add left handed helix here
    phis = util.get_phis(coord) # [tensor] L-1
    psis = util.get_psis(coord) # [tensor] L-1
    assert len(phis)==len(psis), "Length of phi and psi should be the same"
    assert len(phis)==len(helix[0])-1, "Length of phi and psi should be L-1"
    phis.append(torch.tensor([0.])) # add dummy phi for the last residue
    psis.append(torch.tensor([0.])) # add dummy psi for the last residue
    phis_torch = torch.FloatTensor(phis)
    psis_torch = torch.FloatTensor(psis)
    p_phis_psis = torch.where((phis_torch>0) & (psis_torch >0), 1,0)
    left_handed_helix = helix * p_phis_psis

    right_handed_helix = helix.long() - left_handed_helix

    #left_handed_helix = torch.ones_like(helix) #temp
    onehot = torch.stack([loop, right_handed_helix, strand, left_handed_helix], dim=-1)
    onehot = onehot.squeeze(0) if len(org_shape)==4 else onehot
    return onehot
