import numpy as np
import torch
import sys,os,glob
#import matplotlib.pyplot as plt

### util
class _util():
    num2aa=[
        'ALA','ARG','ASN','ASP','CYS',
        'GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO',
        'SER','THR','TRP','TYR','VAL',
        ]

    aa2num= {x:i for i,x in enumerate(num2aa)}

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
    aa_1_N = {a:n for n,a in enumerate(alpha_1)}

    aa123 = {aa1: aa3 for aa1, aa3 in zip(alpha_1, num2aa)}
    aa321 = {aa3: aa1 for aa1, aa3 in zip(alpha_1, num2aa)}

    # minimal sc atom representation (Nx8)
    aa2short=[
        (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), # ala
        (" N  "," CA "," C  "," CB "," CG "," CD "," NE "," CZ "), # arg
        (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), # asn
        (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), # asp
        (" N  "," CA "," C  "," CB "," SG ",  None,  None,  None), # cys
        (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), # gln
        (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), # glu
        (" N  "," CA "," C  ",  None,  None,  None,  None,  None), # gly
        (" N  "," CA "," C  "," CB "," CG "," ND1",  None,  None), # his
        (" N  "," CA "," C  "," CB "," CG1"," CD1",  None,  None), # ile
        (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # leu
        (" N  "," CA "," C  "," CB "," CG "," CD "," CE "," NZ "), # lys
        (" N  "," CA "," C  "," CB "," CG "," SD "," CE ",  None), # met
        (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # phe
        (" N  "," CA "," C  "," CB "," CG "," CD ",  None,  None), # pro
        (" N  "," CA "," C  "," CB "," OG ",  None,  None,  None), # ser
        (" N  "," CA "," C  "," CB "," OG1",  None,  None,  None), # thr
        (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # trp
        (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # tyr
        (" N  "," CA "," C  "," CB "," CG1",  None,  None,  None), # val
    ]

    # full sc atom representation (Nx14)
    aa2long=[
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
        (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
        (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
        (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
        (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
        (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
        (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
    ]

    # build the "alternate" sc mapping
    aa2longalt=[
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH2"," NH1",  None,  None,  None), # arg
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD2"," OD1",  None,  None,  None,  None,  None,  None), # asp
        (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE2"," OE1",  None,  None,  None,  None,  None), # glu
        (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
        (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1",  None,  None,  None,  None,  None,  None), # leu
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
        (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ ",  None,  None,  None), # phe
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
        (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
        (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ "," OH ",  None,  None), # tyr
        (" N  "," CA "," C  "," O  "," CB "," CG2"," CG1",  None,  None,  None,  None,  None,  None,  None), # val
    ]



def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [_util.aa2num[r[1]] if r[1] in _util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num


    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(_util.aa2long[_util.aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)
            
    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14] [N, CA, C, CB, ..]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het
    return out

# process ideal frames
def make_frame(X, Y):
    if type(X) != torch.Tensor or type(Y) != torch.Tensor:
        X = torch.tensor(X).clone().detach()
        Y = torch.tensor(Y).clone().detach()

    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn, dim=-1)
    Zn =  Z / torch.linalg.norm(Z)
    return torch.stack((Xn,Yn,Zn), dim=-1)

# ang between vectors
def th_ang_v(ab,bc,eps:float=1e-8):
    if type(ab) != torch.Tensor or type(bc) != torch.Tensor:
        ab = torch.tensor(ab).clone().detach()
        bc = torch.tensor(bc).clone().detach()
    def th_norm(x,eps:float=1e-8):
        #x = torch.tensor(x)
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        #x = torch.tensor(x)
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

# dihedral between vectors
def th_dih_v(ab,bc,cd):
    # convert to torch
    if type(ab) != torch.Tensor or type(bc) != torch.Tensor or type(cd) != torch.Tensor:
        ab = torch.tensor(ab).clone().detach()
        bc = torch.tensor(bc).clone().detach()
        cd = torch.tensor(cd).clone().detach()
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih # cos, sin

# dihedral between points
def th_dih(a,b,c,d):
    if type(a) != torch.Tensor or type(b) != torch.Tensor or type(c) != torch.Tensor or type(d) != torch.Tensor:
        a = torch.tensor(a).clone().detach()
        b = torch.tensor(b).clone().detach()
        c = torch.tensor(c).clone().detach()
        d = torch.tensor(d).clone().detach()
    return th_dih_v(a-b,b-c,c-d)


### convert cos/sin to angle
def trig_2_ang(cos_angle=None, sin_angle=None, radian=True):
    ### phi sign is correctly assigned, psi seems to be flipped TODO verify
    if radian:
        return torch.atan2(sin_angle,cos_angle)
    else:
        return torch.atan2(sin_angle,cos_angle)*180/np.pi

def get_phi(xyz): # bb xyz, 2 frames
    # input xyz - torch.tensor [2, 14, 3], only use first 5 atoms (N, CA, C, O, CB)
    # output phi dihedral angle, in degrees
    phi_cos_sin = th_dih(xyz[0][2],xyz[1][0],xyz[1][1],xyz[1][2]) # C_i-N_i+1-Ca_i+1-C_i+1
    phi_ang = trig_2_ang(cos_angle=phi_cos_sin[0], sin_angle=phi_cos_sin[1],radian=False)
    return phi_ang

def get_psi(xyz): # bb xyz, 2 frames
    # input xyz - torch.tensor [2, 14, 3], only use first 5 atoms (N, CA, C, O, CB)
    # output psi dihedral angle, in degrees
    #psi_cos_sin = th_dih(xyz[1][0],xyz[1][1],xyz[1][2],xyz[1][3])
    psi_cos_sin = th_dih(xyz[0][0],xyz[1][1],xyz[0][2],xyz[1][0]) # N_i-Ca_i-C_i+1_N_i+1
    psi_ang = -trig_2_ang(cos_angle=psi_cos_sin[0], sin_angle=psi_cos_sin[1],radian=False)
    return psi_ang

def get_phis(xyz): # bb_xyz, all frames, from residue 2 to n (1-indexed)
    assert len(xyz.shape) >= 3, "input should be [L, atom, xyz]"
    if len(xyz.shape) > 3:
        xyz = xyz[-1,:] #xyz = xyz[len(xyz.shape)-4,:] # or xyz[-1,:]

    phi = []
    for i in range(1,xyz.shape[0]):
        phi.append(get_phi(xyz[i-1:i+1]))
    return phi

def get_psis(xyz): # bb_xyz, all frames, from residue 1 to n-1 (1-indexed)
    assert len(xyz.shape) >= 3, "input should be [L, atom, xyz]"
    if len(xyz.shape) > 3:
        xyz = xyz[-1,:] #xyz = xyz[len(xyz.shape)-4,:]

    psi = []
    for i in range(1,xyz.shape[0]):
        psi.append(get_psi(xyz[i-1:i+1]))
    return psi
