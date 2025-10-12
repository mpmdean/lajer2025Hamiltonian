import edrixs_python as edrixs
import sys
sys.modules['edrixs'] = edrixs
import scipy
import numpy as np

def constructMockH(a,b,c,v_name,c_name,v_noccu,c_holes, t_or_L='t'):
    n_occu =v_noccu
    info_shell = edrixs.info_atomic_shell()
    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]
    c_orbl = info_shell[c_name][0]

    # number of orbitals including spin degree of freedom
    v_norb = info_shell[v_name][1]
    c_norb = info_shell[c_name][1]

    basis = edrixs.get_fock_bin_by_N(v_norb, n_occu, c_norb, c_norb-c_holes)
    print(len(basis))
    orb_momD = edrixs.get_orb_momentum(v_orbl, ispin=True)
    orb_momP = edrixs.get_orb_momentum(c_orbl, ispin=True)
    orb_momTot=np.array([scipy.linalg.block_diag(orb_momD[i],orb_momP[i]) for i in range(3)])
    spin_momD = edrixs.get_spin_momentum(v_orbl)
    spin_momP = edrixs.get_spin_momentum(c_orbl)
    spin_momTot=np.array([scipy.linalg.block_diag(spin_momD[i],spin_momP[i]) for i in range(3)])
    Lop,Sop=edrixs.build_opers(2, [orb_momTot, spin_momTot],basis)

    L2op=np.dot(Lop[0], Lop[0]) + np.dot(Lop[1], Lop[1]) + np.dot(Lop[2], Lop[2])

    S2op=np.dot(Sop[0], Sop[0]) + np.dot(Sop[1], Sop[1]) + np.dot(Sop[2], Sop[2])

    if t_or_L=='t':
        t2gorbs=np.array([0,0,1,1,1,1,0,0,1,1])
        t2gmatc=edrixs.basis_transform.cb_op(np.diag(t2gorbs),edrixs.basis_transform.tmat_r2c('d',ispin=True))
        torLfock=edrixs.build_opers(2, t2gmatc, basis)
    elif t_or_L=='L':
        torLfock=L2op

    gr_elems=1j*np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1],
                        [1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[0,0,0]],dtype=complex)
    gr_elems[:3] *= np.pi/2
    gr_elems[3:6] *= np.pi
    gr_elems[6:9] *= np.pi*3/2
    gr_elems[9:15] *= 1.0/np.sqrt(2)*np.pi
    gr_elems[15:19] *= 1.0/np.sqrt(3)*2*np.pi/3
    gr_elems[19:23] *= 1.0/np.sqrt(3)*4*np.pi/3

    symarr=np.zeros([gr_elems.shape[0],Sop.shape[1],Sop.shape[1]],dtype=complex)
    Ops=np.zeros([gr_elems.shape[0],Sop.shape[1],Sop.shape[1]],dtype=complex)
    for i in range(gr_elems.shape[0]):
        Ops[i]=scipy.linalg.expm(np.tensordot(gr_elems[i],Lop,axes=1))

    mockH0=np.zeros([len(basis),len(basis)],dtype=complex)
    mockH0=a*S2op+b*Sop[2]+c*torLfock+0.000*Lop[2]
    eval_i,evec_i=np.linalg.eigh(mockH0)
    e=np.round(eval_i,decimals=6)
    ind_list=np.array([0])
    for i in range(1,e.size):
        if e[i] > e[i-1]:
            ind_list=np.append(ind_list,i)
    ind_list=np.append(ind_list,eval_i.shape[0])
    print(ind_list[1:]-ind_list[:-1])
    for i in range(Ops.shape[0]):
        Ops[i] = np.dot(np.conjugate(evec_i).T,np.dot(Ops[i],evec_i))

    blocks=[]
    for k in range(ind_list.shape[0]-1):
        nb=ind_list[k+1]-ind_list[k]
        block=np.zeros([nb,nb],dtype=complex)
        block2=np.zeros([nb,nb],dtype=complex)
        block += np.random.rand(nb,nb)+1j*np.random.rand(nb,nb)
        block += np.conjugate(block.T)

        for l in range(Ops.shape[0]):
            block2 += np.dot(np.conjugate(Ops[l][ind_list[k]:ind_list[k+1],ind_list[k]:ind_list[k+1]].T),
                             np.dot(block,Ops[l][ind_list[k]:ind_list[k+1],ind_list[k]:ind_list[k+1]]))
        block2 /= Ops.shape[0]
        blocks.append(block2)
    mockH=scipy.linalg.block_diag(*blocks)
    eval2_i,evec2_i=np.linalg.eigh(mockH)
    evecf_i=np.dot(evec_i,evec2_i)

    for i in range(Ops.shape[0]):
        Ops[i] = np.dot(np.conjugate(evec2_i).T,np.dot(Ops[i],evec2_i))
    mat=np.mean(np.array([np.abs(Ops[i][:,:])**2 for i in range(24)]),axis=0)
    mat=np.round(np.abs(mat),decimals=6)
    indtab=[np.where(mat[i]>0)[0] for i in range(len(mat))]
    stab=[list(x) for x in set(tuple(x) for x in indtab)]
    sind=np.argsort([stab[i][0] for i in range(len(stab))])
    mylist = [stab[i] for i in sind]
    trlist=np.array([[np.trace(Ops[i][mylist[j]][:,mylist[j]]).real for i1,i in enumerate([23,15,9,0,3])] for j in range(len(mylist))])
    tr0list=np.array([np.trace(Ops[i]).real for i1,i in enumerate([23,15,9,0,3])])
    ichmat=1.0/24*np.array([[1,8,6,6,3],[1,8,-6,-6,3],[2,-8,0,0,6],[3,0,-6,6,-3],[3,0,6,-6,-3]])
    sumIrreps0=np.dot(ichmat,tr0list)
    annotTab=np.array([np.dot(ichmat,trlist[i]) for i in range(len(trlist))])
    print(annotTab.shape)
    sumtab=np.sum(annotTab,axis=1)
    print(sumtab)
    if np.max(np.round(sumtab))>1:
        print("Annotation failed: reducible representation present: "+str(np.max(sumtab)))
    annotRes=np.argmax(annotTab,axis=1)
    annotRes2=np.zeros(len(basis))
    for i in range(len(mylist)):
        for j in range(len(mylist[i])):
            annotRes2[mylist[i][j]]=annotRes[i].real
    counts=np.sum(annotTab,axis=0)
    #vals,counts=np.unique(np.argmax(np.array([np.dot(ichmat,trlist[i]) for i in range(len(trlist))]),axis=1),return_counts=True)
    if any(np.round(counts,6) != np.round(sumIrreps0,6)):
        print("Annotation failed: not all copies of irreps accounted for")
        #print(counts)
        print(sumIrreps0)
    Szevals=np.round(2*np.diag(edrixs.cb_op(Sop[2],evecf_i))).real
    S2evals=np.round((-1+np.sqrt(1+4*np.diag(edrixs.cb_op(S2op,evecf_i)).real)))
    L2evals=np.round((-1+np.sqrt(1+4*np.diag(edrixs.cb_op(L2op,evecf_i)).real))/2)
    t2gevals=np.round(np.diag(edrixs.cb_op(torLfock,evecf_i)).real,decimals=2)
    #Sinds=np.lexsort((t2gevals,annotRes2,Szevals,S2evals))
    if t_or_L=='L':
        Sinds=np.lexsort((Szevals,L2evals,annotRes2,S2evals))
    elif t_or_L=='t':
        Sinds=np.lexsort((t2gevals,Szevals,annotRes2,S2evals))
    evecf_i2=evecf_i[:,Sinds]

    if t_or_L=='L':
        annots=np.array([S2evals,annotRes2,L2evals,Szevals]).T[Sinds]
    elif t_or_L=='t':
        annots=np.array([S2evals,annotRes2,Szevals,t2gevals]).T[Sinds]
    #annots=np.array([S2evals,Szevals,annotRes2,t2gevals]).T[Sinds]

    return evecf_i2, Ops, annots,sumIrreps0

def getWeight(annots,basis,evecs,*,S2Q=None,SZQ=None,AnnotQ=None,t2gQ=None):
    inds=np.ones(basis.shape[0])>0
    if S2Q is not None:
        inds = (annots[:,0]==S2Q)
    if AnnotQ is not None:
        inds = (inds*(annots[:,1]==AnnotQ))
    if SZQ is not None:
        inds = (inds*(annots[:,2]==SZQ))
    if t2gQ is not None:
        inds = (inds*(annots[:,3]==t2gQ))
    inds = np.nonzero(inds)
    return np.sum(np.abs(np.dot(np.conjugate(basis.T)[inds],evecs))**2,axis=0)

def getWeightL(annots,basis,evecs,*,S2Q=None,L2Q=None,AnnotQ=None,SZQ=None):
    inds=np.ones(basis.shape[0])>0
    if S2Q is not None:
        inds = (annots[:,0]==S2Q)
    if AnnotQ is not None:
        inds = (inds*(annots[:,1]==AnnotQ))
    if L2Q is not None:
        inds = (inds*(annots[:,2]==L2Q))
    if SZQ is not None:
        inds = (inds*(annots[:,3]==SZQ))
    inds = np.nonzero(inds)
    return np.sum(np.abs(np.dot(np.conjugate(basis.T)[inds],evecs))**2,axis=0)