import numpy as np
from scipy.linalg import block_diag
from scipy import sparse
import time
import sys, os

class mesh2D:
    '''
    Mesh class for a two level mesh with prolongation and
    restrication operators 
    A 2 level mesh library
    p = point matrix Npx2 (x1, x2)
    e = edge  matrix Nex2 (p1, p2, Ie)
        Ie is a marker that is inherited to finer edges when mesh is
        refined. In lodTools.py, 
                      Ie < 0    Neumann boundary
                      Ie = 0    interior edge
                      Ie > 0    Dirichlet boundary
    et = edge to triangle, T- negative means boundary (t+, t-)
    t = triangle matrix, Ntx3 monotone index increasing (p1, p2, p3)
    
    Authors:
    D. Elfverson
    F. Hellman
    
    Contributors:
    D. Peterseim
    
    '''
    
    def __init__(self,p,t,e,te):
        self.p = p.copy() 
        self.t = t.copy()
        self.e = e.copy()
        self.te = te.copy()
        
    # Refine mesh
    def refine(self,times=1):
        """
        REFINE refines the current triangulation by dividing
        all triangles into 4 congruent triangles.

        Input:
            mesh:  current mesh
        
        Output: 
             The mesh object is modified after refinement
               P:  prolongation matrix
                   extrapolates grid functions from the coarse to the refined mesh
               R:  restriction matrix
                   restricts fine grid functions to coarse grid (nodal
                   interpolation)
        
        D. Peterseim 22-04-2008
        """

        i = 0
        while i < times:
            i = i + 1
            # Compute edges
            Np = np.size(self.p,0)
            Nt = np.size(self.t,0)
            e = self.e
            Ne = np.size(e,0)

            # pe - matrix mapping point pair (p1, p2) to edge index 
            data = np.hstack([np.arange(0,Ne),np.arange(0,Ne)])
            pe = sparse.csc_matrix((data, \
                                       (e[:,[0,1]].T.flatten(), \
                                        e[:,[1,0]].T.flatten())), shape=(Np, Np), dtype=int)
            
            # Construct Lagrange prolongation (P) and restriction (R)
            # matrix
            P1 = sparse.csc_matrix((np.ones(Ne), \
                                    (np.arange(0,Ne),e[:,0])), shape=(Ne, Np))
            P2 = sparse.csc_matrix((np.ones(Ne), \
                                    (np.arange(0,Ne),e[:,1])), shape=(Ne, Np))
            P = sparse.vstack([sparse.eye(Np),(P1+P2)/2]).tocsc()
            R = sparse.hstack([sparse.eye(Np),sparse.csc_matrix(([], ([], [])), shape=(Np, Ne))]).tocsc()
            # New nodes from the mid points of each edge
            pAppend = (self.p[e[:,0]] + self.p[e[:,1]] )/2
            pNew = np.vstack([self.p,pAppend])
            NpNew = Np+Ne
            
            emarker = np.arange(Np,Np+Ne) # global numbers of new nodes

            # refine each triangle into four triangles
            #     2             
            #    / \      
            #   5 - 4     
            #  / \ / \    
            # 0 - 3 - 1   
            p = np.zeros((Nt,6),int)
            p[:,0:3] = self.t
            p[:,3] = emarker[pe[self.t[:,0],self.t[:,1]]].astype(int)
            p[:,4] = emarker[pe[self.t[:,1],self.t[:,2]]].astype(int)
            p[:,5] = emarker[pe[self.t[:,2],self.t[:,0]]].astype(int)

            # ed - Nt x 3 matrix. Row is triangle. Column is new edge
            # in the order (3 -> 5),  (3 -> 4),  (4 -> 5)
            # ed[i,j] is 1 if the direction of edge is according to the 
            # numbering above. If the edge has opposite direction, 
            # ed[i,j] is -1.
            ed = 2*np.column_stack([p[:,3] < p[:,5], p[:,3] < p[:,4], p[:,4] < p[:,5]]).astype(int)-1

            # This is the order of the new triangles
            #      .    
            #     /2\   
            #    . - .  
            #   /0\3/1\ 
            #  . - . - .

            # We don't know that p[:,3] is less than p[:,5], sorting is needed.
            tNewUnsorted = np.vstack([np.column_stack([p[:,0], p[:,3], p[:,5]]), \
                                      np.column_stack([p[:,1], p[:,3], p[:,4]]), \
                                      np.column_stack([p[:,2], p[:,4], p[:,5]]), \
                                      np.column_stack([p[:,3], p[:,4], p[:,5]])]);
            tNew = np.sort(tNewUnsorted,axis=1)

            # Update e: Let all new edges inherit the edge index (e[:,2]). New edges get index 0
            # First two parts: split edge. 
            # Last three parts: the three new edges within the triangles
            eNewUnsorted = np.vstack([np.column_stack([e[:,0], emarker]), \
                                      np.column_stack([emarker, e[:,1]]), \
                                      np.column_stack([p[:,3], p[:,5]]),  \
                                      np.column_stack([p[:,3], p[:,4]]),  \
                                      np.column_stack([p[:,4], p[:,5]])])
            eNew = np.sort(eNewUnsorted,axis=1)

            # Add edge index last
            eNew = np.column_stack([eNew, np.hstack([e[:,2], e[:,2], np.zeros(3*Nt,int)])])

            NeNew = np.size(eNew, 0)
            data = np.hstack([np.arange(0,NeNew),np.arange(0,NeNew)])
            peNew = sparse.csc_matrix((data, \
                                       (eNew[:,[0,1]].T.flatten(), \
                                        eNew[:,[1,0]].T.flatten())), shape=(NpNew, NpNew), dtype=int)

            # Given te, compute teNew
            # te = (e0, e1, e2)
            te = self.te
            teNew = np.vstack([peNew[tNew[:,1], tNew[:,2]], \
                               peNew[tNew[:,2], tNew[:,0]], \
                               peNew[tNew[:,0], tNew[:,1]]])
            teNew = np.asarray(teNew.T)

            self.e = eNew
            pOld = self.p
            self.p = pNew
            tOld = self.t
            self.t = tNew
            self.te = teNew

            # Construct Raviart-Thomas prolongation matrix (PRT)
            # First two identity matrices stacked on top of each other.
            # Then three matrices, P0, P1, P2 constructed from te are summed
            tInd3 = np.arange(0,3*Nt)
            pFactors = [1./2, 1./4, 1./4]
            #pFactors = [1, 1./2, 1./np.sqrt(2)]
            INe = sparse.eye(Ne)*pFactors[0]
            # P0 is the 1/2-edge
            P0 = sparse.csc_matrix((pFactors[1]*np.repeat([1, 1, -1],Nt)*ed.T.flatten(), \
                                    (tInd3, te.T.flatten())), \
                                    shape=(3*Nt, Ne))
            # P1 and P2 are the 1/sqrt(2)-edges
            P1 = sparse.csc_matrix((pFactors[2]*np.repeat([-1, 1, -1],Nt)*ed.T.flatten(), \
                                    (tInd3, te[:,[2, 0, 1]].T.flatten())), \
                                    shape=(3*Nt, Ne))

            P2 = sparse.csc_matrix((pFactors[2]*np.repeat([1, 1, 1],Nt)*ed.T.flatten(), \
                                    (tInd3, te[:,[1, 2, 0]].T.flatten())), \
                                    shape=(3*Nt, Ne))

            PRT = sparse.vstack([INe, -INe, P0+P1+P2])            
            
            if hasattr(self, 'PRT'):
                self.PRT = PRT*self.PRT
                # print(f"hasattr: PRT")

            if hasattr(self, 'HE'):
                HE = sparse.vstack([0.5*sparse.eye(Ne), -0.5*sparse.eye(Ne), sparse.csc_matrix((3*Nt, Ne))], dtype='float64').tocsc()
                self.HE = HE*self.HE
                self.IRT = self.HE.T.copy()
                self.IRT.data = np.sign(self.IRT.data)
                # print(f"self.IRT.data: {self.IRT.data.shape}")
                print(f"mesh.IRT.shape: {self.IRT.shape}")
                print(f"hasattr: HE")

            if hasattr(self, 'P'):
                self.P = P*self.P
                # print(f"hasattr: P")
                
            if hasattr(self, 'R'):
                self.R = self.R*R
                # print(f"hasattr: R")
                
            if hasattr(self, 'H'):
                H = sparse.vstack([sparse.eye(Nt),sparse.eye(Nt),sparse.eye(Nt),sparse.eye(Nt)],dtype='int').tocsc()            
                self.H = H*self.H
                self.IQ = self.H.T.copy()
                # print(f"hasattr: H")

            if hasattr(self, 'PDG'):
                tp0 = np.array([[  1,  0, 0],[0.5, 0.5,  0],[0.5,  0,0.5]])
                tm0 = np.array([[  1,  0, 0],[0.5,   0,0.5],[0.5,0.5,  0]])                
                tm1 = sparse.csc_matrix(np.array([[  0,  1, 0],[0.5, 0.5,  0],[  0,0.5,0.5]]))
                tp1 = sparse.csc_matrix(np.array([[  0,  1, 0],[  0, 0.5,0.5],[0.5,0.5,  0]]))
                tp2 = sparse.csc_matrix(np.array([[  0,  0, 1],[0.5,   0,0.5],[  0,0.5,0.5]]))
                tm2 = sparse.csc_matrix(np.array([[  0,  0, 1],[  0, 0.5,0.5],[0.5,  0,0.5]]))
                tpmp3 = sparse.csc_matrix(np.array([[0.5,0.5, 0],[0.5,  0,0.5],[  0,0.5,0.5]]))
                tpmm3 = sparse.csc_matrix(np.array([[0.5,0.5, 0],[  0,0.5,0.5],[0.5,  0,0.5]]))
                tmpm3 = sparse.csc_matrix(np.array([[0.5, 0,0.5],[ 0.5,0.5, 0],[  0, 0.5,0.5]]))

                tPCycleC = self.__compute_orientation(tOld,pOld)
                tPCycle  = self.__compute_orientation(tNew,pNew)

                Indm0 = (tPCycle[0*Nt:1*Nt])*(1-tPCycleC) + (1-tPCycle[0*Nt:1*Nt])*tPCycleC
                Indp0 = 1- Indm0
                Indm1 = (tPCycle[1*Nt:2*Nt])*(1-tPCycleC) + (1-tPCycle[1*Nt:2*Nt])*tPCycleC
                Indp1 = 1 -Indm1
                Indm2 = (tPCycle[2*Nt:3*Nt])*(1-tPCycleC) + (1-tPCycle[2*Nt:3*Nt])*tPCycleC
                Indp2 = 1 - Indm2

                Indpmp3 = Indp0*Indm1*Indp2
                Indpmm3 = Indp0*Indm1*Indm2
                Indmmp3 = Indm0*Indm1*Indp2
                
                T0 = sparse.kron(sparse.diags(Indp0,0),tp0) + sparse.kron(sparse.diags(Indm0,0),tm0)
                T1 = sparse.kron(sparse.diags(Indp1,0),tp1) + sparse.kron(sparse.diags(Indm1,0),tm1)
                T2 = sparse.kron(sparse.diags(Indp2,0),tp2) + sparse.kron(sparse.diags(Indm2,0),tm2)
                T3 = sparse.kron(sparse.diags(Indpmp3,0),tpmp3) + sparse.kron(sparse.diags(Indpmm3,0),tpmm3) + sparse.kron(sparse.diags(Indmmp3,0),tmpm3)
                    
                PDG = sparse.vstack([T0,T1,T2,T3]).tocsc()
                
                self.PDG = PDG*self.PDG
                # print(f"hasattr: PDG")

                
                    
    def create_pt(self):
        return self.__create_pt()
    
    def __create_pt(self):
        """Create a point to elemnt map, can be used to create node patches. 
        See __create_t for more infomration how to create larger patches"""

        Np = np.size(self.p,0)
        Nt = np.size(self.t,0)
        
        row = np.zeros(3*Nt)
        row[::3] = np.arange(0,Nt)
        row[1::3] = np.arange(0,Nt)
        row[2::3] = np.arange(0,Nt)
        col = self.t[:,0:3].flatten()
        val = np.ones(3*Nt)
        pt = sparse.csc_matrix((val, (row, col)), shape=(Nt, Np))

        return pt
        
    def __create_tt(self):
        """Create a triangle to triangle incidence matrix, via vertices.  Can
        be used to expand patches.

        tt[t1, t2] = t1 is incident to t2 via a vertex

        To create a 1-layer patch of a triangle t0, let vector l0 =
        [0, ..., 1, ..., 0], with the 1 on the t0:th position. Get the
        1-layer and 2-layer patches by
        
          l1 = tt*l0
          l2 = tt*l1.

        A k-layer patch matrix for any triangle can be generate by
        
          Lk = tt*tt*...*tt

        with k-1 number of matrix multipliations. Lk[:,t0] is the
        k-layer patch centered around triangle t0.
        """
        pt = self.__create_pt()
        tp = pt.T
        tt = pt*tp

        return tt

    def create_tt(self):
        return self.__create_tt()

    def create_tet(self):
        """Create a triangle to triangle incidence matrix, via edges. Can
        be used to expand integration subdomains.

        tt[t1, t2] = t1 is incident to t2 via a vertex
        """
        te = self.te
        nnz = np.size(te)
        teMatrix = sparse.csc_matrix(([1]*nnz,
                                      (te.flatten(),
                                       np.repeat(np.arange(nnz//3),3))))
        etMatrix = teMatrix.T
        tet = etMatrix*teMatrix
        
        return tet

    def create_pe(self):
        """Create a point to edge incidence matrix."""

        row = np.repeat(np.arange(self.e.shape[0]), 2)
        col = self.e[:,:2].flatten()
        data = [1]*row.size
        pe = sparse.csc_matrix((data, (row, col)), shape=(self.e.shape[0], self.p.shape[0]))
        return pe
    
    def __compute_orientation(self,t,p):
        # tCycle is 0 if the triangle indexing is counter-clockwise, 1 otherwise
        tPCycle = ((p[t[:,1],0]-p[t[:,0],0])*(p[t[:,2],1]-p[t[:,0],1]) - \
                   (p[t[:,1],1]-p[t[:,0],1])*(p[t[:,2],0]-p[t[:,0],0])) < 0
        return tPCycle

    def __create_cg2dg(self):
        """ Maps a continuous dofs to a discontinuous dofs"""
        Nt = np.size(self.coarse.t,0)
        Np = np.size(self.coarse.p,0)
        row = np.arange(0,3*Nt)
        col = self.coarse.t.flatten()
        cg2dg = sparse.csc_matrix(([1]*3*Nt, (row, col)), shape=(3*Nt, Np))
        return cg2dg
    
    def setCoarse(self):
        """
        Make this the coarse mesh
        """
        self.P = sparse.eye(np.size(self.p,0)).tocsc()
        self.R = sparse.eye(np.size(self.p,0)).tocsc()
        self.H = sparse.eye(np.size(self.t,0)).tocsc()

        self.PRT = sparse.eye(np.size(self.e,0)).tocsc()
        self.HE = sparse.eye(np.size(self.e,0)).tocsc()

        self.PDG = sparse.eye(3*np.size(self.t,0)).tocsc()
        
        self.coarse = mesh2D(self.p,self.t,self.e,self.te)
        self.coarse.pt = self.__create_pt()
        self.coarse.tt = self.__create_tt()

        self.coarse.cg2dg = self.__create_cg2dg()

    def setFineGivenCoarse(self, meshCoarse):
        """
        Make this the fine mesh, where the coarse mesh is given by meshCoarse.
        NOTE: P, R are NOT YET implemented (but it is easy to do if needed)
        """
        self.P = None
        self.R = None
        
        H, PRT, IRT, IRT_new = create_hierarchy(meshCoarse, self)
        self.H = H
        self.PRT = PRT
        self.IRT = IRT
        self.IRT_new = IRT_new
        # BUG: FOR MESHES WITH DIFFERENT SIZES OF THE FINE ELEMENTS,
        # THIS IS NOT THE DEFINITION IF IQ
        self.IQ = H.T.copy()

        self.coarse = meshCoarse
        self.coarse.pt = meshCoarse.__create_pt()
        self.coarse.tt = meshCoarse.__create_tt()

    def create_et(self, tInd=None):
        """
        Create a NePatch x 2 edge to triangle incidence matrix for a patch.
        tInd contains the triangle indices of the patch. NePatch is
        the number of edges in the patch.

        Returns: et, eInd

        et[e,0] = triangle index of adjacent triangle to edge e in
                  direction of negative normal

        et[e,1] = triangle index of adjacent triangle to edge e in
                  direction of normal

        pLocal[e,0] = returns local index in et[e,0] which is not on
                      the edge e

        pLocal[e,1] = returns local index in et[e,1] which is not on
                      the edge e


        If tInd is not given, all triangles are considered. This way
        it is possible to find the exterior and interior edges of
        patch.

        """

        if tInd is None:
            Nt = np.size(self.t,0)
            tInd = np.arange(0,Nt)
        else:
            Nt = np.size(tInd,0)

        teP = self.te[tInd,:]
        tP = self.t[tInd,:]
        p = self.p

        # tCycle is 0 if the triangle indexing is counter-clockwise, 1 otherwise
        tPCycle = self.__compute_orientation(tP,p)

        Ne = np.size(self.e,0)
        data = np.hstack([tInd, tInd, tInd])
        # Add +1 on the triangle indices, since we subtract by 1 in the end to get -1
        # when a triangle is missing on the other side of the edge
        etP = sparse.csc_matrix((data+1, \
                                 (np.hstack([teP[:,0], \
                                             teP[:,1], \
                                             teP[:,2]]), \
                                  np.hstack([tPCycle, \
                                             1-tPCycle, \
                                             tPCycle]))), shape=(Ne, 2), dtype=int)
        eInd = np.unique(etP.indices)
        etP = etP[eInd,:].todense()-1
        etP = np.array(etP)

        NetP = np.size(etP,0)
        
        # Find the orientation of the local indexes
        t = self.t
        eP = self.e[eInd]
        pLocalP = -np.ones((NetP,2), dtype=int)
        ind0Interior = etP[:,0]>-1
        et0 = etP[ind0Interior,0]
        pLocal0 = -1*np.ones(et0.shape[0], dtype=int)
        pLocal0[(eP[ind0Interior,0]==t[et0,0])&(eP[ind0Interior,1]==t[et0,1])] = 2
        pLocal0[(eP[ind0Interior,0]==t[et0,0])&(eP[ind0Interior,1]==t[et0,2])] = 1
        pLocal0[(eP[ind0Interior,0]==t[et0,1])&(eP[ind0Interior,1]==t[et0,2])] = 0
        pLocalP[ind0Interior,0] = pLocal0
        
        et1 = etP[:,1]
        ind1Interior = etP[:,1]>-1
        et1 =  etP[ind1Interior,1]
        pLocal1 = -1*np.ones(et1.shape[0], dtype=int)
        pLocal1[(eP[ind1Interior,0]==t[et1,0])&(eP[ind1Interior,1]==t[et1,1])] = 2
        pLocal1[(eP[ind1Interior,0]==t[et1,0])&(eP[ind1Interior,1]==t[et1,2])] = 1
        pLocal1[(eP[ind1Interior,0]==t[et1,1])&(eP[ind1Interior,1]==t[et1,2])] = 0
        pLocalP[ind1Interior,1] = pLocal1

        
        return etP, eInd, pLocalP
        

    def plot(self,K=None, tInd=None, color=None, width=None, keepaxis=False):
        import matplotlib.pyplot as pla
        """
        Plot the mesh
        """
        if tInd is None:
            tInd = np.arange(0,np.size(self.t,0))
        if color is None:
            color = [0, 0, 0, 1]
        if width is None:
            width = 1

        fig = pla.triplot(self.p[:, 0], self.p[:, 1], self.t[tInd,:], color=color, linewidth=width)
            
        if (K != None):
            fig = pla.plot(self.p[self.t[K][np.array([0,1,2,0])],0], self.p[self.t[K][np.array([0,1,2,0])],1], 'ro-', linewidth=2.0)
        if not keepaxis:
            pla.axis('off')
        for tInd in np.arange(0,np.size(self.t,0)):
            tPnt = np.mean(self.p[self.t[tInd]], axis=0)
            pla.text(tPnt[0], tPnt[1], str(tInd), color='red') # , fontsize=12

    def plotEdges(self,eInd=None,eColors=None):
        import matplotlib.pyplot as pl
        import matplotlib.collections as mc
        if eInd is None:
            eInd = np.arange(0,np.size(self.e,0))

        lines = [[(self.p[self.e[eI,0],0], self.p[self.e[eI,0],1]), \
                  (self.p[self.e[eI,1],0], self.p[self.e[eI,1],1])] \
                 for eI in eInd]
        lc = mc.LineCollection(lines,colors=eColors,linewidth=2)
        ax = pl.gca()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        for eInd in np.arange(0,np.size(self.e,0)):
            ePnt = np.mean(self.p[self.e[eInd][:2]], axis=0)
            pl.text(ePnt[0], ePnt[1], str(eInd), color='blue') # , fontsize=12
        
    def plotVertices(self):
        import matplotlib.pyplot as pl
        for pi in np.arange(0,np.size(self.p,0)):
            pl.text(self.p[pi,0], self.p[pi,1], str(pi), color='green') # , fontsize=12

    def show(self):
        import matplotlib.pyplot as pl
        pl.show()

"""
Create a mesh and mesh.coarse structure based on a coarse and a
conforming fine mesh. 
"""
def create_hierarchy(meshCoarse, meshFine):

    t = meshFine.t
    p = meshFine.p
    e = meshFine.e
    te = meshFine.te

    T = meshCoarse.t
    P = meshCoarse.p
    E = meshCoarse.e
    ET,_,_ = meshCoarse.create_et()

    # Step 1: Create H-matrix
    # Map each fine element to the coarse element containing it.
    def create_H(meshCoarse, meshFine):
        # Check if tri is inside Tri by checking that the midpoint of
        # tri is inside Tri (conformity assumed)
        def areInside(Tri):
            # m = t midpoints
            m = np.mean(p[t,:],1)
            # vectors from P0 to P1 and so on
            P0P1 = P[Tri[1]] - P[Tri[0]]
            P1P2 = P[Tri[2]] - P[Tri[1]]
            P2P0 = P[Tri[0]] - P[Tri[2]]
            # vectors from corners to midpoint of fine triangles
            P0m = m - P[Tri[0]]
            P1m = m - P[Tri[1]]
            P2m = m - P[Tri[2]]
            # Check cross product between edges and from corner to
            # midpoint
            c0 = np.cross(P0P1, P0m)
            c1 = np.cross(P1P2, P1m)
            c2 = np.cross(P2P0, P2m)
            # If all signs are equal of the cross products, the
            # midpoint is within the coarse triangle
            tIndInside = np.where((np.sign(c1)==np.sign(c0)) & (np.sign(c2)==np.sign(c0)))[0]
            return tIndInside

        # Construct H-matrix (NT x Nt)
        Hrow = np.zeros(np.size(t,0),dtype=int)
        Hcol = np.zeros(np.size(t,0),dtype=int)
        i = 0
        TInd = 0
        for Tri in T:
            tIndInside = areInside(Tri)
            n = np.size(tIndInside)
            Hrow[i:i+n] = tIndInside
            Hcol[i:i+n] = TInd
            i = i + n
            TInd = TInd + 1
        
        NT = np.size(T,0)
        Nt = np.size(t,0)
        H = sparse.csc_matrix((0*Hrow+1, (Hrow, Hcol)), shape=(Nt, NT))
        return H

    # Step 2: Create PRT matrix
    # The prolongation matrix from coarse to fine edges (Ne x NE), 
    # mapping coarse RT basis functions to fine edges. 
    def create_PRT(meshCoarse, meshFine, H):
        NE = np.size(E,0)
        Ne = np.size(e,0)

        PRTrow = []
        PRTcol = []
        PRTval = []
        # Loop over coarse edges
        for EInd in range(NE):
            Edge = E[EInd,:].squeeze()
            EMeasure = np.linalg.norm(P[Edge[1]]-P[Edge[0]])
            # First collect row, cols and vals in these local
            # lists
            PRTrowE = []
            PRTcolE = []
            PRTvalE = []
            # Loop over t+-triangle and t--triangle
            for i in range(2):
                TInd = ET[EInd,i]
                if TInd<0:
                    continue

                Tri = T[TInd,:].squeeze()
                TMeasure = 0.5*np.abs(np.cross(P[Tri[1]]-P[Tri[0]], \
                                               P[Tri[2]]-P[Tri[1]]))
                # Sigma determines whether the normal is outgoing or ingoing.
                # sigma = -1 for i = 0; sigma = 1 for i = 1.
                sigma = (-1)**(i-1)
                # P0 is the point opposite to the edge of the coarse
                # triangle (used to define the RT basis function)
                P0Ind = np.setdiff1d(Tri, Edge[:2], True)
                P0 = P[P0Ind].squeeze()
                # Get indices of all fine triangles in coarse triangle
                tInds = sparse.find(H[:,TInd])[0]
                # Get all fine edges
                eInds = np.unique(te[tInds,:].flatten().squeeze())
                for eInd in eInds:
                    # xm is midpoint of edge
                    edge = e[eInd].squeeze()
                    xm = np.mean(p[edge[:2]],0)
                    eMeasure = np.linalg.norm(p[edge[1]]-p[edge[0]])
                    tangent = p[edge[1]] - p[edge[0]]
                    # The computed normal is the counter-clockwise rotated tangents i.e. (x,y) -> (-y,x).
                    normal = np.cross(tangent, [0, 0, -1])[:2]/ \
                             np.linalg.norm(tangent)
                    # Compute value of coarse basis function at edge
                    # midpoint
                    v = sigma*eMeasure/(2*TMeasure)*np.dot(normal, xm-P0)
                    PRTrowE.append(eInd)
                    PRTcolE.append(EInd)
                    PRTvalE.append(v)
            # Ensure each fine edge (eInd) is only added once per coarse edge (EInd) in the PRT matrix, 
            # even if the fine edge is shared by two adjacent coarse elements.
            PRTrowE, indices = np.unique(PRTrowE, return_index=True)
            PRTcolE = [PRTcolE[idx] for idx in indices]
            PRTvalE = [PRTvalE[idx] for idx in indices]
            PRTrow.extend(PRTrowE)
            PRTcol.extend(PRTcolE)
            PRTval.extend(PRTvalE)

        PRT = sparse.csc_matrix((PRTval, (PRTrow, PRTcol)), shape=(Ne, NE))
        return PRT

    # Step 3: Create IRT
    # Identify fine edges lying on coarse edges and interpolate their contributions.
    def create_IRT(meshCoarse, meshFine, PRT):
        NE = np.size(E,0)
        Ne = np.size(e,0)

        IRTrow = []
        IRTcol = []
        IRTval = []

        epsilon = 1e-10

        # Loop over coarse edges
        for EInd in range(NE):
            Edge = E[EInd,:].squeeze()
            P0 = P[Edge[0]]
            P1 = P[Edge[1]]
            # Find potential fine edges
            eInd = sparse.find(PRT[:,EInd])[0]
            eMidPoints = np.mean(p[e[eInd,:2],:],1)
            vLine = P1-P0
            vTest = eMidPoints-P0
            vNorm = np.linalg.norm(vLine)*np.linalg.norm(vTest,axis=1)
            # If this area is ~0, then the midpoint of fine
            # edge is on coarse edge (extended line)
            area = np.abs(np.cross(vLine, vTest))
            P01 = P[Edge[:2]]
            onEdge = (area/vNorm <= epsilon) & np.all((np.min(P01,0)-epsilon <= eMidPoints) & (eMidPoints <= np.max(P01,0)+epsilon),1)
            eIndOnEdge = eInd[onEdge]
            # Val is length relationship fine/coarse
            val = np.sign(np.dot(p[e[eIndOnEdge,1]]-p[e[eIndOnEdge,0]], P1-P0))
            IRTrow.extend(0*eIndOnEdge + EInd)
            IRTcol.extend(eIndOnEdge)
            IRTval.extend(val)

        IRT = sparse.csc_matrix((IRTval, (IRTrow, IRTcol)), shape=(NE, Ne), dtype=np.float64)
        return IRT

    H = create_H(meshCoarse, meshFine)
    PRT = create_PRT(meshCoarse, meshFine, H)
    IRT = create_IRT(meshCoarse, meshFine, PRT)
    return H, PRT, IRT

def create_e(p, t):
    """Create e map from p and t. All edge identifiers are set to 0."""
    Nt = np.size(t,0)
    Np = np.size(p,0)

    factor = 2.**np.ceil(np.log2(Np))
    eFloat = []
    for tInd in range(Nt):
        for pInd in range(0,3):
            e1 = t[tInd, pInd%3]
            e2 = t[tInd, (pInd+1)%3]
            # e12Float is a floating point number with integer part
            # equal to e1 and fraction part equal to e2/shift. 
            e12Float = min(e1,e2) + max(e1,e2)/factor
            eFloat.append(e12Float)

    eFloatUnique = np.unique(np.array(eFloat))
    Ne = eFloatUnique.size
    e = np.array([eFloatUnique.astype(int),
                  ((eFloatUnique%1)*factor).astype(int),
                  np.zeros(Ne,dtype=int)]).T
    return e

def create_te(t, e):
    """Create te map from t and e"""
    Nt = np.size(t,0)

    eNoId = e[:,:2]

    factor = 2.**np.ceil(np.log2(np.max(eNoId)))
    eFloat = eNoId[:,0] + eNoId[:,1]/factor

    # Really slow brute force version without sorting
    # To be used for small meshes only
    te = np.zeros((Nt,3),dtype=int)
    for tInd in range(Nt):
        e12Float0 = t[tInd,1] + t[tInd,2]/factor
        e12Float1 = t[tInd,0] + t[tInd,2]/factor
        e12Float2 = t[tInd,0] + t[tInd,1]/factor
        e0Ind = np.where(eFloat==e12Float0)[0]
        e1Ind = np.where(eFloat==e12Float1)[0]
        e2Ind = np.where(eFloat==e12Float2)[0]
        te[tInd,0] = e0Ind
        te[tInd,1] = e1Ind
        te[tInd,2] = e2Ind
    return te

def unitSquareMN(M, N):
    """
    Create an M x N mesh of unitsquare (northwest-southeast diagonals)
    """
    X, Y = np.meshgrid(np.linspace(0,1,M+1), np.linspace(0,1,N+1))
    p = np.column_stack([X.flatten(), Y.flatten()])
    tSquareFlat = np.array([0, 1, M+1, 1, M+1, M+2])
    tRowFlat = np.arange(0,M)[...,np.newaxis] + tSquareFlat[np.newaxis,...]
    tRowFlatFlat = tRowFlat.flatten()
    tFlat = (M+1)*np.arange(0, N)[...,np.newaxis] + tRowFlatFlat[np.newaxis,...]
    t = np.reshape(tFlat, (2*M*N, 3))

    e = create_e(p, t)
    te = create_te(t, e)
    mesh = mesh2D(p,t,e,te)
    return mesh

def LShape(refine=0):
    """
    Create an inital Lshape domain with number of refinements
    """
    if refine == 0:
        raise Exception("LShape not defined for refine=0, need atleast 1.")
    p = np.array([[0.5, 0.5], [1., 0.5], [1., 1.], [0.5, 1.],[0., 1.],[0., 0.5],[0., 0.],[0.5, 0.]]).astype(float)
    t = np.array([[0, 1, 2], [0, 2, 3], [3, 4, 5], \
                          [0, 3, 5], [5, 6, 7], [0, 5, 7]]).astype(int)

    e = np.array([[0, 1, 1],[1, 2, 2],[2, 3, 3],[3, 4, 3],[4, 5, 4], \
                  [5, 6, 4],[6, 7, 5],[0, 7, 6], \
                  [5, 7, 0], [0, 5, 0], [3, 5, 0], [0, 3, 0], [0, 2, 0]]).astype(int)

    te = np.array([[1, 12, 0], \
                   [2, 11, 12], \
                   [4, 10, 3], \
                   [10, 9, 11], \
                   [6, 8, 5], \
                   [8, 7, 9]])

# This is the old version. Seems to be incorrect.
#    te = np.array([[1, 12, 0], \
#                   [2, 11, 12], \
#                   [4, 10, 3], \
#                   [9, 11, 10], \
#                   [6, 8, 5], \
#                   [7, 9, 8]])

    mesh = mesh2D(p,t,e,te)
    i = 1
    while i < refine:
        i = i + 1
        mesh.refine()
    return mesh

def unitSquare(refine=0):
    """
    Create an inital unit square with number of refinements
    """
    p = np.array([[0., 0.],[1., 0.], [1., 1.], [0., 1.]]).astype(float)
    t = np.array([[0, 1, 3], [1, 2, 3]]).astype(int)
    e = np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 3, 1], \
                  [1, 3, 0]]).astype(int)
    te = np.array([[4, 3, 0], \
                   [2, 4, 1]]).astype(int)

    mesh = mesh2D(p,t,e,te)
    i = 0
    while i < refine:
        i = i + 1
        mesh.refine()
    return mesh

def unitSquare2(refine=0):
    """
    Create an inital unit square with number of refinements
    """
    p = np.array([[0., 0.],[1., 0.], [1., 1.], [0., 1.], [0.5, 0.5]]).astype(float)
    t = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]).astype(int)
    e = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3], [0, 3, 4], \
                  [0, 4, 0], [1, 4, 0], [2, 4, 0], [3, 4, 0]]).astype(int)
    te = np.array([[5, 4, 0], \
                   [6, 5, 1], \
                   [7, 6, 2], \
                   [7, 4, 3]]).astype(int)

    mesh = mesh2D(p,t,e,te)
    i = 0
    while i < refine:
        i = i + 1
        mesh.refine()
    return mesh


def slitDomain(refine=0):
    """
    Create an inital unit square with number of refinements
    """
    p = np.array([[0., 0.], [0.5, 0.], [0.5, 0.], [1., 0.], [1., 0.5], [1., 1.], \
                  [0.5, 1.],[0., 1.], [0., 0.5], [0.5, 0.5]]).astype(float)
    t = np.array([[0, 1, 9], [2, 3, 9], [3, 4, 9], [4, 5, 9], \
                  [5, 6, 9], [6, 7, 9], [7, 8, 9], [0, 8, 9]]).astype(int)
    e = np.array([[0, 1, 1], [2, 3, 1], [3, 4, 2], [4, 5, 2], \
                  [5, 6, 3], [6, 7, 3], [7, 8, 4], [0, 8, 4], \
                  [0, 9, 0], [1, 9, 5], [2, 9, 5], [3, 9, 0], \
                  [4, 9, 0], [5, 9, 0], [6, 9, 0], [7, 9, 0], [8, 9, 0]]).astype(int)
    te = np.array([[9 ,  8, 0], \
                   [11, 10, 1], \
                   [12, 11, 2], \
                   [13, 12, 3], \
                   [14, 13, 4], \
                   [15, 14, 5], \
                   [16, 15, 6], \
                    [16,  8, 7]]).astype(int)
    mesh = mesh2D(p,t,e,te)
    i = 1
    while i < refine:
        i = i + 1
        mesh.refine()
    return mesh

def referenceTriangle(refine=0):
    """
    Create a reference triangle
    """
    p = np.array([[0., 0.],[1., 0.], [0., 1.]]).astype(float)
    t = np.array([[0, 1, 2]]).astype(int)
    e = np.array([[1, 2, 1], [0, 2, 1], [0, 1, 1]]).astype(int)
    te = np.array([[0, 1, 2]]).astype(int);
    mesh = mesh2D(p,t,e,te)
    i = 0
    while i < refine:
        i = i + 1
        mesh.refine()
    return mesh

# Tests
def testConsistencyBetween_e_t_te_et():
    refine = 5
    mesh = LShape(refine)
    
    Nt = np.size(mesh.t,0)
    te = mesh.te
    # Check that points to edges of triangle are also points
    # of triangle
    for tInd in np.arange(0,Nt):
        pointsFrom_te = mesh.e[te[tInd,0:2],0:1].flatten()
        pointsFrom_t  = mesh.t[tInd,:].flatten()
        if np.size(np.setdiff1d(pointsFrom_te, pointsFrom_t),0) != 0:
            raise(Exception("Test failed: te, e and t inconsistent"))
            
    # Check et and te are consistent
    et,_,_ = mesh.create_et()
    for tInd in np.arange(0,Nt):
        if np.sum(et[te[tInd,:]]==tInd) != 3:
            raise(Exception("Test failed: te and et inconsistent"))
            
if __name__ == "__main__":

    # Tests
    testConsistencyBetween_e_t_te_et()

    t = time.time()
    refine = 3
    mesh = unitSquare(refine)
    mesh.setCoarse()
    mesh.refine(refine)
    print(time.time()-t)

    t = time.time()
    ttCoarse = mesh.coarse.tt
    twoLayerTCoarse = ttCoarse*ttCoarse
    print(time.time()-t)

    t = time.time()
    twoLayerTFine = mesh.H*twoLayerTCoarse
    print(time.time()-t)
    K = 55
    mesh.plot(None, twoLayerTFine[:,K].nonzero()[0])
    mesh.coarse.plot(K)

    # Fix below!
    K = 63
    H = 2**(-refine)    
    L = np.ceil(np.log(2**(4))).astype(int)

    mesh.show()


