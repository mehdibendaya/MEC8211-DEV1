# -*- coding: utf-8 -*-

import numpy as np

# =============================================================================
# =============================================================================
# =====================  Premier cas : Régime permanent  ======================
# =============================================================================
# =============================================================================

def C_analytique(prm):
        
        """ Fonction qui calcule f(x) pour le premier cas
        Entrée : 
        - x : vecteur de position 
        Sortie :
        - y : vecteur contenant la valeur numérique de la fonction """
        r = np.linspace(0, prm.R, prm.n)
        y=(0.25*(prm.S/prm.D_eff)*(prm.R*prm.R)*(((r*r)/(prm.R*prm.R))-1))+prm.Ce
          
        return r,y
 

# =============================================================================
# =============================================================================
# =================  Premier cas : Schema d'ordre 1  ==========================
# =============================================================================
# =============================================================================


def PbB(prm):
    """ Fonction qui résout le systeme  
    Entrées:
        - prm : vecteur contenant la parametres globaux du systeme

    Sorties :
        - c : Matrice (array) qui contient la solution numérique
        - tps : vecteur (liste) qui contient les différents temps de résolution"""        

    dr = prm.dr #Pas en espace
    dt = prm.dt
    D_eff=prm.D_eff
    n  = prm.n
    r = np.linspace(0, prm.R, n) #Discrétisation en espace
    A = np.zeros([prm.n, prm.n]) #Matrice A
    b = np.zeros(prm.n) #Vecteur b
    t=0   
    tps=[0]
    err_t_tdt=10
    # Remplissage du centre de la matrice A et du vecteur b
    # c_t=C_analytique(prm)[1]+0.5*np.ones(prm.n)
    c_t=np.ones(n)
    bb=7
    a=(prm.Ce-bb)/prm.R
    c_t[:-1] = [a*r[i]+bb for i in range(n-1)]
    c_t[-1]=prm.Ce
    c=[c_t]
    # Remplissage du centre de la matrice A et du vecteur b   
    dr_inv=1/dr
    dt_D_eff=dt*D_eff
    dr2_inv=1/dr**2
    
    for i in range(1, n-1):
        A[i, i+1] = -dt_D_eff*(dr_inv/r[i]+dr2_inv)
        A[i, i] = 1+dt_D_eff*(dr_inv/r[i]+2*dr2_inv)
        A[i, i-1] = -dt_D_eff*dr2_inv
    
    A[-1, -1] = 1
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    
    while err_t_tdt>prm.err_t_tdt:
        
        b = np.zeros(n)
        b[1:n-1]=-dt*prm.S+c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce

        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        err_t_tdt=np.linalg.norm((c_t-c_tdt)/c_t)
        c_t=c_tdt
        t+=prm.dt
        tps.append(t)
        print(err_t_tdt)
        # print(A)
    return c_tdt,tps

# =============================================================================
# =============================================================================
# =====================  Deuxieme cas : Schema d'ordre 2  =====================
# =============================================================================
# =============================================================================  

def PbF(prm):
    """ Fonction qui résout le systeme  pour le deuxième cas
    Entrées:
        - prm : vecteur contenant la position 

    Sorties :
        - c : Matrice (array) qui contient la solution numérique
        - tps : vecteur (liste) qui contient les différents temps de résolution"""        

    dr = prm.dr #Pas en espace
    dt = prm.dt
    D_eff=prm.D_eff
    n  = prm.n
    r = np.linspace(0, prm.R, n) #Discrétisation en espace
    A = np.zeros([prm.n, prm.n]) #Matrice A
    b = np.zeros(prm.n) #Vecteur b
    t=0   
    tps=[0]
    err_t_tdt=10
    # Remplissage du centre de la matrice A et du vecteur b
    c_t=np.ones(n)
    b=7
    a=(prm.Ce-b)/prm.R
    c_t = C_analytique(prm)[1]

    # Remplissage du centre de la matrice A et du vecteur b    
    dr_inv=0.5/dr
    dt_D_eff=dt*D_eff
    dr2_inv=1/dr**2
    
    for i in range(1, prm.n-1):
        A[i, i+1] = -dt_D_eff*(dr_inv/r[i]+dr2_inv)
        A[i, i] = 1+dt_D_eff*(2*dr2_inv)
        A[i, i-1] = -dt*D_eff*(dr2_inv-dr_inv/r[i])
    
    A[-1, -1] = 1
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    while err_t_tdt>prm.err_t_tdt:
        b = np.zeros(n)
        b[1:n-1]=-dt*prm.S+c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce
        
        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        err_t_tdt=np.linalg.norm(c_t-c_tdt)
        c_t[:]=c_tdt[:]
        t+=prm.dt
        tps.append(t)

    return c_tdt,tps
