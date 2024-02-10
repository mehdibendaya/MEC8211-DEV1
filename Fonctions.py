# -*- coding: utf-8 -*-

import numpy as np

# =============================================================================
# =============================================================================
# =====================  Premier cas : Régime permanent  ======================
# =============================================================================
# =============================================================================

def C_analytique(prm):
        
        """ Fonction qui calcule la solution analytique
        Entrée : 
        - prm : classe contenant les donnees du probleme
        Sortie :
        - y : vecteur contenant la valeur numérique de la fonction 
        - r : vecteur contenant le domaine descretise 
        """
        r = np.linspace(0, prm.R, prm.n)
        y=(0.25*(prm.S/prm.D_eff)*(prm.R*prm.R)*(((r*r)/(prm.R*prm.R))-1))+prm.Ce
          
        return r,y
 

# =============================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 1  ======================
# =============================================================================
# =============================================================================  
 
# ============================================================================= 
# ==============================Regime transitoire=============================
# ============================================================================= 
def PbB(prm):
    from time import time
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

    c_t=np.ones(n)
    c_t[:-1] = [0 for i in range(n-1)]
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
    
    i=0
    start = time()
    
    
    b[1:n-1]=-dt*prm.S+c_t[1:n-1]
    b[0] = 0
    b[-1] = prm.Ce

    # Résolution du système matriciel
    c_tdt = np.linalg.solve(A, b)
    c_t=c_tdt
    t+=prm.dt
    tps.append(t)
    
    while err_t_tdt>prm.err_t_tdt:
        
        b[1:n-1]=-dt*prm.S+c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce

        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        err_t_tdt=np.linalg.norm((c_t-c_tdt)/c_t)
        c_t=c_tdt
        t+=prm.dt
        tps.append(t)
        i+=1
        if i%10000==0:
            duration = time() - start
            print(duration,err_t_tdt)
            start = time()
        # print(A)
    return c_tdt,tps

# ============================================================================= 
# ==============================Regime stationnaire============================
# ============================================================================= 
def PbB_S(prm):
    from time import time
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
    c_t[:-1] = [0 for i in range(n-1)]
    c_t[-1]=prm.Ce
    c=[c_t]
    # Remplissage du centre de la matrice A et du vecteur b    
    
    dr_inv=1/dr
    dr2_inv=1/dr**2
    
    for i in range(1, n-1):
        A[i, i+1] = -D_eff*(dr_inv/r[i]+dr2_inv)
        A[i, i] = D_eff*(dr_inv/r[i]+2*dr2_inv)
        A[i, i-1] = -D_eff*dr2_inv
    
    A[-1, -1] = 1
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    i=0
    start = time()
    
    b[1:n-1]=-prm.S
    b[0] = 0
    b[-1] = prm.Ce

    # Résolution du système matriciel
    c_tdt = np.linalg.solve(A, b)
    c_t=c_tdt
    t+=prm.dt
    tps.append(t)
    
    
    while err_t_tdt>prm.err_t_tdt:
        
        b = np.zeros(n)
        b[1:n-1]=-prm.S
        b[0] = 0
        b[-1] = prm.Ce

        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        err_t_tdt=np.linalg.norm((c_t-c_tdt)/c_t)
        c_t=c_tdt
        t+=prm.dt
        tps.append(t)
        i+=1
        if i%10000==0:
            duration = time() - start
            print(duration,err_t_tdt)
            start = time()
        # print(A)
    return c_tdt,tps

# =============================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 2  ======================
# =============================================================================
# =============================================================================  
 
# ============================================================================= 
# ==============================Regime transitoire=============================
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
    c_t[:-1] = [0 for i in range(n-1)]
    c_t[-1]=prm.Ce
    c=[c_t]

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
    
    
    
    b[1:n-1]=-dt*prm.S+c_t[1:n-1]
    b[0] = 0
    b[-1] = prm.Ce
    c_tdt = np.linalg.solve(A, b)
    c_t[:]=c_tdt[:]
    t+=prm.dt
    tps.append(t)
    
    while err_t_tdt>prm.err_t_tdt:

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

# ============================================================================= 
# ==============================Regime stationnaire============================
# ============================================================================= 
def PbF_S(prm):
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
    c_t[:-1] = [0 for i in range(n-1)]
    c_t[-1]=prm.Ce
    c=[c_t]

    # Remplissage du centre de la matrice A et du vecteur b  
    dr_inv=0.5/dr
    dr2_inv=1/dr**2
    
    for i in range(1, prm.n-1):
        A[i, i+1] = -D_eff*(dr_inv/r[i] + dr2_inv)
        A[i, i] = D_eff*(2*dr2_inv)
        A[i, i-1] = -D_eff*(dr2_inv-dr_inv/r[i])
    
    A[-1, -1] = 1
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    
    b[1:n-1]=-prm.S
    b[0] = 0
    b[-1] = prm.Ce
    c_tdt = np.linalg.solve(A, b)
    c_t[:]=c_tdt[:]
    t+=prm.dt
    tps.append(t)
    
    
    while err_t_tdt>prm.err_t_tdt:

        b[1:n-1]=-prm.S
        b[0] = 0
        b[-1] = prm.Ce
        
        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        err_t_tdt=np.linalg.norm((c_tdt-c_t)/c_t)
        c_t[:]=c_tdt[:]
        t+=prm.dt
        tps.append(t)

    return c_tdt,tps
