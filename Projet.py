# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pytest
from Fonctions import *
from os import environ
N_THREADS = '12'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS
from time import time
start = time()

# Assignation des paramètres pour le premier cas
class param():
    S=8*10**-9
    D=1
    R=D/2
    Ce=12
    D_eff=10**-10
    dr=0.0001
    dt=0.5*dr**2/(D_eff*10)
    n  = int(R/dr)
    err_t_tdt=10**-9
    
    
# prmPB=param()    
# plt.figure() 
# plt.grid()   
# r,C_analy=C_analytique(prmPB) 
# plt.plot(r,C_analy) 

# C_num,tps=Pb_S1(prmPB)
# plt.plot(r,C_num) 
  
# duration = time() - start
# print(f'Took {duration:.3f} seconds')

# # ===========================================================================
    


# =============================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 1  ======================
# =============================================================================
# =============================================================================  
 
# ============================================================================= 
# ==============================Regime transitoire=============================
# =============================================================================    
def PartieE():    
     
    
    # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.05,0.025,0.01,0.009,0.008,0.007,0.006,0.005]   
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre1= []
    r_ordre1= []
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbB(prm)
        
        c_ordre1.append(C_num)
        r_ordre1.append(r)
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n) # Calcul de l'erreur
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
        
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L1")
    plt.title("Convergence de l'erreur L1 en fonction de dr")
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L2")
    plt.title("Convergence de l'erreur L2 en fonction de dr")
    print(dr_testee,erreur_L2)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur Linf")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de dr")
    

# ============================================================================= 
# ==============================Regime stationnaire============================
# =============================================================================    
def PartieE_S():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.05,0.025,0.01,0.005,0.0025,0.001,0.0008,0.0007,0.0006,0.0005,0.00025,0.0001]   
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre1= []
    r_ordre1= []   
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbB_S(prm)
        
        c_ordre1.append(C_num)
        r_ordre1.append(r)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n) # Calcul de l'erreur
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
        
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L1")
    plt.title("Convergence de l'erreur L1 en fonction de dr")
    print(dr_testee,erreur_L1)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L2")
    plt.title("Convergence de l'erreur L2 en fonction de dr")
    print(dr_testee,erreur_L2)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur Linf")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de dr")
    print(dr_testee,erreur_Linf)
    
    plt.figure()
    for i in range(len(c_ordre1)):
        if dr_testee[i] in [0.1,0.01,0.001,0.0001,0.00001]:
          lab='dr='+str(dr_testee[i])
          plt.plot(r_ordre1[i],c_ordre1[i],'-.',label=lab)
    plt.legend()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("C")
    plt.title("Profil de concentration en fonction de dr") 
# =============================================================================
# =============================================================================
# =====================  Deuxieme cas : Schema d'ordre 2  ======================
# =============================================================================
# =============================================================================  
    
def PartieF():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.05,0.025,0.01]   
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= []   
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbF(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n) # Calcul de l'erreur
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
        
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L1")
    plt.title("Convergence de l'erreur L1 en fonction de dr")
    print(dr_testee,erreur_L1)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L2")
    plt.title("Convergence de l'erreur L2 en fonction de dr")
    print(dr_testee,erreur_L2)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur Linf")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de dr")
    print(dr_testee,erreur_Linf)
    
    plt.figure()
    for i in range(len(c_ordre2)):
       if dr_testee[i] in [0.1,0.01,0.001,0.0001,0.00001]:
         lab='dr='+str(dr_testee[i])
         plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.legend()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("C")
    plt.title("Profil de concentration en fonction de dr") 
    

# ============================================================================= 
# ==============================Regime stationnaire============================
# =============================================================================    
def PartieF_S():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.05,0.025,0.01,0.005,0.0025,0.001,0.0005,0.00025,0.0001]   
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= []   
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbF_S(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n) # Calcul de l'erreur
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
        
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L1")
    plt.title("Convergence de l'erreur L1 en fonction de dr")
    print(dr_testee,erreur_L1)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur L2")
    plt.title("Convergence de l'erreur L2 en fonction de dr")
    print(dr_testee,erreur_L2)
    
    plt.figure()
    # plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("Erreur Linf")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de dr")
    print(dr_testee,erreur_Linf)
    
    plt.figure()
    for i in range(len(c_ordre2)):
        if dr_testee[i] in [0.1,0.01,0.001,0.0001,0.00001]:
            
    # plt.rcParams['text.usetex'] = True
          lab='dr='+str(dr_testee[i])
          plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.legend()
    plt.grid()
    plt.xlabel("dr")
    plt.ylabel("C")
    plt.title("Profil de concentration en fonction de dr") 
        

# print("Veuillez attendre la vérification du code est en cours.")    
# pytest.main(['-q', '--tb=long', 'Tests.py'])    
