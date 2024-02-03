# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pytest
from Fonctions import *


# Assignation des paramètres pour le premier cas
class param():
    S=8*10**-9
    D=1
    R=D/2
    Ce=12
    D_eff=10**-10
    dr=0.0001
    dt=0.001
    n  = int(R/dr)
    err_t_tdt=0.00001
    
    
prmPB=param()    
plt.figure() 
plt.grid()   
r,C_analy=C_analytique(prmPB) 
plt.plot(r,C_analy) 

C_num,tps=PbB(prmPB)
plt.plot(r,C_num) 
  

# # =============================================================================
    


# =============================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 1  ======================
# =============================================================================
# =============================================================================  
    
def PartieE():    
     
    
    # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.05,0.025,0.01,0.005,0.0025,0.001,0.0005,0.00025,0.0001,0.00005,0.00001]   
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
       
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        
        r,C_analy=C_analytique(prm)
        C_num,tps=PbB(prm)
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n) # Calcul de l'erreur
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.xlabel("Erreur L1")
    plt.title("Convergence de l'erreur L1 en fonction de dr")
    
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.xlabel("Erreur L2")
    plt.title("Convergence de l'erreur L2 en fonction de dr")
    print(dr_testee,erreur_L2)
    
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("dr")
    plt.xlabel("Erreur Linf")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de dr")


# =============================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 2  ======================
# =============================================================================
# =============================================================================  
    
def PartieF():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
     prm = param()
     dr_testee = [0.1,0.05,0.025,0.01,0.005,0.0025,0.001,0.0005,0.00025,0.0001,0.00005,0.00001]   
   
   
     # Initialisation des vecteurs
     erreur_L1 = []
     erreur_L2 = []
     erreur_Linf = []
    
     # Calcul de de l'erreur
     for dr_act in dr_testee:
        
         prm.dr = dr_act
         
         r,C_analy=C_analytique(prm)
         C_num,tps=PbB(prm)
         print(tps)
         epsilon_h=C_num-C_analy
         
         erreur_L1.append(np.sum(abs(epsilon_h))/prm.n) # Calcul de l'erreur
         erreur_L2.append(np.linalg.norm(epsilon_h))
         erreur_Linf.append(max(abs(epsilon_h)))
         
     plt.figure()
     plt.rcParams['text.usetex'] = True
     plt.loglog(dr_testee,erreur_L1,'.')
     plt.gca().invert_xaxis()
     plt.grid()
     plt.xlabel("dr")
     plt.xlabel("Erreur L1")
     plt.title("Convergence de l'erreur L1 en fonction de dr")
     
     plt.figure()
     plt.rcParams['text.usetex'] = True
     plt.loglog(dr_testee,erreur_L2,'.')
     plt.gca().invert_xaxis()
     plt.grid()
     plt.xlabel("dr")
     plt.xlabel("Erreur L2")
     plt.title("Convergence de l'erreur L2 en fonction de dr")
     
     plt.figure()
     plt.rcParams['text.usetex'] = True
     plt.loglog(dr_testee,erreur_Linf,'.')
     plt.gca().invert_xaxis()
     plt.grid()
     plt.xlabel("dr")
     plt.xlabel("Erreur Linf")
     plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de dr") 


# print("Veuillez attendre la vérification du code est en cours.")    
# pytest.main(['-q', '--tb=long', 'Tests.py'])    
