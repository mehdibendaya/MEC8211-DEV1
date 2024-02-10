# -*- coding: utf-8 -*-
''' Ce code permet de resoudre l'equation de diffusion de sel dans le beton.
Le projet ce divise 4 sous-programme:
    - PartieE() : Schema d'ordre 1 en resolvant le systeme transitoire 
    - PartieE_S() : Schema d'ordre 1 en resolvant le systeme stationnaire
    - PartieF() : Schema d'ordre 2 en resolvant le systeme transitoire 
    - PartieF_S() : Schema d'ordre 2 en resolvant le systeme stationnaire
    '''
from os import environ
from time import time
from math import log
import numpy as np
import matplotlib.pyplot as plt
from Fonctions import *


N_THREADS = '12'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

start = time()


'''Creation d'une classe qui servira pour tous les cas  La classe pourra etre modifier au besoin'''
class param():
    S=8**-9 #Terme source
    D=1        #Diametre de la colonne
    R=D/2      #Rayon de la colonne
    Ce=12      #Concentration en sel de l'eau
    D_eff=10**-10 #Coefficient de diffusion du sel dans le beton
    dr=0.0001  #Pas en espace
    dt=0.5*dr**2/(D_eff*10) # Pas en temps
    n  = int(R/dr) #Nombre de noeuds
    err_t_tdt=10**-7 #Condition d'arret        
    
  
duration = time() - start
print(f'Took {duration:.3f} seconds')


'''# ==========================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 1  ======================
# =============================================================================
# =============================================================================  
 
# ============================================================================= 
# ==============================Regime transitoire=============================
# =========================================================================='''  
def partieE():    
     
    
    '''Initialisation des paramètres (Valeurs qui peuvent être modifiées)'''
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
        
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n)
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
    

'''# ==========================================================================
# ==============================Regime stationnaire============================
# =========================================================================='''
def PartieE_S():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.01,0.001,0.005,0.004,0.003,0.002,0.001,0.0009,
                 0.0008,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.00019,
                 0.00018,0.00017,0.00016,0.00015,0.00014,0.00013,0.00012,
                 0.00011,0.0001]    
      
      
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
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))
        
       

    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("Erreur $L_{1}$")
    plt.title("Convergence de l'erreur $L_{1}$ en fonction de $\Delta$r")
    plt.savefig('Conv_Ordre1_L1', dpi=1000)
    print("Erreur L1")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L1[i])
    for i in range(1,len(dr_testee)):    
        print(log(erreur_L1[i-1]/erreur_L1[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("Erreur $L_{2}$")
    plt.title("Convergence de l'erreur $L_{2}$  en fonction de $\Delta$r")
    plt.savefig('Conv_Ordre1_L2', dpi=1000)
    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1):    
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("Erreur $L_{\infty}$")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de $\Delta$r")
    plt.savefig('Conv_Ordre1_Linf', dpi=1000)
    print("Erreur Linf")

    
    plt.figure()
    for i in range(len(c_ordre1)):
        if dr_testee[i] in [0.0003,0.0002,0.0001]:
            lab='dr='+str(dr_testee[i])
            plt.plot(r_ordre1[i],c_ordre1[i],'-.',label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre2_C_dr', dpi=1000)
    
'''# ==========================================================================
# =============================================================================
# =====================  Deuxieme cas : Schema d'ordre 2  =====================
# =============================================================================
# =========================================================================='''  
    
def PartieF():    
     
    
    '''Initialisation des paramètres (Valeurs qui peuvent être modifiées)'''
    prm = param()
    dr_testee = [0.1,0.05,0.025,0.01]   
      
      
    '''Initialisation des vecteurs'''
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= []   
    
    ''''Calcul de de l'erreur'''
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbF(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n)
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
    

'''# ==========================================================================
# ==============================Regime stationnaire============================
# =========================================================================='''  
def PartieF_S():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = [0.1,0.01,0.001,0.005,0.004,0.003,0.002,0.001,0.0009,
                 0.0008,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.00019,
                 0.00018,0.00017,0.00016,0.00015,0.00014,0.00013,0.00012,
                 0.00011,0.0001]    
      
      
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
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n)
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
        
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("Erreur $L_{1}$")
    plt.title("Convergence de l'erreur $L_{1}$ en fonction de $\Delta$r")
    plt.savefig('Conv_Ordre2_L1', dpi=1000)
    print("Erreur L1")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L1[i])
    for i in range(1,len(dr_testee)):    
        print(log(erreur_L1[i-1]/erreur_L1[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L2,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("Erreur $L_{2}$")
    plt.title("Convergence de l'erreur $L_{2}$  en fonction de $\Delta$r")
    plt.savefig('Conv_Ordre2_L2', dpi=1000)
    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1): 
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
        
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_Linf,'.')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("Erreur $L_{\infty}$")
    plt.title("Convergence de l'erreur $L_{\infty}$ en fonction de $\Delta$r")
    plt.savefig('Conv_Ordre2_Linf', dpi=1000)
    
    plt.figure()
    plt.rcParams['text.usetex'] = True
    for i in range(len(c_ordre2)):     
    
       lab='dr='+str(dr_testee[i])
       plt.plot(r_ordre2[i],c_ordre2[i],label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("$\Delta$r [$m^{-1}$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre2_C_dr', dpi=1000)
    