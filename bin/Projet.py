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
from Tests_unitaires import *
import unittest


N_THREADS = '12'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS


'''Creation d'une classe qui servira pour tous les cas  La classe pourra etre modifier au besoin'''
class param():
    S=8e-9 #Terme source [mol/m3/s]
    D=1        #Diametre de la colonne [m]
    R=D/2      #Rayon de la colonne [m]
    Ce=12      #Concentration en sel de l'eau [mol/m3]
    D_eff=1e-10 #Coefficient de diffusion du sel dans le beton [m2/s]
    #Pour regime transitoire    
    dr=0.005  #Pas en espace
    dt=0.5*dr**2/(D_eff*10) # Pas en temps
    err_t_tdt=10**-7 #Condition d'arret        


'''# ==========================================================================
# =============================================================================
# =====================  Premier cas : Schema d'ordre 1  ======================
# =============================================================================
# =============================================================================  
 
# ============================================================================= 
# ==============================Regime transitoire=============================
# =========================================================================='''  
def PartieE():    
     
    
    '''Initialisation des paramètres (Valeurs qui peuvent être modifiées)'''
    prm = param()
    dr_testee = [0.05,0.025,0.01,0.005]   
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre1= []
    r_ordre1= []
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        prm.n  = int(prm.R/prm.dr)
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num=PbB(prm)
        
        c_ordre1.append(C_num)
        r_ordre1.append(r)
        
        epsilon_h=C_num-C_analy
        
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n)
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
    print("Erreur L1")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L1[i])
    for i in range(1,len(dr_testee)):    
        print(log(erreur_L1[i-1]/erreur_L1[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1):    
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
       
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.',label='$L_{1}$')  
    plt.loglog(dr_testee,erreur_L2,'.',label='$L_{2}$')
    plt.loglog(dr_testee,erreur_Linf,'.',label='$L_{\infty}$')
    plt.gca().invert_xaxis()
    plt.xlabel("$\Delta$r")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid()
    plt.title("Convergence de l'erreur  en fonction de $\Delta$r")
    
    
    plt.figure()
    for i in range(len(c_ordre1)):
        lab='dr='+str(dr_testee[i])
        plt.plot(r_ordre1[i],c_ordre1[i],'-.',label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("$\Delta$r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre1_C_dr', dpi=1000)
    

'''# ==========================================================================
# ==============================Regime stationnaire============================
# =========================================================================='''
def PartieE_S():    
     
    
     # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    n_test=[3,5,10,20,40,80,160,320,640,1280,2560,3000,4000,5000]    
      
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre1= []
    r_ordre1= []   
    dr_testee = [] 
    # Calcul de de l'erreur
    for n in n_test:
       
        # prm.dr = dr_act
        # print(prm.dr)
        C_analy,r1=C_analytique2(prm,n)
        C_num,r,dr=PbB_S(prm,n)
        
        c_ordre1.append(C_num)
        r_ordre1.append(r)
        dr_testee.append(dr)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))

    

    print("Erreur L1")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L1[i])
    for i in range(1,len(dr_testee)):    
        print(log(erreur_L1[i-1]/erreur_L1[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1):    
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    

    plt.figure(1)
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'x-',label='$L_{1}$')  
    plt.loglog(dr_testee,erreur_L2,'x-',label='$L_{2}$')
    plt.loglog(dr_testee,erreur_Linf,'x-',label='$L_{\infty}$')
    plt.gca().invert_xaxis()
    plt.xlabel("$\Delta$r")
    plt.ylabel("Erreur")
    plt.legend()
    plt.title("Convergence de l'erreur  en fonction de $\Delta$r")
    plt.savefig('Ordre1_Conv_dr', dpi=1000)

    
    plt.figure(2)
    for i in range(len(c_ordre1)):
        lab='dr='+str(dr_testee[i])
        plt.plot(r_ordre1[i],c_ordre1[i],'-.',label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("$\Delta$r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre1_C_dr', dpi=1000)
    
'''# ==========================================================================
# =============================================================================
# =====================  Deuxieme cas : Schema d'ordre 2  =====================
# =============================================================================
# =========================================================================='''  
    
def PartieF():    
     
    
    '''Initialisation des paramètres (Valeurs qui peuvent être modifiées)'''
    prm = param()
    dr_testee = [0.1,0.04,0.02,0.01,0.005]   
      
      
    '''Initialisation des vecteurs'''
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= []   
    
    ''''Calcul de de l'erreur'''
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        prm.n  = int(prm.R/prm.dr)
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbF(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n)
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
       
        
    
    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1):    
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    
    
    plt.figure(1)
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.',label='$L_{1}$')  
    plt.loglog(dr_testee,erreur_L2,'.',label='$L_{2}$')
    plt.loglog(dr_testee,erreur_Linf,'.',label='$L_{\infty}$')
    plt.gca().invert_xaxis()
    plt.xlabel("$\delta$r")
    plt.ylabel("Erreur")
    plt.legend()
    plt.title("Convergence de l'erreur  en fonction de dr")
    plt.savefig('Ordre2_Conv_dr', dpi=1000)

    
    plt.figure(2)
    for i in range(len(c_ordre1)):
        if dr_testee[i] in [0.0003,0.0002,0.0001]:
            lab='dr='+str(dr_testee[i])
            plt.plot(r_ordre1[i],c_ordre1[i],'-.',label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("$\Delta$r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre2_C_dr', dpi=1000)
    

'''# ==========================================================================
# ==============================Regime stationnaire============================
# =========================================================================='''  
def PartieF_S():    
     
    
    
    '''Initialisation des vecteurs'''
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= [] 
    # Initialisation des paramètres (Valeurs qui peuvent être modifiées)
    prm = param()
    dr_testee = []    
    n_test=[6,12,24,48,96,128]

    # Calcul de de l'erreur
    for n in n_test:
       
        
        C_analy,r1=C_analytique2(prm,n)
        C_num,r,dr=PbF_S(prm,n)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        dr_testee.append(dr)
        
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))

        
       
        
       
        
    
    print("Erreur L1")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L1[i])
    for i in range(1,len(dr_testee)):    
        print(log(erreur_L1[i-1]/erreur_L1[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    

    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1): 
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
        
    
    plt.figure(1)
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'x-',label='$L_{1}$')  
    plt.loglog(dr_testee,erreur_L2,'x-',label='$L_{2}$')
    plt.loglog(dr_testee,erreur_Linf,'x-',label='$L_{\infty}$')
    plt.gca().invert_xaxis()
    plt.xlabel("$\Delta$r")
    plt.ylabel("Erreur")
    plt.legend()
    plt.title("Convergence de l'erreur  en fonction de dr")
    plt.savefig('Ordre2_Conv_dr', dpi=1000)
    
   
    
    plt.figure(2)
    plt.rcParams['text.usetex'] = True
    for i in range(len(c_ordre2)):     
    
       lab='dr='+str(dr_testee[i])
       plt.plot(r_ordre2[i],c_ordre2[i],label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("$\Delta$r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre2_C_dr', dpi=1000)
    
# print("Veuillez attendre la vérification du code est en cours.")    
unittest.main(module=__name__)  