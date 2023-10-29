import numpy as np
import pandas as pd
from pathlib import Path  

def dp_dt (d, f, gamma, a, p, t):
  
  prod_da = np.matmul(d,a)
  
  dp      = p * (( prod_da ) / ( 1 + f * p) - gamma )

  return dp

def da_dt (K, epsilon, d_t, f, a, p, t):
  
  prod_dp = np.matmul(d_t,p/(1+f*p))
  
  da      = a * (prod_dp + epsilon *( K - a ))

  return  da 

# Control parameters

plant_threshold       = 1/100000
pollinator_threshold  = 1/100000
rep                   = 25


# Time parameters
tsim = 500.0       # Time of simulation
h    = 0.002       # Size of the intervalue to evaluate
ite  = int(tsim/h) # Number of iterations

for caty in range(1,10,1):
  #  Parameters
  N       = 500        # Number of plants
  v       = caty/10       # Ratio between plants and pollinators
  M       = int(N/v)   # Number of pollinators


  mu_d    = 1.0
  sigma_d2= 10.0

  f       = 1.0     
  gamma   = 1.0      # Plant death rate
  epsilon = 1.0      # Regulation rate pollinator
  K       = 1.0      # Carrying capacity

  p_s     = np.zeros((N,rep))
  a_s     = np.zeros((M,rep))
  phi_s   = np.zeros(rep)
  psi_s   = np.zeros(rep)

  for z in range(rep):
    t     = 0        # Initial time
    
    #  Initialization of vectors
    p       = 10*np.random.rand(N)
    a       = 10*np.random.rand(M)

    d       = np.random.normal(mu_d/N, np.sqrt(sigma_d2/N), size=(N,M))
    d_t     = np.transpose(d)

    for i in range(ite):
      t   += h

      k1p  = h*dp_dt (d, f, gamma, a, p              , t      )
      k2p  = h*dp_dt (d, f, gamma, a, p + (1/2) * k1p, t + h/2)
      k3p  = h*dp_dt (d, f, gamma, a, p + (1/2) * k2p, t + h/2)
      k4p  = h*dp_dt (d, f, gamma, a, p +         k3p, t + h  )
                  
      k1a  = h*da_dt (K, epsilon, d_t, f, a              , p, t      )
      k2a  = h*da_dt (K, epsilon, d_t, f, a + (1/2) * k1a, p, t + h/2)
      k3a  = h*da_dt (K, epsilon, d_t, f, a + (1/2) * k2a, p, t + h/2)
      k4a  = h*da_dt (K, epsilon, d_t, f, a +         k3a, p, t + h  )

      p   += (1/6)*( k1p + 2 * k2p + 2 * k3p + k4p )
      a   += (1/6)*( k1a + 2 * k2a + 2 * k3a + k4a )

    posArrCount1=0
    for i in range(len(p)):
      if (p[i] >= plant_threshold):
        posArrCount1 += 1


    posArrCount2=0
    for m in range(len(a)):
      if (a[m] >= pollinator_threshold):
          posArrCount2 += 1

    phi_s[z] = posArrCount1/N
    psi_s[z] = posArrCount2/M
    p_s[:,z] = p
    a_s[:,z] = a

  filepath = Path('Grafica_4/Pollinator phi_s gamma 1; F 1; epsilon 1; K '+str(K)+'; m_d '+str(mu_d)+'; s_d^2 '+str(sigma_d2)+';N '+str(N)+';v '+str(v)+'.csv')  
  filepath.parent.mkdir(parents=True, exist_ok=True)  
  saves_phi = pd.DataFrame(phi_s)
  saves_phi.to_csv(filepath, index=False)

  filepath = Path('Grafica_4/Pollinator psi_s gamma 1; F 1; epsilon 1; K '+str(K)+'; m_d '+str(mu_d)+'; s_d^2 '+str(sigma_d2)+';N '+str(N)+';v '+str(v)+'.csv')  
  filepath.parent.mkdir(parents=True, exist_ok=True)  
  saves_phi = pd.DataFrame(psi_s)
  saves_phi.to_csv(filepath, index=False)

  filepath = Path('Grafica_4/Pollinator p_s gamma 1; F 1; epsilon 1; K '+str(K)+'; m_d '+str(mu_d)+'; s_d^2 '+str(sigma_d2)+';N '+str(N)+';v '+str(v)+'.csv')  
  filepath.parent.mkdir(parents=True, exist_ok=True)  
  saves_p = pd.DataFrame(p_s)
  saves_p.to_csv(filepath, index=False)

  filepath = Path('Grafica_4/Pollinator a_s gamma 1; F 1; epsilon 1; K '+str(K)+'; m_d '+str(mu_d)+'; s_d^2 '+str(sigma_d2)+';N '+str(N)+';v '+str(v)+'.csv')  
  filepath.parent.mkdir(parents=True, exist_ok=True)  
  saves_a = pd.DataFrame(a_s)
  saves_a.to_csv(filepath, index=False)