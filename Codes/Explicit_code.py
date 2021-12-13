"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

# FEniCS Code to solve dynamic structural problem with central difference scheme

# Importing modules
from fenics import *
import numpy as np 
from matplotlib import pyplot as plt 
from csv import DictWriter
import time as Time

tic = Time.time()

# info(parameters,True)
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

# Creating mesh and defining functionspace
mesh = UnitSquareMesh(10,10)

V = VectorFunctionSpace(mesh,'CG',1)         # For displacement
W = TensorFunctionSpace(mesh,'DG',0,(2,2))   # For stress

dim = mesh.geometry().dim()
hmin = mesh.hmin()                           # Minimum cell diameter

plot(mesh)
plt.show()

# Defining Material properties
E = 1000.0                                     # Young's modulus
nu = 0.3                                       # Poisson's ratio
lamda_ = Constant((E*nu)/((1+nu)*(1-2*nu)))    # Lame's constants
mu = Constant(E/(2*(1+nu)))
rho = Constant(1.0)                            # Density

# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0,DOLFIN_EPS)')
top = CompiledSubDomain('on_boundary and near(x[1],1,DOLFIN_EPS)')
left = CompiledSubDomain('on_boundary and near(x[0],0,DOLFIN_EPS)')
right = CompiledSubDomain('on_boundary and near(x[0],1,DOLFIN_EPS)')

# Defining Dirichlet B.C
bc_bottom = DirichletBC(V.sub(1),Constant(0),bottom)
bc_left = DirichletBC(V.sub(0),Constant(0),left)
bcs = [bc_bottom,bc_left]

boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries,1)
right.mark(boundaries,2)
ds = Measure('ds',subdomain_data=boundaries)


# Defining load - Traction (Ramp function)
class Traction(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        if self.t <= 0.2:
            values[0] = 5*self.t
        else:
            values[0] = 1
        values[1] = 0
    def value_shape(self):
        return (2,)


p = Traction(0,degree=1)         # Initializing the load

# Trial and test functions
du = TrialFunction(V)
v = TestFunction(V)

# Definning fields
def epsilon(u):
    '''Function to define strain = 0.5(grad(u)+grad(u).Transpose)'''
    return sym(grad(u))

def sigma(u):
    '''Constitutive relation for stress and strain for linear elastic model under plane strain condition '''
    return 2*mu*epsilon(u) + lamda_*tr(epsilon(u))*Identity(dim)

# Mass form
def Mass(u,v):
    '''Defining the mass matrix. Later will be used to compute Kinetic energy'''
    return rho*inner(u,v)*dx

# Elastic form
def K(u,v):
    '''Defining the stiffness matrix. Later will be used to compute Strain energy'''
    return inner(sigma(u),nabla_grad(v))*dx

def work(t,v):
    '''Defining the force vector. Later will be used to compute Work done'''
    return dot(t,v)*ds(2)

# Strain energy
def Psi(u):
    '''Strain energy density'''
    return ((lamda_/2)*(tr(epsilon(u))**2) + mu*tr(epsilon(u)*epsilon(u)))*dx

u_new = Function(V, name = 'Displacement u_(n+1)')         # U_(n+1)
u_old = Function(V, name = 'Displacement u_(n)')                             # U_n
u_old_ = Function(V, name = 'Displacement u_(n-1)')                            # U_(n-1)

c = Constant(sqrt(E/rho))                       # Stress waave speed
dt = 0.01*hmin/c                                # Time step for explicit method
T = 0.5                                         # Total simulation time
print(float(dt))       

def a_old(u_new,u_old,u_old_,ufl=True):
    '''Function to define acceleration, a_n = (U_(n+1) - 2U_n + U_(n-1))/dt^2'''
    if ufl == True:
        dt_ = dt
    else:
        dt_ = float(dt)
    return (u_new - 2*u_old + u_old_)/dt_**2

def v_old(u_new,u_old_,ufl=True):
    '''Function to define velocity, v_n = (U_(n+1) - U_(n-1))/2*dt'''
    if ufl == True:
        dt_ = dt
    else:
        dt_ = float(dt)
    return (u_new - u_old_)/(2*dt_)

def Update(u_new,u_old,u_old_):
    '''Function to update the displacement fields'''
    u_new_vec = u_new.vector()
    u_old_vec = u_old.vector()
    u_old_vec_ = u_old_.vector()

    u_old_.vector()[:] = u_old_vec
    u_old.vector()[:] = u_new_vec

# Form equations and assembling
form = Mass(a_old(du,u_old,u_old_),v) + K(u_old,v) - work(p,v) # Form equation [M]{a_n} + [K][u_n] = F_n
a,L = lhs(form),rhs(form)

k = assemble(a)                          # Assembling stiffness k = [M]/dt^2
[bc.apply(k) for bc in bcs]              # Applying Bounadary conditions

solver = LUSolver(k,'mumps')
solver.parameters['symmetric'] = True

time = np.arange(0,T+float(dt),float(dt))    # time vector with time step dt
size = time.size
u_tip = np.zeros((size,))
energies = np.zeros((size,4))
work_done = 0

file = open('Explicit_energy.csv','w')  
csv_wr = DictWriter(file,fieldnames=['Time','u_tip','Kinetic_Energy','Strain_Energy','Work','Total_Energy'])
csv_wr.writeheader()

xdmf_dis = XDMFFile('Explicit/disp.xdmf')
xdmf_dis.parameters['flush_output'] = True
xdmf_dis.parameters['rewrite_function_mesh'] = False

for (i,t) in enumerate(time):         
    p.t = t                           # Changing load with time 
    print('time',t)
    
    f = assemble(L)                   # Assemble the force vector, f = F_n - [K]{U_n} - (1/dt^2)[M]{U_(n-1)-2U_n}  
    [bc.apply(f) for bc in bcs]       # Applying bounadry conditions to force vector
    
    solver.solve(k,u_new.vector(),f)   # Solving the linear equations
    

    u_tip[i] = u_old(1.,0)[0]         # Storing Tip Displacement (x,y) = (1,0)
    E_kin = assemble(0.5*Mass(v_old(u_new,u_old_),v_old(u_new,u_old_))) # Kinetic energy
    E_elas = assemble(Psi(u_old))                                       # strain energy
    work_done += assemble(work(p,u_old-u_old_))                         # Work done by external force
    E_tot = E_kin+E_elas                                                # Total energy
    energies[i,:] = np.array([E_kin,E_elas,work_done,E_tot])

    csv_wr.writerow({'Time':t,'u_tip':u_old(1.,0)[0],'Kinetic_Energy':E_kin,'Strain_Energy':E_elas,'Work':work_done,'Total_Energy':E_tot})

    xdmf_dis.write(u_old,t) 

    Update(u_new,u_old,u_old_)       # updating the fields


file.close()

# Plot tip displacement 

plt.figure()
plt.tight_layout()
plt.plot(time[0:size], u_tip)
plt.xlabel("Time (s)")
plt.ylabel("$u_{x}(1,0)$")
plt.title('Tip displacement')

# Plot energies 

plt.figure()
plt.tight_layout()
plt.plot(time[0:size], energies[:,0])
plt.xlabel("Time (s)")
plt.ylabel("KE (J)")
plt.title('Kinetic Energy')


plt.figure()
plt.tight_layout()
plt.plot(time[0:size], energies[:,1])
plt.xlabel("Time (s)")
plt.ylabel("EE (J)")
plt.title("Elastic Energy")


plt.figure()
plt.tight_layout()
plt.plot(time[0:size], energies[:,2])
plt.xlabel("Time (s)")
plt.ylabel("Work (J)")
plt.title("Work")


plt.figure()
plt.tight_layout()
plt.plot(time[0:size], energies)
plt.legend(( "kinetic", "elastic", "work_done","total"))
plt.xlabel("Time (s)")
plt.ylabel("Energies")
plt.title("Energies")

stress = project(sigma(u_old_),W)

plt.figure()
p1 = plot(stress[0,0],title='"$\sigma_{xx}$ [Pa]"')
plt.colorbar(p1)

plt.figure()
p2 = plot(stress[0,1],title='"$\sigma_{xy}$ [Pa]"')
plt.colorbar(p2)

plt.figure()
p3 = plot(stress[1,1],title='"$\sigma_{yy}$ [Pa]"')
plt.colorbar(p3)

toc = Time.time()

print('Total simulation time in seconds ',(toc-tic))
plt.show()



