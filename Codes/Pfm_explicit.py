"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

# FEniCS Code to solve dynamic phase field fracture problem with Explicit central difference scheme
# Importing modules
from fenics import *
from ufl import max_value
import numpy as np 
from matplotlib import pyplot as plt
from csv import DictWriter
import time as Time

tic = Time.time()

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

# Creating mesh and defining functionspace
mesh_size = 100
mesh = UnitSquareMesh(mesh_size,mesh_size)
plot(mesh)
plt.show()

V = FunctionSpace(mesh,"CG",1)                  # For phase field
W = VectorFunctionSpace(mesh,"CG",1)            # For displacement
WW = FunctionSpace(mesh,"DG",0)                 # For projecting Energy
TT = TensorFunctionSpace(mesh,'DG',0,(2,2))     # For projecting stress and strains 

dim = mesh.geometry().dim()
hmin = mesh.hmin()
print('Minimum cell size is ',hmin)

# Trial and test functions
p , q = TrialFunction(V) , TestFunction(V)
u , v = TrialFunction(W) , TestFunction(W)

# Material properties
E = 1000                                                                         # Young's modulus
nu = 0.3                                                                         # Poisson's ratio
G_c = 0.025                                                                      # Fracture toughness
l_0 = 0.1                                                                        # Length scale parameter
lmbda , mu = Constant(E*nu/((1.0 + nu )*(1.0-2.0*nu))) , Constant(E/(2*(1+nu)))  # Lame's constants
rho = Constant(1.0)                                                              # Density

# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0,DOLFIN_EPS)')
top = CompiledSubDomain('on_boundary and near(x[1],1,DOLFIN_EPS)')
left = CompiledSubDomain('on_boundary and near(x[0],0,DOLFIN_EPS)')
right = CompiledSubDomain('on_boundary and near(x[0],1,DOLFIN_EPS)')

# Defining Dirichlet B.C
bc_bottom = DirichletBC(W,Constant((0,0)),bottom)
bc_left = DirichletBC(W.sub(0),Constant(0),left)
bc_right = DirichletBC(W.sub(0),Constant(0),right)

bcs_u = [bc_bottom,bc_left,bc_right]

boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure('ds',subdomain_data=boundaries)

# Defining load - Traction (Ramp function)
class Traction(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        values[0] = 0
        if self.t <= 0.02:
            values[1] = 250*self.t
        else:
            values[1] = 5
    def value_shape(self):
        return (2,)

load = Traction(0,degree=1)         # Initiating the load

# Definning fields
def epsilon(u):
    '''Function to define strain = 0.5(grad(u)+grad(u).Transpose)'''
    return sym(grad(u))

def sigma(u,pold):
    '''Constitutive relation for stress and strain for linear elastic model under plane strain condition '''
    return (((1-pold)**2)*(2*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(dim)))


def psi(u):
    '''Positive part of strain energy density based on volumetric and deviotoric spilt of strain. Responsible for driving the crack'''
    return 0.5*(lmbda + (2*mu)/3)*(0.5*(tr(epsilon(u)) + abs(tr(epsilon(u)))))**2 + mu*inner(dev(epsilon(u)),dev(epsilon(u)))

# Mass form
def Mass(u,v):
    '''Defining the mass matrix. Later will be used to compute Kinetic energy'''
    return rho*inner(u,v)*dx

# Elastic form
def K(u,pold,v):
    '''Defining the stiffness matrix. Later will be used to compute Strain energy'''
    return inner(sigma(u,pold),nabla_grad(v))*dx

def work(t,v):
    '''Defining the force vector. Later will be used to compute Work done'''
    return dot(t,v)*ds(1)      

# Strain energy
def Psi_total(u):
    '''Strain energy density'''
    return ((lmbda/2)*(tr(epsilon(u))**2) + mu*tr(epsilon(u)*epsilon(u)))*dx

def crack_energy(pold):
    '''Surface energy of crack'''
    return ((G_c/2)*((pold**2/l_0) + l_0*dot(grad(pold),grad(pold))))*dx
  

u_new = Function(W, name = 'Displacement u_(n+1)')   # u_(n+1)
u_old = Function(W, name = 'Displacement u_(n)')                       # u_n
u_old_ = Function(W, name = 'Displacement u_(n-1)')                      # u_(n-1)
pold ,Hold = Function(V, name = 'Phase_field') , Function(WW)   # p_n, History energy

c = Constant(sqrt(E/rho))                       # Stress waave speed
dt = 0.05*hmin/c                                # Time step for explicit method
T = 0.1                                         # Total simulation time
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

    u_old_.vector()[:] = u_old_vec
    u_old.vector()[:] = u_new_vec

# Form equation [M]{a_n} + [K][u_n] = F_n
form_u = Mass(a_old(u,u_old,u_old_),v) + K(u_old,pold,v) - work(load,v)  

# Form equation for phase field evolution
form_phi = (G_c*l_0*inner(grad(p),grad(q)))*dx + (((G_c/l_0) + 2*psi(u_old))*p*q)*dx - (2*psi(u_old)*q)*dx

# Form equation to include History field in phase field to take care of irreversibility condotion.
# form_phi = (G_c*l_0*inner(grad(p),grad(q)))*dx + (((G_c/l_0) + 2*max_value(psi(u_old),Hold))*p*q)*dx - (2*max_value(psi(u_old),Hold)*q)*dx

a, L = lhs(form_u), rhs(form_u)
k,f= assemble(a),PETScVector()
[bc.apply(k) for bc in bcs_u]

solver = LUSolver(k,'mumps')
solver.parameters['symmetric'] = True

phi_problem = LinearVariationalProblem(lhs(form_phi),rhs(form_phi),pold,bcs=None)
phi_solver = LinearVariationalSolver(phi_problem)

time = np.arange(0,T,float(dt))          # time vector with time step dt
size = time.size
energies = np.zeros((size,5))
work_done = 0

file = open('Pfm_Explicit.csv','w')  # File to store Energy and displacement
csv_wr = DictWriter(file,fieldnames=['Time','Kinetic_Energy','Strain_Energy','Surface_Energy','Work','Total_Energy'])
csv_wr.writeheader()

xdmf_file = XDMFFile('Pfm/results.xdmf')
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False

for (i,t) in enumerate(time):
    print('time',t)
    load.t = t                                                       # updating load

    phi_solver.solve()                                               # Solving Phase field equation

    assemble(L,tensor=f)                                             # aasemble the rhs of equation of motion
    [bc.apply(f) for bc in bcs_u]                                    # Apply boundary conditions
    solver.solve(k,u_new.vector(),f)                                 # Solving equation of motion 

    E_kin = assemble(0.5*Mass(v_old(u_new,u_old_),v_old(u_new,u_old_))) # Kinetic energy
    E_elas = assemble(Psi_total(u_old))             # strain energy
    work_done += assemble(work(load,u_old-u_old_))  # Work by external force
    surface_energy = assemble(crack_energy(pold))   # Crack surface energy
    E_tot = E_kin + E_elas + surface_energy         # Total energy

    energies[i,:] = np.array([E_kin,E_elas,surface_energy,work_done,E_tot])

    xdmf_file.write(u_old,t)
    xdmf_file.write(pold,t)

    csv_wr.writerow({'Time':t,'Kinetic_Energy':E_kin,'Strain_Energy':E_elas,'Surface_Energy':surface_energy,'Work':work_done,'Total_Energy':E_tot})

    # Hold.assign(project(max_value(psi(u_old),Hold),WW))  # Updating history field
    Update(u_new,u_old,u_old_)                           # Updating displacement fields

file.close()

# Plotting Energy 
plt.figure()
plt.tight_layout()
plt.plot(time, energies[:,0])
plt.xlabel("Time (s)")
plt.ylabel("KE (J)")
plt.title('Kinetic Energy')


plt.figure()
plt.tight_layout()
plt.plot(time, energies[:,1])
plt.xlabel("Time (s)")
plt.ylabel("EE (J)")
plt.title("Elastic Energy")


plt.figure()
plt.tight_layout()
plt.plot(time, energies[:,2])
plt.xlabel("Time (s)")
plt.ylabel("SE (J)")
plt.title("Surface Energy of Crack")

plt.figure()
plt.tight_layout()
plt.plot(time, energies[:,3])
plt.xlabel("Time (s)")
plt.ylabel("Work done (J)")
plt.title("Work")


plt.figure()
plt.tight_layout()
plt.plot(time, energies)
plt.legend(( "kinetic", "elastic", "Crack","work_done","total"))
plt.xlabel("Time")
plt.ylabel("Energies")
plt.title("Energies-Explicit CD")

plt.figure()
p1 = plot(pold,title='Phase field')
plt.colorbar(p1)

stress= project(sigma(u_old,pold),TT)

plt.figure()
p2 = plot(stress[1,1],title="$\sigma_{yy}$ [Pa]")
plt.colorbar(p2)

plt.figure()
p2 = plot(u_new[1],title="$u_{y}$")
plt.colorbar(p2)

ux, uy = u_new.split(deepcopy=True)

y_lim = np.linspace(0,1)
disp_y = [uy(0.5,i) for i in y_lim]
phi_y = [pold(0.5,i) for i in y_lim]

file1 = open('Pfm_comparision_explicit.csv','w')                  # File to store Energy and displacement
csv_wr = DictWriter(file1,fieldnames=['x','u','phi'])
csv_wr.writeheader()

for j in range(0,len(y_lim)):
    csv_wr.writerow({'x':y_lim[j],'u':disp_y[j],'phi':phi_y[j]})
file.close()

plt.figure()
plt.tight_layout()
plt.plot(y_lim, disp_y)
plt.xlabel("y")
plt.ylabel("Displacement $u_{y}$")

plt.figure()
plt.tight_layout()
plt.plot(y_lim, phi_y)
plt.xlabel("y")
plt.ylabel("Phase_field")

toc = Time.time()

print('Total simulation time in minutes ',(toc-tic)/60)
plt.show()








