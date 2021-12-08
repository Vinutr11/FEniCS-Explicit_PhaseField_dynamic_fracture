"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

# FEniCS Code to solve dynamic crack branching problem using phase field Explicit method 
# Importing modules
from fenics import *
import numpy as np 
from matplotlib import pyplot as plt
from csv import DictWriter
import time as Time

tic = Time.time()

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

# Creating mesh and defining functionspace
mesh = RectangleMesh(Point(0,0),Point(0.1,0.04),400,160,'crossed')
plot(mesh)
plt.show()

dim = mesh.geometry().dim()
hmin = mesh.hmin()
print('Minimum cell size is ',hmin)
print('Number of cells are ',mesh.num_cells())

V = FunctionSpace(mesh,"CG",1)                  # For phase field
W = VectorFunctionSpace(mesh,"CG",1)            # For displacement
WW = FunctionSpace(mesh,"DG",0)                 # For projecting Energy
TT = TensorFunctionSpace(mesh,'DG',0,(2,2))     # For projecting stress and strains 

# Trial and test functions
p , q = TrialFunction(V) , TestFunction(V)
u , v = TrialFunction(W) , TestFunction(W)

# Material properties
E = Constant(32e9)                                                               # Young's modulus
nu = 0.2                                                                         # Poisson's ratio
G_c = Constant(3)                                                                # Fracture toughness
l_0 = Constant(0.5e-3)                                                          # Length scale parameter
lmbda , mu = Constant(E*nu/((1.0 + nu )*(1.0-2.0*nu))) , Constant(E/(2*(1+nu)))  # Lame's constants
rho = Constant(2450)                                                             # Density

# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0,1e-8)')
top = CompiledSubDomain('on_boundary and near(x[1],0.04,1e-8)')

crack = CompiledSubDomain('near(x[1],0.02,l_0) and x[0] <= 0.05',l_0 = l_0)
bc_phi = DirichletBC(V,Constant(1),crack)                                        # Initial Crack length


boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
top.mark(boundaries,1)
bottom.mark(boundaries,2)
ds = Measure('ds',subdomain_data=boundaries)

# Defining load - Traction top
class Traction_top(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        values[0] = 0
        if self.t <= 0.0:
            values[1] = 0
        else:
            values[1] = 1e6
    def value_shape(self):
        return (2,)

class Traction_botttom(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        values[0] = 0
        if self.t <= 0.0:
            values[1] = 0
        else:
            values[1] = -1e6
    def value_shape(self):
        return (2,)

load_top = Traction_top(0,degree=1)         # Initiating the load
load_bottom = Traction_botttom(0,degree=1)


# Definning fields
def epsilon(u):
    '''Function to define strain = 0.5(grad(u)+grad(u).Transpose)'''
    return sym(grad(u))

def eig_pos(strain):
    '''Positive part of strain tensor'''
    I1 = tr(strain)
    I2 = det(strain)
    e1,e2 = (I1 + sqrt(I1**2 - 4*I2))/2, (I1 - sqrt(I1**2 - 4*I2))/2
    eig1 = as_tensor([[conditional(gt(e1,Constant(0)),e1,Constant(0)),0],[0,conditional(gt(e2,Constant(0)),e2,Constant(0))]])
    return eig1

def eig_neg(strain):
    '''Negative part of strain tensor'''
    I1 = tr(strain)
    I2 = det(strain)
    e1,e2 = (I1 + sqrt(I1**2 - 4*I2))/2 , (I1 - sqrt(I1**2 - 4*I2))/2
    eig1 = as_tensor([[conditional(lt(e1,Constant(0)),e1,Constant(0)),0],[0,conditional(lt(e2,Constant(0)),e2,Constant(0))]])
    return eig1

def Psi_pos(strain):
    '''Positive strain energy density'''
    return (lmbda/2)*((tr(strain)+abs(tr(strain)))/2)**2+ mu*tr(eig_pos(strain)*eig_pos(strain))

def Psi_neg(strain):
    '''Negative strain energy density'''
    return (lmbda/2)*((tr(strain)-abs(tr(strain)))/2)**2+ mu*tr(eig_neg(strain)*eig_neg(strain))

def sigma(u,p):
    '''Constitutive relation for stress and strain for linear elastic model under plane strain condition '''
    strain = variable(epsilon(u))
    return ((1-Constant(1e-15))*(1-p)**2 + Constant(1e-15))*diff(Psi_pos(strain),strain) + diff(Psi_neg(strain),strain)

def Psi_total(u,p):
    '''Total strain energy density'''
    strain = epsilon(u)
    return (((1-Constant(1e-15))*(1-p)**2 + Constant(1e-15))*((lmbda/2)*((tr(strain)+abs(tr(strain)))/2)**2 + mu*tr((eig_pos(strain))*(eig_pos(strain)))) + (lmbda/2)*((tr(strain)-abs(tr(strain)))/2)**2 + mu*tr((eig_neg(strain))*(eig_neg(strain))))*dx

# Mass form
def Mass(u,v):
    '''Defining the mass matrix'''
    return rho*inner(u,v)*dx

# Elastic form
def K(u,p,v):
    '''Defining the stiffness matrix'''
    return inner(sigma(u,p),nabla_grad(v))*dx

def work(t1,t2,v):
    '''Defining the force vector'''
    return (dot(t1,v)*ds(1) + dot(t2,v)*ds(2))   

def crack_energy(p):
    '''Surface energy of crack'''
    return ((G_c/2)*((p**2/l_0) + l_0*dot(grad(p),grad(p))))*dx

u_new = Function(W,name='displacement')   # u_(n+1)
u_old = Function(W)                       # u_n
u_old_ = Function(W)                      # u_(n-1)
pold  = Function(V)   

dt = 1e-8                                # Time step for explicit method
T = 80e-6                                         # Total simulation time
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

# Equation of motion
form_u = Mass(a_old(u,u_old,u_old_),v) + K(u_old,pold,v) - work(load_top,load_bottom,v)

# Phase field form equation
form_phi = (G_c*l_0*inner(grad(p),grad(q)))*dx + (((G_c/l_0) + 2*(1-Constant(1e-15))*Psi_pos(epsilon(u_old)))*p*q)*dx - (2*(1-Constant(1e-15))*Psi_pos(epsilon(u_old))*q)*dx

a, L = lhs(form_u), rhs(form_u)
k,f= assemble(a),PETScVector()

solver = LUSolver(k,'mumps')
solver.parameters['symmetric'] = True

phi_problem = LinearVariationalProblem(lhs(form_phi),rhs(form_phi),pold,bcs=bc_phi)
phi_solver = LinearVariationalSolver(phi_problem)

time = np.arange(0,T+float(dt),float(dt))          # time vector with time step dt
size = time.size
energies = np.zeros((size,2))

file = open('dynamic_crack_branch.csv','w') 
csv_wr = DictWriter(file,fieldnames=['Time','Strain_Energy','Surface_Energy'])
csv_wr.writeheader()

Vtk = File('dynamic_crack_branch/PF.pvd')

for (i,t) in enumerate(time):
    print('time',t)
    load_top.t  = t                                 
    load_bottom.t = t

    phi_solver.solve()  

    assemble(L,tensor=f)
    solver.solve(k,u_new.vector(),f)                                         # solving Equation of motion

    strain_Energy = assemble(Psi_total(u_old,pold))                           # strain energy
    surface_energy = assemble(crack_energy(pold))                             # Crack surface energy

    energies[i,:] = np.array([strain_Energy,surface_energy])

    Vtk << (pold,t)
    csv_wr.writerow({'Time':t*1e6,'Strain_Energy':strain_Energy,'Surface_Energy':surface_energy})

    Update(u_new,u_old,u_old_)                                                  # Updating displacement fields


file.close()

# Plot energies 

plt.figure()
p1 = plot(pold)
plt.colorbar(p1)
plt.savefig('Phase field')

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, energies[:,0])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel("EE")
plt.title(r"Strain Energy [$J$]")
plt.savefig("Strain Energy")

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, energies[:,1])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel("SE")
plt.title(r"Crack Energy [$J$]")
plt.savefig("Surface Energy")

plt.figure()
p2 = plot(u_old_[0],title='$u_{x}$')
plt.colorbar(p2)
plt.savefig('x_displacement')

plt.figure()
p3 = plot(u_old_[1],title='$u_{y}$')
plt.colorbar(p3)
plt.savefig('y_displacement')

plt.figure()
p4 = plot(sigma(u_old_,pold)[0,0],title="$\sigma_{xx}$ [Pa]")
plt.colorbar(p4)
plt.savefig('sigma_xx')

plt.figure()
p5 = plot(sigma(u_old_,pold)[0,1],title="$\sigma_{xy}$ [Pa]")
plt.colorbar(p5)
plt.savefig('sigma_xy')

plt.figure()
p6 = plot(sigma(u_old_,pold)[1,1],title="$\sigma_{yy}$ [Pa]")
plt.colorbar(p6)
plt.savefig('sigma_yy')

toc = Time.time()

print('Total simulation time in minutes ',(toc-tic)/60)
plt.show()


