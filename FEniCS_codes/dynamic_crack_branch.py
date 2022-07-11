"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

'''Code to solve dynamic crack branch problem using explicit method'''

# Importing modules
from fenics import *
import numpy as np 
from matplotlib import pyplot as plt
from csv import DictWriter
import time as Time
from ufl import max_value, eq
import os

tic = Time.time()

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
set_log_active(False)

# Creating mesh and defining functionspace
mesh = RectangleMesh(Point(0,0),Point(0.1,0.04),400,160,'crossed')
plot(mesh)
plt.show()

coordinates = mesh.coordinates()
print(f'Total number of vertices {coordinates.shape[0]}')

x_coordinates = coordinates[:,0]
y_coordinates = coordinates[:,1]

upper_crack_tip = y_coordinates >= 0.02

dim = mesh.geometry().dim()
hmin = mesh.hmin()
print('Minimum cell size is ',hmin)
print('Number of cells are ',mesh.num_cells())

V = FunctionSpace(mesh,"CG",1)                  # For phase field
W = VectorFunctionSpace(mesh,"CG",1)            # For displacement
WW = FunctionSpace(mesh,"DG",0)                 # For projecting Energy
TT = TensorFunctionSpace(mesh,'DG',0,(2,2))     # For projecting stress and strains 

print(f'Number of degrees of freedom for phase field {V.dim()}')
print(f'Number of degrees of freedom for displacement {W.dim()}')

# Trial and test functions
p , q = TrialFunction(V) , TestFunction(V)
u , v = TrialFunction(W) , TestFunction(W)

# Material properties
E = Constant(32e9)                                                               # Young's modulus
nu = Constant(0.2)                                                               # Poisson's ratio
G_c = Constant(3)                                                                # Fracture toughness
l_0 = Constant(0.5e-3)                                                           # Length scale parameter
lmbda , mu = Constant(E*nu/((1.0 + nu )*(1.0-2.0*nu))) , Constant(E/(2*(1+nu)))  # Lame's constants
rho = Constant(2450)                                                             # Density

# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0,1e-8)')
top = CompiledSubDomain('on_boundary and near(x[1],0.04,1e-8)')

crack = CompiledSubDomain('near(x[1],0.02,l_0/2) and x[0] <= 0.05',l_0 = l_0)
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

# Defining load - Traction bottom
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

load_top = Traction_top(0,degree=1)                                              # Initiating the load
load_bottom = Traction_botttom(0,degree=1)


# Definning fields
def epsilon(u):
    '''Function to define strain = 0.5(grad(u)+grad(u).Transpose)'''
    return sym(grad(u))

def eigenvalues(strain):
    '''Eigenvalues of strain tensor'''
    I1 = tr(strain)
    I2 = det(strain)
    cond1 = conditional(gt((I1**2 - 4*I2),Constant(-1e-5)),Constant(0),Constant(np.nan))

    if eq(cond1,Constant(np.nan)):
        raise ValueError('Eigenvalues are not real')

    else:
        cond2 = conditional(gt((I1**2 - 4*I2),Constant(0)),sqrt(I1**2 - 4*I2),cond1)
        e1,e2 = (I1 + cond2)/2 , (I1 - cond2)/2 
        return e1,e2

def eig_pos(strain):
    '''Positive part of strain tensor'''
    e1,e2 = eigenvalues(strain)
    eig1 = as_tensor([[conditional(gt(e1,Constant(0)),e1,Constant(0)),0],[0,conditional(gt(e2,Constant(0)),e2,Constant(0))]])
    return eig1

def eig_neg(strain):
    '''Negative part of strain tensor'''
    e1,e2 = eigenvalues(strain)
    eig2 = as_tensor([[conditional(lt(e1,Constant(0)),e1,Constant(0)),0],[0,conditional(lt(e2,Constant(0)),e2,Constant(0))]])
    return eig2

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
    '''Total strain energy'''
    strain = epsilon(u)
    return (((1-Constant(1e-15))*(1-p)**2 + Constant(1e-15))*Psi_pos(strain) + Psi_neg(strain))*dx

# Mass form
def Mass(u,v):
    '''Mass matrix'''
    return rho*inner(u,v)*dx

# Elastic form
def K(u,p,v):
    '''Stiffness matrix'''
    return inner(sigma(u,p),nabla_grad(v))*dx

def work(t1,t2,v):
    '''Force vector'''
    return (dot(t1,v)*ds(1) + dot(t2,v)*ds(2))   

def crack_energy(p):
    '''Surface energy of crack'''
    return ((G_c/2)*((p**2/l_0) + l_0*dot(grad(p),grad(p))))*dx

def H_project(u_old,WW,Hold):
    '''Update the History field'''
    WW_trial = TrialFunction(WW)
    WW_test = TestFunction(WW)

    psi_max = max_value(Psi_pos(epsilon(u_old)),Hold)
    a_proj = inner(WW_trial,WW_test)*dx
    b_proj = inner(psi_max,WW_test)*dx

    solver = LocalSolver(a_proj,b_proj)
    solver.factorize()
    solver.solve_global_rhs(Hold)


u_new = Function(W, name = 'Displacement u_(n+1)')     # u_(n+1)
u_old = Function(W, name = 'Displacement u_(n)')       # u_n
u_old_ = Function(W, name = 'Displacement u_(n-1)')    # u_(n-1)
pold  = Function(V, name = 'Phase_field')  
Hold = Function(WW, name = 'History_field')

dt = Constant(1e-8)                               # Time step for explicit method
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
form_phi = (G_c*l_0*inner(grad(p),grad(q)))*dx + (((G_c/l_0) + 2*(1-Constant(1e-15))*max_value(Psi_pos(epsilon(u_old)),Hold))*p*q)*dx - (2*(1-Constant(1e-15))*max_value(Psi_pos(epsilon(u_old)),Hold)*q)*dx

a, L = lhs(form_u), rhs(form_u)
k,f= assemble(a),PETScVector()                                    # Assembling LHS and applying boundary condition for equation of motion

solver = LUSolver(k,'mumps')
solver.parameters['symmetric'] = True

phi_problem = LinearVariationalProblem(lhs(form_phi),rhs(form_phi),pold,bcs=bc_phi)       # Phase field equation
phi_solver = LinearVariationalSolver(phi_problem)

time = np.arange(0,T+float(dt),float(dt))                         # time vector with time step dt
size = time.size
energies = np.zeros((size,2))

folder_name = 'Dynamic_crack_branch_results'
os.mkdir(folder_name)

file = open(os.path.join(folder_name,'Results.csv'),'w')  
csv_wr = DictWriter(file,fieldnames=['Time','Strain_Energy','Surface_Energy','crack_tip_x','crack_tip_y'])
csv_wr.writeheader()

xdmf_file = XDMFFile(os.path.join(folder_name,'Results.xdmf'))
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False

for (i,t) in enumerate(time):
    print('time',t)
    load_top.t  = t                                 # updating load
    load_bottom.t = t

    phi_solver.solve()                              # Solving Phase field equation

    assemble(L,tensor=f)                            # assemble the rhs of equation of motion
    solver.solve(k,u_new.vector(),f)                                          # solving Equation of motion

    strain_Energy = assemble(Psi_total(u_old,pold))                           # strain energy
    surface_energy = assemble(crack_energy(pold))                             # Crack surface energy

    energies[i,:] = np.array([strain_Energy,surface_energy])

    if np.any(np.isnan(energies)):
        raise ValueError('Simulation stopped. nan is encountered in output')

    # Track the position of crack tip for the upper branch
    filter = np.where((pold.compute_vertex_values() > 0.75) & (upper_crack_tip), x_coordinates, 0)
    index = np.argmax(filter)

    xdmf_file.write(pold,t)
    csv_wr.writerow({'Time':t*1e6,'Strain_Energy':strain_Energy,'Surface_Energy':surface_energy,'crack_tip_x':coordinates[index,0],'crack_tip_y':coordinates[index,1]})

    H_project(u_old,WW,Hold)                                                    # Updating history field
    Update(u_new,u_old,u_old_)                                                  # Updating displacement fields

file.close()

# Plot energies 

plt.figure()
p1 = plot(pold)
plt.colorbar(p1)
plt.savefig(os.path.join(folder_name,'Phase_field'))

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, energies[:,0])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel("EE")
plt.title(r"Strain Energy [$J$]")
plt.savefig(os.path.join(folder_name,'Strain Energy'))

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, energies[:,1])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel("SE")
plt.title(r"Crack Energy [$J$]")
plt.savefig(os.path.join(folder_name,'Surface Energy'))

plt.figure()
p2 = plot(u_old_[0],title='$u_{x}$')
plt.colorbar(p2)
plt.savefig(os.path.join(folder_name,'x displacement'))

plt.figure()
p3 = plot(u_old_[1],title='$u_{y}$')
plt.colorbar(p3)
plt.savefig(os.path.join(folder_name,'y displacement'))

plt.figure()
p4 = plot(sigma(u_old_,pold)[0,0],title="$\sigma_{xx}$ [Pa]")
plt.colorbar(p4)
plt.savefig(os.path.join(folder_name,'Sigma_xx'))

plt.figure()
p5 = plot(sigma(u_old_,pold)[0,1],title="$\sigma_{xy}$ [Pa]")
plt.colorbar(p5)
plt.savefig(os.path.join(folder_name,'Sigma_xy'))

plt.figure()
p6 = plot(sigma(u_old_,pold)[1,1],title="$\sigma_{yy}$ [Pa]")
plt.colorbar(p6)
plt.savefig(os.path.join(folder_name,'Sigma_yy'))

stress = Function(TT, name = 'Stress_Fields')
stress.assign(project(sigma(u_old_,pold),TT))

xdmf_file.write(u_old_,time[-1])
xdmf_file.write(stress,time[-1])

toc = Time.time()

print('Total simulation time in minutes ',(toc-tic)/60)
plt.show()

