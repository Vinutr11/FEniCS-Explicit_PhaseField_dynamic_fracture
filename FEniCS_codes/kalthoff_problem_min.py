"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

'''Code to solve kalthoff winkler problem using global minimization for non-linear phase field equation''' 
# Importing modules
from fenics import *
import numpy as np 
from matplotlib import pyplot as plt
import time as Time
from csv import DictWriter
import os
from ufl import eq

tic = Time.time()

parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, 
                                 "representation":"uflacs", "quadrature_degree": 2})
set_log_active(False)

# Creating mesh and defining functionspace
mesh_file = os.path.join(os.path.split(os.getcwd())[0],'Mesh_files/kalthoff_mesh_1.xml')
mesh = Mesh(mesh_file)
plot(mesh)
plt.show()

coordinates = mesh.coordinates()
print(f'Total number of vertices {coordinates.shape[0]}')

x_coordinates = coordinates[:,0]
initial_crack_tip = (0.049,0.025)

dim = mesh.geometry().dim()
hmin = mesh.hmin()
print('Total number of cells', mesh.num_cells())
print('Mininmun cell size', mesh.hmin())

V = FunctionSpace(mesh,"CG",1)                  # For phase field
W = VectorFunctionSpace(mesh,"CG",1)            # For displacement
TT = TensorFunctionSpace(mesh,'DG',0,(2,2))     # For projecting stress and strains 

print(f'Number of degrees of freedom for phase field {V.dim()}')
print(f'Number of degrees of freedom for displacement {W.dim()}')

# Trial and test functions
p , q = TrialFunction(V) , TestFunction(V)
u , v = TrialFunction(W) , TestFunction(W)

# Material properties
E = Constant(190e9)                                                                       # Young's modulus
nu = Constant(0.3)                                                                        # Poisson's ratio
G_c = Constant(2.213*1e4)                                                                 # Fracture toughness
l_0 = Constant(5e-4)                                                                      # Length scale parameter
lmbda , mu = Constant(E*nu/((1.0 + nu )*(1.0-2.0*nu))) , Constant(E/(2*(1+nu)))           # Lame's constants
rho = Constant(8000)                                                                      # Density
f = Constant(2812.25*1e6)                                                                 # Strength
l_ch = Constant(0.53*1e-3)
c_0 = Constant(pi)
 
# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0,1e-8)')
side = CompiledSubDomain('on_boundary and (near(x[0],0,1e-8) and x[1] < 0.025)')

class Velocity(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        if self.t <= 1e-6:
            values[0] = (16.5*self.t)/1e-6
        else:
            values[0] = 16.5

velocity = Velocity(0,degree=1)

bc_bottom = DirichletBC(W.sub(1),Constant(0),bottom)                  # Boundary conditions for displacement
bc_side = DirichletBC(W.sub(0),velocity,side)                         # Boundary conditions for velocity

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

def Omega(p):
    '''Energy degradation function'''
    a1 = (4*l_ch)/(c_0*l_0)
    a2 = Constant(-0.5)
    return ((1-p)**2)/(((1-p)**2) + a1*p*(1 + a2*p))

def sigma(u,p):
    '''Constitutive relation for stress and strain for linear elastic model under plane strain condition '''
    strain = variable(epsilon(u))
    return Omega(p)*diff(Psi_pos(strain),strain) + diff(Psi_neg(strain),strain)

def Psi_total(u,p):
    '''Total strain energy'''
    strain = epsilon(u)
    return (Omega(p)*Psi_pos(strain) + Psi_neg(strain))*dx

# Mass form
def Mass(u,v):
    '''Defining the mass matrix'''
    return rho*inner(u,v)*dx

# Elastic form
def K(u,p,v):
    '''Defining the stiffness matrix'''
    return inner(sigma(u,p),nabla_grad(v))*dx

def crack_energy(p):
    '''Surface energy of crack'''
    return ((G_c/c_0)*(((2*p-p**2)/l_0) + l_0*dot(grad(p),grad(p))))*dx

a_new = Function(W, name = 'Acceleration a_(n+1)')
a_old = Function(W, name = 'Acceleration a_(n)')
u_new = Function(W, name = 'Displacement u_(n+1)')
u_old = Function(W, name = 'Displacement u_(n)')
v_old = Function(W, name = 'Velocity v_(n)')
p_new = Function(V, name = 'Phase_field')  

l_b = Function(V)
u_b = interpolate(Constant(1.),V)

dt = Constant(1e-8)                               # Time step for explicit method
T = 90e-6                                         # Total simulation time
print(float(dt))  

def displacement_new(u_old,v_old,a_old,ufl=True):
    if ufl == True:
        dt_ = dt
    else:
        dt_ = float(dt)
    return (u_old + dt_*v_old + (dt_**2/2)*a_old)

def velocity_new(v_old,a_old,a_new,ufl=True):
    if ufl == True:
        dt_ = dt
    else:
        dt_ = float(dt)
    return (v_old + (a_old + a_new)*(dt_/2))

def Update_displacement(u_new,u_old,v_old,a_old):
    u_new.vector()[:] = displacement_new(u_old.vector(),v_old.vector(),a_old.vector(),ufl=False)

def Update(a_new,a_old,u_old,v_old,u_new):
    a_new_vec = a_new.vector()
    a_old_vec = a_old.vector()
    v_old_vec = v_old.vector()
    u_new_vec = u_new.vector()

    v_new_vec = velocity_new(v_old_vec,a_old_vec,a_new_vec,False)

    a_old.vector()[:] = a_new_vec
    u_old.vector()[:] = u_new_vec
    v_old.vector()[:] = v_new_vec

# Equation of motion
form_u = Mass(u,v) + K(u_new,p_new,v) 

a, L= lhs(form_u), rhs(form_u)
k = assemble(a)
f = PETScVector()

solver = LUSolver(k,'mumps')
solver.parameters['symmetric'] = True

Elastic_Energy = Psi_total(u_new,p_new)
Damage = crack_energy(p_new)
Total_Energy = Elastic_Energy + Damage

Phi_Jacobian = derivative(Total_Energy,p_new,q)
phi_Hessian = derivative(Phi_Jacobian,p_new,p)

class DamageProblem(OptimisationProblem):

    def f(self, x):
        """Function to be minimised"""
        p_new.vector()[:] = x
        return assemble(Total_Energy)

    def F(self, b, x):
        """Gradient (first derivative)"""
        p_new.vector()[:] = x
        assemble(Phi_Jacobian, b)

    def J(self, A, x):
        """Hessian (second derivative)"""
        p_new.vector()[:] = x
        assemble(phi_Hessian, A)

solver_phi = PETScTAOSolver()
solver_phi.parameters.update({"method": "tron","linear_solver" : "umfpack", 
                                    "line_search": "gpcg", "report": True, "gradient_absolute_tol":1e-8, "gradient_relative_tol":1e-6})

info(solver_phi.parameters,True)

time = np.arange(0,T+float(dt),float(dt))          
size = time.size
energies = np.zeros((size,2))
energies[0,:] = np.array([0,assemble(crack_energy(p_new))])

folder_name = 'Kathoff_problem_results_min'
os.mkdir(folder_name)

file = open(os.path.join(folder_name,'Results.csv'),'w')   # File to store Energy 
csv_wr = DictWriter(file,fieldnames=['Time','Strain_Energy','Surface_Energy','crack_tip_x','crack_tip_y'])
csv_wr.writeheader()

xdmf_file = XDMFFile(os.path.join(folder_name,'Results.xdmf'))
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False

for (i,dt) in enumerate(np.diff(time)):
    t = time[i+1]
    print('time',t)       

    velocity.t = time[i]                                         
    bc_side.apply(v_old.vector())                    # Apply boundary conditions to velocity vector

    Update_displacement(u_new,u_old,v_old,a_old)     # Updating the displacement
    bc_bottom.apply(u_new.vector())                  # Apply boundary conditions to displacement vector

    solver_phi.solve(DamageProblem(),p_new.vector(),l_b.vector(),u_b.vector()) 

    assemble(L,tensor=f)
    solver.solve(k,a_new.vector(),f)

    E_elas = assemble(Psi_total(u_new,p_new))                                       # Strain energy
    surface_energy = assemble(crack_energy(p_new))                                  # Crack energy

    energies[i+1,:] = np.array([E_elas,surface_energy])

    if np.any(np.isnan(energies)):
        raise ValueError('Simulation stopped. nan is encountered in output')

    xdmf_file.write(u_new,t)
    xdmf_file.write(p_new,t)

    filter = np.where(p_new.compute_vertex_values() > 0.85, x_coordinates, 0)
    index = np.argmax(filter)

    if index == 0:
       csv_wr.writerow({'Time':t,'Strain_Energy':E_elas,'Surface_Energy':surface_energy,'crack_tip_x':initial_crack_tip[0],'crack_tip_y':initial_crack_tip[1]})
    else:
       csv_wr.writerow({'Time':t,'Strain_Energy':E_elas,'Surface_Energy':surface_energy,'crack_tip_x':coordinates[index,0],'crack_tip_y':coordinates[index,1]})

    Update(a_new,a_old,u_old,v_old,u_new)                                             # Updating the fields 
    l_b.vector()[:] = p_new.vector()

file.close()

# Plot energies 

plt.figure()
p1 = plot(p_new)
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
p2 = plot(u_new[0],title='$u_{x}$')
plt.colorbar(p2)
plt.savefig(os.path.join(folder_name,'x displacement'))

plt.figure()
p3 = plot(u_new[1],title='$u_{y}$')
plt.colorbar(p3)
plt.savefig(os.path.join(folder_name,'y displacement'))

plt.figure()
p4 = plot(sigma(u_new,p_new)[0,0],title="$\sigma_{xx}$ [Pa]")
plt.colorbar(p4)
plt.savefig(os.path.join(folder_name,'Sigma_xx'))

plt.figure()
p5 = plot(sigma(u_new,p_new)[0,1],title="$\sigma_{xy}$ [Pa]")
plt.colorbar(p5)
plt.savefig(os.path.join(folder_name,'Sigma_xy'))

plt.figure()
p6 = plot(sigma(u_new,p_new)[1,1],title="$\sigma_{yy}$ [Pa]")
plt.colorbar(p6)
plt.savefig(os.path.join(folder_name,'Sigma_yy'))

toc = Time.time()

print('Total simulation time in minutes ',(toc-tic)/60)
plt.show()


