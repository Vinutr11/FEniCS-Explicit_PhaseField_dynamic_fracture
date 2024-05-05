"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

'''Code to solve kalthoff winkler problem using hertz law with a non-linear point contact''' 
# Importing modules
from fenics import *
import numpy as np 
from matplotlib import pyplot as plt
import time as Time
from csv import DictWriter
from ufl import max_value, eq
import os

tic = Time.time()

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
set_log_active(False)

# Creating mesh and defining functionspace
mesh_file = os.path.join(os.path.split(os.getcwd())[0],'Mesh_files/kalthoff_mesh_1.xml')
mesh = Mesh(mesh_file)
plot(mesh)
plt.show()

dim = mesh.geometry().dim()
hmin = mesh.hmin()
print('Total number of cells', mesh.num_cells())
print('Mininmun cell size', mesh.hmin())

V_dom = VectorElement('CG', mesh.ufl_cell(), 1)       # Elastic domain
R_imp = FiniteElement('R', mesh.ufl_cell(), 0)        # Rigid impactor
Mix = MixedElement([V_dom,R_imp])

V = FunctionSpace(mesh,"CG",1)                  # For phase field
W = FunctionSpace(mesh, Mix)                    # For displacement
TT = TensorFunctionSpace(mesh,'DG',0,(2,2))     # For projecting stress and strains 
WW = FunctionSpace(mesh,"DG",0)                 # For projecting Energy

print(f'Number of degrees of freedom for phase field {V.dim()}')
print(f'Number of degrees of freedom for displacement {W.dim()}')

# Trial and test functions
p , q = TrialFunction(V) , TestFunction(V)
u_dom, u_imp = TrialFunctions(W)
v_dom, v_imp = TestFunctions(W)

# Material properties
E = Constant(190e9)                                                                       # Young's modulus
nu = Constant(0.3)                                                                        # Poisson's ratio
G_c = Constant(2.2e4)                                                                     # Fracture toughness
l_0 = Constant(3.9e-4)                                                                    # Length scale parameter
lmbda , mu = Constant(E*nu/((1.0 + nu )*(1.0-2.0*nu))) , Constant(E/(2*(1+nu)))           # Lame's constants
rho_domain = Constant(8000) 
rho_impact = Constant(1570)
k_0 = Constant(1e15)

# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0,1e-8)')
side = CompiledSubDomain('on_boundary and (near(x[0],0,1e-8) and x[1] < 0.025)')

bc_bottom = DirichletBC(W.sub(0).sub(1),Constant(0),bottom)                                # Boundary conditions for domain displacement

boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
side.mark(boundaries,1)
ds = Measure('ds', subdomain_data=boundaries, domain=mesh)

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
def Mass(rho,u,v):
   '''Mass matrix'''
   return rho*inner(u,v)*dx

# Elastic form
def K(u,p,v):
   '''Stiffness matrix'''
   return inner(sigma(u,p),nabla_grad(v))*dx

def crack_energy(p):
   '''Surface energy of crack'''
   return ((G_c/2)*((p**2/l_0) + l_0*dot(grad(p),grad(p))))*dx

u_new = Function(W, name = 'Displacement u_(n+1)')     # u_(n+1)
u_old = Function(W, name = 'Displacement u_(n)')       # u_n
u_old_ = Function(W, name = 'Displacement u_(n-1)')    # u_(n-1)
pold  = Function(V, name = 'Phase_field') 
stress = Function(TT, name = 'Stress fields')
Hold = Function(WW, name = 'History_field')

u_new_dom, u_new_imp = u_new.split()
u_old_dom, u_old_imp = u_old.split()
u_old__dom, u_old__imp = u_old_.split()

dt = Constant(1e-8)                               # Time step for explicit method
T = 90e-6                                         # Total simulation time
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

def Stress_project(sigma,TT,stress):
    '''Update the History field'''
    WW_trial = TrialFunction(TT)
    WW_test = TestFunction(TT)

    a_proj = inner(WW_trial,WW_test)*dx
    b_proj = inner(sigma,WW_test)*dx

    solver = LocalSolver(a_proj,b_proj)
    solver.factorize()
    solver.solve_local_rhs(stress)

def H_project(u_old_dom,WW,Hold):
    '''Update the History field'''
    WW_trial = TrialFunction(WW)
    WW_test = TestFunction(WW)

    psi_max = max_value(Psi_pos(epsilon(u_old_dom)),Hold)
    a_proj = inner(WW_trial,WW_test)*dx
    b_proj = inner(psi_max,WW_test)*dx

    solver = LocalSolver(a_proj,b_proj)
    solver.factorize()
    solver.solve_global_rhs(Hold)

# Initialization
v_initial = interpolate(Constant((0,0,32)),W)      # Initial velocity of ridid impactor
u_old_.vector()[:] = -dt*v_initial.vector()
bc_bottom.apply(u_old_.vector()) 

# Local vector for force
impact_point = Point(0,0.0125)                     # Impact point where nonlinear force is applied
a_loc_vec = Function(W).vector()
global_force = Function(W).vector()

p1 = PointSource(W.sub(0).sub(0), impact_point, 1.0)
p2 = PointSource(W.sub(1), impact_point, -1.0)

p1.apply(a_loc_vec)
p2.apply(a_loc_vec)

def force_vector(u):
    '''Local nonlinear force vector'''
    u_values = u(0,0.0125)
    alpha = (u_values[2]-u_values[0])
    factor = k_0*((alpha + abs(alpha))/2)**(3/2)
    return Constant(factor)*a_loc_vec

# Equation of motion
Mass_form = Mass(rho_domain,a_old(u_dom,u_old_dom,u_old__dom),v_dom) + Mass(rho_impact,a_old(u_imp,u_old_imp,u_old__imp),v_imp)
Stiffness_form = K(u_old_dom,pold,v_dom)
form_u = Mass_form + Stiffness_form

# Phase field form equation
form_phi = (G_c*l_0*inner(grad(p),grad(q)))*dx + (((G_c/l_0) + 2*(1-Constant(1e-15))*max_value(Psi_pos(epsilon(u_old_dom)),Hold))*p*q)*dx - (2*(1-Constant(1e-15))*max_value(Psi_pos(epsilon(u_old_dom)),Hold)*q)*dx

a, L = lhs(form_u), rhs(form_u)       # Assembling LHS and applying boundary condition for equation of motion
M, f= assemble(a), PETScVector()
bc_bottom.apply(M) 

solver = LUSolver(M,'mumps')
solver.parameters['symmetric'] = True

phi_problem = LinearVariationalProblem(lhs(form_phi),rhs(form_phi),pold)       # Phase field equation
phi_solver = LinearVariationalSolver(phi_problem)

time = np.arange(0,T+float(dt),float(dt))          # time vector with time step dt
size = time.size
energies = np.zeros((size,4))
disp = np.zeros((size,2))

folder_name = 'Impact_problem2_results'
os.mkdir(folder_name)

file = open(os.path.join(folder_name,'Results.csv'),'w') 
csv_wr = DictWriter(file,fieldnames=['Time','Strain_Energy','Surface_Energy','Impactor_KE','Domain_KE','Imp_disp','dom_hit_disp'])
csv_wr.writeheader()

xdmf_file = XDMFFile(os.path.join(folder_name,'Results.xdmf'))
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False

for (i,t) in enumerate(time):
    print('time',t)
    phi_solver.solve()                          # Solving Phase field equation

    assemble(L,tensor=f) 
    global_force[:] = f + force_vector(u_old)   # assemble the rhs of equation of motion
    bc_bottom.apply(global_force)

    solver.solve(M,u_new.vector(),global_force)                                              # solving Equation of motion

    strain_Energy = assemble(Psi_total(u_old_dom,pold))                                      # strain energy
    surface_energy = assemble(crack_energy(pold))                                            # Crack surface energy
    Imp_KE = assemble(0.5*Mass(rho_impact,v_old(u_new_imp,u_old__imp),v_old(u_new_imp,u_old__imp))) # Impactor Kinetic energy
    dom_KE = assemble(0.5*Mass(rho_domain,v_old(u_new_dom,u_old__dom),v_old(u_new_dom,u_old__dom))) # Elastic domain Kinetic energy

    energies[i,:] = np.array([strain_Energy,surface_energy,Imp_KE,dom_KE])
    disp[i,:] = np.array([u_old_dom(0,0.0125)[0],u_old_imp(0,0.0125)])

    if np.any(np.isnan(energies)):
        raise ValueError('Simulation stopped. nan is encountered in output')

    Stress_project(sigma(u_old_dom,pold),TT,stress)

    xdmf_file.write(pold,t)
    xdmf_file.write(u_old_dom,t)
    xdmf_file.write(stress,t)

    csv_wr.writerow({'Time':t*1e6,'Strain_Energy':strain_Energy,'Surface_Energy':surface_energy,'Domain_KE':dom_KE,'Impactor_KE':Imp_KE,'Imp_disp':u_old_imp(0,0.0125),'dom_hit_disp':u_old_dom(0,0.0125)[0]})

    H_project(u_old_dom,WW,Hold)
    Update(u_new,u_old,u_old_)                                              

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
plt.tight_layout()
plt.plot(1e6*time, energies[:,2])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel("KE")
plt.title(r"Impactor Kinetic Energy [$J$]")
plt.savefig(os.path.join(folder_name, 'Impactor Kinetic Energy'))

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, energies[:,3])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel("KE")
plt.title(r"Domain Kinetic Energy [$J$]")
plt.savefig(os.path.join(folder_name,'Domain Kinetic Energy'))

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, disp[:,0])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"$u_{hit}$")
plt.title(r"$u_{x}$ of impact point")
plt.savefig(os.path.join(folder_name,'Domain u_x'))

plt.figure()
plt.tight_layout()
plt.plot(1e6*time, disp[:,1])
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"$u_{imp}$")
plt.title(r"Impactor displacement")
plt.savefig(os.path.join(folder_name,'Impactor disp'))

toc = Time.time()

print('Total simulation time in minutes ',(toc-tic)/60)

