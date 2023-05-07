using LinearAlgebra, Distributions, Polynomials, Roots
using Optim, Convex, JuMP, OSQP, Ipopt
# using Gurobi, KNITRO

# GRB_ENV = Gurobi.Env()
const cutoff = 1e-3 # small cutoff in constraints to avoid numerical issues with optimization solvers

"Function to get control input for the HDVs in the simulation"
function input_for_HDV(cars::Vector{Car}, my_car::Int64, your_car::Int64, vd::Float64, W; T = 0.2, γ = 1e-3, uncertain = false)
    my_x = cars[my_car].st[1]; my_y = cars[my_car].st[2]
    my_v = cars[my_car].st[4]
    your_x = cars[your_car].st[1]; your_y = cars[your_car].st[2]
    your_v = cars[your_car].st[4]

    u = IRL_CFM(my_x, my_y, my_v, your_x, your_y, your_v, vd, W; T = T, γ = γ, uncertain = uncertain)
    return u
end

"Solve optimization problem (IRL) to find control action for HDV"
function IRL_CFM(my_x, my_y, my_v, your_x, your_y, your_v, v_d, W; T = 0.2, γ = 1e-3, uncertain = false)
    q = (0.5*T^2)^2 
    r = (my_x+T*my_v-your_x-T*your_v)/(0.5*T^2)
    ϵ = (my_y-your_y)^2 + γ
    p = (W[1]+T^2*W[2])/W[3]
    v = T*W[2]*(my_v-v_d)/p/W[3]

    coeffs = [p*q*v*r^2+ϵ*p*v-q*r, p*q*(r^2+2r*v)+ϵ*p-q, p*q*(2r+v), p*q]
    root = roots(Polynomial(coeffs)) # Find roots of polynomial
    root = root[isreal.(root)]; u = Real(root[end])
    # find_zero(Polynomial(coeffs), (-1e3, 1e3), Bisection())[end] 

    if uncertain
        return u + rand(Normal(0., 0.1))
    else
        return u
    end
end

"The class to implement the MPC_Planner"
mutable struct MPC_Planner
    T::Float64
    H::Int64
    params
    v_min; v_max; a_min; a_max;
    y_ref::Float64; v_ref::Float64
    st; u; st_Hf; st_Hb
    st_nom::Matrix{Float64}; u_nom::Matrix{Float64}

    function MPC_Planner(T::Float64, H::Int64)
        obj = new(T, H)
        return obj
    end
end

"Set the control parameters"
function set_params(c::MPC_Planner, control_params)
    c.params = control_params
end

"Set the physical limits of the traffic and vehciles"
function set_limit(c::MPC_Planner, bounds)
    c.v_min = bounds["v_min"]; c.v_max = bounds["v_max"]
    c.a_min = bounds["a_min"]; c.a_max = bounds["a_max"]
end

"Set the references"
function set_ref(c::MPC_Planner, v_ref)
    c.v_ref = v_ref
end

"Set state from all states"
function set_state(c::MPC_Planner, CAV::Car, front_HDV::Car, behind_HDV::Car)
    c.st = CAV.st
    c.u = CAV.u
    c.st_Hf = front_HDV.st
    c.st_Hb = behind_HDV.st
end

"Predict nominal trajectory over control horizon given the nominal control inputs"
function set_nominal(c::MPC_Planner, nom_inputs::AbstractMatrix)
    c.u_nom = deepcopy(nom_inputs)
    c.st_nom = zeros(4, c.H+1)
    c.st_nom[:,1] = [c.st[1], c.st[4], c.st_Hb[1], c.st_Hb[4]]
    for k in 1:c.H
        c.st_nom[:,k+1] = c.st_nom[:,k] + c.T*[c.st_nom[2,k], 0, c.st_nom[4,k], 0] 
                + c.T*[0.5*c.T*c.u_nom[1,k], c.u_nom[1,k], 0.5*c.T*c.u_nom[2,k], c.u_nom[2,k]]
    end
end

"Function to solve MPC motion planning problem"
function nonlinearMPC(c::MPC_Planner, vH; solver = "Ipopt", τ0 = 1.5, d0 = 5.0)
    vd = c.v_ref; dt = c.T; ϵ = c.params["ϵ"]
    W1 = [c.params["W_A"][1], c.params["W_H"][1]]
    W2 = [c.params["W_A"][2], c.params["W_H"][2]]
    W12 = c.params["W_AH"]

    if solver == "Ipopt"
        model = JuMP.Model(Ipopt.Optimizer)
    end
    set_silent(model)
    set_time_limit_sec(model, 0.2)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "max_iter", 100)
    # set_optimizer_attribute(model, "hsllib", "/Users/vietanhle/Documents/solver/lib/libhsl.dylib")

    @variable(model, u[1:2,1:c.H])
    set_start_value.(u, hcat(c.u_nom[:, 2:end], zeros(2,1)))
    p0 = [c.st[1], c.st_Hb[1]] 
    v0 = [c.st[4], c.st_Hb[4]] 
    v = @expression(model, v0 .+ cumsum(u*dt, dims=2))
    dp = hcat([k==1 ? dt*v0 + 0.5dt^2*u[:,k] : dt*v[:,k-1] + 0.5dt^2*u[:,k] for k in 1:c.H]...)
    p = @expression(model, p0 .+ cumsum(dp, dims=2))
    # Bound constraints
    @constraints(model, begin
        c.a_min .<= u .<= c.a_max
        c.v_min .<= v[1,:] .<= c.v_max
    end)

    # Headway constraint 
    x_front = [c.st_Hf[1] + k*dt*c.st_Hf[4] for k = 1:c.H]

    @constraints(model, begin
        d0 .<= x_front - (p[1,:] + τ0 * v[1,:])
    end)

    # Objective function
    J = sum(W1.*u.^2) + sum(W2.*(v.-[vd,vH]).^2)
    for k = 1:c.H
        xi = p[1,k]; xj = p[2,k]
        J = @NLexpression(model, J - W12*log((xi-xj)^2 + ϵ))
    end
    @NLobjective(model, Min, J)

    t_comp = @elapsed JuMP.optimize!(model)
    sol = value.(u)
    set_nominal(c, sol)
    return sol, t_comp
end

function lane_selection(c::MPC_Planner, τ0::Float64, d0::Float64)
    for k = 1:c.H+1
        xi, vi, xj, vj = c.st_nom[:,k]
        if min((xi + τ0*vi + d0 - xj), (xj + τ0*vj + d0 - xi)) >= 0.0
            return 1
        end
    end   
    return 0
end

##############################################################
##############################################################

"The class to implement the lower-level tracking MPC"
mutable struct LL_MPC
    T::Float64
    H::Int64
    lf::Float64; lr::Float64
    params
    v_min; v_max; a_min; a_max; α_min; α_max
    y_ref; x_ref; v_ref
    st; u
    st_nom::Matrix{Float64}; u_nom::Matrix{Float64}


    function LL_MPC(T::Float64, H::Int64)
        obj = new(T, H)
        return obj
    end
end

"Set the control parameters"
function set_params(c::LL_MPC, control_params, system_params)
    c.params = control_params
    c.lf = system_params["lf"]; c.lr = system_params["lr"] 
end

"Set the physical limits of the traffic and vehciles"
function set_limit(c::LL_MPC, bounds)
    c.v_min = bounds["v_min"]; c.v_max = bounds["v_max"]
    c.a_min = bounds["a_min"]; c.a_max = bounds["a_max"]
    c.α_min = bounds["α_min"]; c.α_max = bounds["α_max"]
end

"Set the references"
function set_ref(c::LL_MPC, y_ref, x_ref, v_ref)
    c.y_ref = y_ref
    c.x_ref = x_ref
    c.v_ref = v_ref
end

"Set state from all states"
function set_state(c::LL_MPC, CAV::Car)
    c.st = CAV.st
    c.u = CAV.u
end

function predict_onestep(c::LL_MPC, x::AbstractVector, u::AbstractVector)
    θ = x[3]; v = x[4] 
    a = u[1]; α = u[2]
    β = atan(c.lr/(c.lf+c.lr)*tan(α))
    dx = zeros(4)
    dx[1] = v*cos(θ+β)
    dx[2] = v*sin(θ+β)
    dx[3] = v/c.lr*sin(β)
    dx[4] = a
    return dx*c.T + x
end

"Predict nominal trajectory over control horizon given the nominal control inputs, for warm-starting"
function set_nominal(c::LL_MPC, nom_inputs::AbstractMatrix)
    c.u_nom = deepcopy(nom_inputs)
    c.st_nom = zeros(4, c.H+1)
    c.st_nom[:,1] = c.st
    for k in 1:c.H
        c.st_nom[:,k+1] = predict_onestep(c, c.st_nom[:,k], c.u_nom[:,k])
    end
end

"Function to solve MPC tracking control problem"
function trackingMPC(c::LL_MPC, U_ref, lane; solver = "Ipopt", τ0 = 1.0, d0 = 5.0, M = 1e3)
    dt = c.T; y_ref = c.y_ref; x_ref = c.x_ref; v_ref = c.v_ref
    if solver == "Ipopt"
        model = JuMP.Model(Ipopt.Optimizer)
    end
    set_silent(model)
    set_time_limit_sec(model, 0.2)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "max_iter", 100)
    # set_optimizer_attribute(model, "hsllib", "/Users/vietanhle/Documents/solver/lib/libhsl.dylib")

    # Define decision variables
    @variables model begin
        z[1:4, 1:c.H+1]
        u[1:2, 1:c.H]
    end

    # Set initial values for the optimization variables 
    warm_u = hcat(c.u_nom[:,2:end], zeros(2,1))
    set_nominal(c, warm_u)
    set_start_value.(u, c.u_nom)
    set_start_value.(z, c.st_nom)

    # dynamics constraints
    ν = c.lr/(c.lf+c.lr) 
    for t = 1:c.H
        # β = @NLexpression(model, atan(tan(u[2,t])*ν))       
        β = @expression(model, u[2,t]*ν) # Nice approximation of the above expression tan(x)≈x
        @NLconstraint(model, z[1,t+1] == z[1,t] + dt*z[4,t]*cos(z[3,t]+β))
        @NLconstraint(model, z[2,t+1] == z[2,t] + dt*z[4,t]*sin(z[3,t]+β))
        @NLconstraint(model, z[3,t+1] == z[3,t] + dt/c.lr*z[4,t]*sin(β))
        @constraint(model, z[4,t+1] == z[4,t] + dt*u[1,t])
    end
    # Initial condition
    @constraint(model, z[:,1] .== c.st)
    @constraint(model, u[1,:] .== U_ref)

    # Bound constraints
    @constraints(model, begin
        # c.a_min .<= u[1,:] .<= c.a_max
        c.α_min .<= u[2,:] .<= c.α_max
        # c.v_min .<= z[4,:] .<= c.v_max
    end)

    @constraints(model, begin
        c.params["y_min"] .<= z[2,:] .<= c.params["y_max"]
    end)

    # Ramp constraints
    @constraints(model, begin
        c.params["Δθ_min"]*dt <= z[3,1] - c.st[3] <= c.params["Δθ_max"]*dt
        c.params["Δθ_min"]*dt .<= z[3,2:end] - z[3,1:end-1] .<= c.params["Δθ_max"]*dt
        c.params["Δα_min"]*dt <= u[2,1] - c.u[2] <= c.params["Δα_max"]*dt
        c.params["Δα_min"]*dt .<= u[2,2:end] - u[2,1:end-1] .<= c.params["Δα_max"]*dt
    end)

    # Safety constraints
    if lane == 1 && c.st[2] - c.y_ref >= d0
        @constraints(model, begin
            d0 .<= z[2,:] - c.y_ref
        end)
    end

    # Objective function
    J = sum(c.params["Wu"]*u[2,:].^2) + sum(c.params["Wy"]*(z[2,:] .- y_ref).^2)
        # + sum(c.params["Wx"]*(z[1,:]-x_ref).^2) + + sum(c.params["Wv"]*(z[4,:]-v_ref).^2)
        + sum(c.params["WΔu"]*(u[2,1]-c.u[2]).^2) + sum(c.params["WΔu"]*(u[2,2:end]-u[2,1:end-1]).^2) 
    
    @objective(model, Min, J)

    t_comp = @elapsed JuMP.optimize!(model)
    sol = value.(u)
    set_nominal(c, sol)
    return sol, t_comp
end

# function linearizedMPC(c::MPC, vd_H; solver = "Ipopt")
#     y_ref = c.y_ref; vd = c.v_ref; dt = c.T; ϵ = c.params["ϵ"]
#     c.solver = solver
#     if c.solver == "Ipopt"
#         model = JuMP.Model(Ipopt.Optimizer)
#     end
#     set_silent(model)
#     set_time_limit_sec(model, 0.1)
#     set_optimizer_attribute(model, "tol", 1e-3)
#     set_optimizer_attribute(model, "max_iter", 100)
#     # set_optimizer_attribute(model, "linear_solver", "ma27")

#     # Define decision variables
#     @variables model begin
#         z[1:4, 1:c.H+1]
#         u[1:2, 1:c.H]
#         zH[1:2, 1:c.H+1]
#         uH[1:c.H]
#         # s[1:2, 1:c.H+1]
#     end

#     # Set initial values for the optimization variables 
#     warm_u = hcat(c.u_nom[:,2:end], zeros(2,1))
#     set_nominal(c, warm_u)
#     set_start_value.(u, c.u_nom)
#     set_start_value.(z, c.st_nom)
    
#     # Add dynamics constraints
#     for t = 1:c.H
#         # for CAV
#         # β = @expression(model, u[2,t]*c.lr/(c.lf+c.lr))
#         β_nom = atan(tan(c.u_nom[2,t])*c.lr/(c.lf+c.lr))
#         θ_nom = c.st_nom[3,t]; v_nom = c.st_nom[4,t] 
#         r, A, B = construct_linear_matrix(c, v_nom, θ_nom, β_nom)
#         @constraints(model, begin z[:,t+1] .== z[:,t] + dt*(r+A*z[:,t] + B*u[:,t]) end)
#         # for HDV
#         @constraints(model, begin zH[:,t+1] .== zH[:,t] + dt*[zH[2,t], 0] + dt*uH[t]*[0.5*dt, 1] end)
#     end

#     # Initial condition
#     @constraint(model, z[:,1] .== c.st)
#     @constraint(model, zH[:,1] .== [c.st_H[1], c.st_H[4]])
    
#     # Bound constraints
#     @constraints(model, begin
#         c.a_min .<= u[1,:] .<= c.a_max
#         c.α_min .<= u[2,:] .<= c.α_max
#         c.v_min .<= z[4,:] .<= c.v_max
#         # 0.0 .<= s
#     end)

#     @constraints(model, begin
#         # c.params["y_min"] .<= z[2,:] .+ s[1,:] 
#         # -c.params["y_max"] .<= -z[2,:] .+ s[2,:] 
#         c.params["y_min"] .<= z[2,:] .<= c.params["y_max"]
#     end)

#     # Ramp constraints
#     @constraints(model, begin
#         c.params["Δθ_min"]*dt <= z[3,1] - c.st[3] <= c.params["Δθ_max"]*dt
#         c.params["Δθ_min"]*dt .<= z[3,2:end] - z[3,1:end-1] .<= c.params["Δθ_max"]*dt
#     end)

#     # Objective function
#     J = sum(c.params["Wu"].*u.^2) + sum(c.params["Wv"]*(z[4,:] .- vd).^2) + sum(c.params["Wy"]*(z[2,:] .- y_ref).^2)
#                 + sum(c.params["WHu"].*uH.^2) + sum(c.params["Wv"]*(zH[2,:] .- vd_H).^2)
#                 # + c.params["λ"]*sum(s)
#     for k in 1:c.H
#         xi = z[1,k+1]; yi = z[2,k+1]
#         xj = zH[1,k+1]
#         J = @NLexpression(model, J - c.params["Wd"]*log((xi-xj)^2 + yi^2 + ϵ))
#         # @constraint(model, 5.0^2 <= (xi-xj)^2 + yi^2)
#     end
#     @NLobjective(model, Min, J)
#     # Safety constraints

#     t_comp = @elapsed JuMP.optimize!(model)
#     sol = value.(u)
#     set_nominal(c, sol)
#     return sol, t_comp
# end

# function construct_linear_matrix(c::MPC, v_nom::Float64, θ_nom::Float64, β_nom::Float64)
#     r = [v_nom*cos(θ_nom+β_nom)+v_nom*sin(θ_nom+β_nom)*(θ_nom+β_nom)-cos(θ_nom+β_nom)*v_nom,
#     v_nom*sin(θ_nom+β_nom)-v_nom*cos(θ_nom+β_nom)*(θ_nom+β_nom)-sin(θ_nom+β_nom)*v_nom,
#     (v_nom*sin(β_nom)-β_nom*v_nom*cos(β_nom)-v_nom*sin(β_nom))/c.lr,
#     0]
#     A = zeros(4,4)
#     A[1,3] = -v_nom*sin(θ_nom+β_nom)
#     A[1,4] = cos(θ_nom+β_nom)
#     A[2,3] = v_nom*cos(θ_nom+β_nom)
#     A[2,4] = sin(θ_nom+β_nom)
#     A[3,4] = sin(β_nom)/c.lr
#     B = zeros(4,2)
#     B[1,2] = -v_nom*sin(θ_nom+β_nom)*c.lr/(c.lf+c.lr)
#     B[2,2] = v_nom*cos(θ_nom+β_nom)*c.lr/(c.lf+c.lr)
#     B[3,2] = v_nom*cos(β_nom)*c.lr/(c.lf+c.lr)
#     B[4,1] = 1 

#     return r, A, B
# end