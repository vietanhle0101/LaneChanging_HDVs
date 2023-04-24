using LinearAlgebra, Distributions, Polynomials
using Optim, Convex, JuMP, OSQP, Ipopt
# using Gurobi, KNITRO

# GRB_ENV = Gurobi.Env()
const cutoff = 1e-4 # small cutoff in constraints to avoid numerical issues with optimization solvers

"Function to get control input for the HDVs in the simulation"
function input_for_HDV(cars::Vector{Car}, my_car::Int64, your_car::Int64, vd::Float64, W; T = 0.2, ϵ = 1e-3, uncertain = false)
    my_x = cars[my_car].st[1]; my_y = cars[my_car].st[2]
    my_v = cars[my_car].st[4]
    your_x = cars[your_car].st[1]; your_y = cars[your_car].st[2]
    your_v = cars[your_car].st[4]

    u = IRL_CFM(my_x, my_y, my_v, your_x, your_y, your_v, vd, W, T, ϵ, uncertain)
    return u
end

"Solve optimization problem (IRL) to find control action for HDV"
function IRL_CFM(my_x, my_y, my_v, your_x, your_y, your_v, v_d, W, T, ϵ, uncertain)
    C = zeros(7)
    C[1] = W[1] + W[2]*T^2
    C[2] = 2W[2]*(my_v - v_d)*T
    C[3] = W[2]*(my_v - v_d)^2
    C[4] = W[3]
    C[5] = (0.5T^2)^2
    C[6] = 2*0.5T^2*(my_x + T*my_v - your_x - T*your_v)
    C[7] = (my_y - your_y)^2 + (my_x + T*my_v - your_x - T*your_v)^2 + ϵ

    # The solution of this problem can be found by finding root of a polynomial (derivative of objective function)
    root = roots(Polynomial([(C[2]C[7]-C[4]C[6]), (C[2]C[6]+2C[1]C[7]-2C[4]C[5]), (C[2]C[5]+2C[1]C[6]), 2C[1]C[5]]))
    sol = root[isreal.(root)]
    u = Real(sol[1])

    if uncertain
        return u + rand(Normal(0., 0.1))
    else
        return u
    end
end

"The class to implement the MPC"
mutable struct MPC
    T::Float64
    H::Int64
    params
    v_min; v_max; a_min; a_max; α_min; α_max
    y_ref::Float64; v_ref::Float64
    lf::Float64; lr::Float64
    st; u
    st_nom::Matrix{Float64}; u_nom::Matrix{Float64}
    solver

    function MPC(T::Float64, H::Int64)
        obj = new(T, H)
        return obj
    end
end

"Set the control parameters"
function set_params(c::MPC, control_params)
    c.params = control_params
end

"Set the physical limits of the traffic and vehciles"
function set_limit(c::MPC, bounds, parameters)
    c.v_min = bounds["v_min"]; c.v_max = bounds["v_max"]
    c.a_min = bounds["a_min"]; c.a_max = bounds["a_max"]
    c.α_min = bounds["α_min"]; c.α_max = bounds["α_max"]
    c.lf = parameters["lf"]; c.lr = parameters["lr"] 
end

"Set the references"
function set_ref(c::MPC, y_ref, v_ref)
    c.y_ref = y_ref
    c.v_ref = v_ref
end

"Set state from all agent states"
function set_state(c::MPC, car::Car)
    c.st = car.st
end

"Set input from all agent inputs"
function set_input(c::MPC, car::Car)
    c.u = car.u
end

"Predict nominal trajectory over control horizon given the nominal control inputs, for warm-starting"
function set_nominal(c::MPC, nom_inputs::AbstractMatrix)
    c.u_nom = deepcopy(nom_inputs)
    c.st_nom = zeros(4, c.H+1)
    c.st_nom[:,1] = c.st
    for k in 1:c.H
        c.st_nom[:,k+1] = predict_onestep(c, c.st_nom[:,k], c.u_nom[:,k])
    end
end

function predict_onestep(c::MPC, x::AbstractVector, u::AbstractVector)
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

"Function to solve MPC motion planning problem"
function formulateMPC(c::MPC; solver = "Ipopt")
    y_ref = c.y_ref; vd = c.v_ref; dt = c.T
    c.solver = solver
    if c.solver == "Ipopt"
        model = JuMP.Model(Ipopt.Optimizer)
    end
    set_silent(model)
    set_time_limit_sec(model, 0.2)

    # Define decision variables
    @variables model begin
        z[1:4, 1:c.H+1]
        u[1:2, 1:c.H]
        # s[1:2, 1:c.H+1]
    end

    warm_u = hcat(c.u_nom[:,2:end], zeros(2,1))
    set_nominal(c, warm_u)
    set_start_value.(u, c.u_nom)
    set_start_value.(z, c.st_nom)
    
    # Add dynamics constraints
    for t = 1:c.H
        β = @NLexpression(model, atan(tan(u[2,t])*c.lr/(c.lf+c.lr)))       
        @NLconstraint(model, z[1,t+1] == z[1,t] + dt*z[4,t]*cos(z[3,t]+β))
        @NLconstraint(model, z[2,t+1] == z[2,t] + dt*z[4,t]*sin(z[3,t]+β))
        @NLconstraint(model, z[3,t+1] == z[3,t] + dt*z[4,t]/c.lr*sin(β))
        @NLconstraint(model, z[4,t+1] == z[4,t] + dt*u[1,t])
    end

    # Initial condition
    @constraint(model, z[:,1] .== c.st)
    
    # Bound constraints
    @constraints(model, begin
        c.a_min .<= u[1,:] .<= c.a_max
        c.α_min .<= u[2,:] .<= c.α_max
        c.v_min .<= z[4,:] .<= c.v_max
        # 0.0 .<= s
    end)

    @constraints(model, begin
        # c.params["y_min"] .<= z[2,:] .+ s[1,:] 
        # -c.params["y_max"] .<= -z[2,:] .+ s[2,:] 
        c.params["y_min"] .<= z[2,:] .<= c.params["y_max"]
    end)

    # Ramp constraints
    @constraints(model, begin
        c.params["Δθ_min"]*dt <= z[3,1] - c.st[3] <= c.params["Δθ_max"]*dt
        c.params["Δθ_min"]*dt .<= z[3,2:end] - z[3,1:end-1] .<= c.params["Δθ_max"]*dt
    end)

    # Objective function
    J = sum(c.params["Wu"].*u.^2) + sum(c.params["Wv"]*(z[4,:] .- vd).^2) + sum(c.params["Wy"]*(z[2,:] .- y_ref).^2)
                # + c.params["λ"]*sum(s)
    @objective(model, Min, J)

    t_comp = @elapsed JuMP.optimize!(model)
    sol = value.(u)
    set_nominal(c, sol)
    return sol, t_comp
end
