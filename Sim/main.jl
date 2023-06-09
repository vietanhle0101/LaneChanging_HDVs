push!(LOAD_PATH, ".")
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Plots, GaussianProcesses, Random, DelimitedFiles

include("fcns.jl")
include("path.jl")
include("car.jl")
include("controller.jl") 
# include("estimator.jl")
# include("util.jl")

Random.seed!(16);
T = 0.2; H = 20; G = 20
# velocity and input bounds
bounds = Dict("v_min" => 0.0, "v_max" => 30.0, "a_min" => -3.0, "a_max" => 2.0, "α_min" => -π/12, "α_max" => π/12)
system_params = Dict("lf" => 1.04, "lr" => 1.54)
CFM_params = Dict("τs" => 2.0, "ds" => 8.0)

## Initialize the car objects
yc_i = 6.0; yc_f = 0.0
vd = 28.0

CAV_1 = Car("CAV", 1,  T, [0.0, yc_i, 0.0, 26.6])
HDV_2 = Car("HDV", 2, T, [30.0, yc_f, 0.0, 28.0])
HDV_3 = Car("HDV", 3, T, [-30.0, yc_f, 0.0, 28.6])
Cars = [CAV_1, HDV_2, HDV_3]
for car in Cars
    set_limit(car, bounds, system_params)
end

Planner = MPC_Planner(T, H)
set_limit(Planner, bounds)
set_ref(Planner, vd)
set_state(Planner, CAV_1, HDV_2, HDV_3)
set_nominal(Planner, zeros(2,H))
W_AH = 1e3
W_H2 = 10.0 .^[0.0, 2.0]
W_H3 = 10.0 .^[-1.0, -0.5]

weights = Dict("W_A" => [1e-1, 1e1], "W_AH" => W_AH, "W_H" => W_H3, "ϵ" => 1e-6)
set_params(Planner, weights)

LL_params = Dict("Wu" => 1e0, "WΔu" => 1e0, "Wy" => 1e-2, "Wx" => 0e2, "Wv" => 0e2,
        "y_min" => yc_f - 3.0, "y_max" => yc_i + 3.0,
        "Δθ_min" => -π/18, "Δθ_max" => π/18, "Δα_min" => -π/18, "Δα_max" => π/18)

LLC = LL_MPC(T, H)
set_limit(LLC, bounds)
set_params(LLC, LL_params, system_params)
set_state(LLC, CAV_1)
set_nominal(LLC, zeros(2,H))

nonlinearMPC(Planner, vd)
lane = lane_selection(Planner, 1.0, 5.0)
set_ref(LLC, yc_f)
MIP_trackingMPC(LLC, Planner)
count_stop = 0

L = 200
t_comp = []
for t in 1:L
    println("Time step ", t)

    set_state(Planner, CAV_1, HDV_2, HDV_3)
    set_state(LLC, CAV_1)

    # Run HDV_2 using IRL-CFM model
    u_HDV_2 = input_for_HDV(Cars, 2, 1, vd, [W_H2; W_AH])
    run_car_following(HDV_2, u_HDV_2*T + HDV_2.st[4])

    # Run HDV_3 using IRL-CFM model
    if HDV_3.st[1] - CAV_1.st[1] > 0.0 hw = HDV_2.st[1] - HDV_3.st[1] else hw = CAV_1.st[1] - HDV_3.st[1] end
    v3_cfm = CTH(HDV_3, hw, CFM_params)
    u_HDV_3 = input_for_HDV(Cars, 3, 1, v3_cfm, [W_H3; W_AH])
    run_car_following(HDV_3, u_HDV_3*T + HDV_3.st[4])

    # Run CAV using MPC
    if HDV_3.st[1] - CAV_1.st[1] > 0.0 hw = HDV_3.st[1] - CAV_1.st[1] else hw = HDV_2.st[1] - CAV_1.st[1] end
    v_cfm = CTH(CAV_1, hw, CFM_params)
    if t > 5 && count_stop < 25
        set_ref(Planner, v_cfm)
        U_ref, solving_time_1 = nonlinearMPC(Planner, v3_cfm)
        lane = lane_selection(Planner, 1.0, 5.0)
        U, solving_time_2 = MIP_trackingMPC(LLC, Planner)
        run_lane_changing(CAV_1, U[:,1])
        append!(t_comp, solving_time_1 + solving_time_2)
    elseif t <= 5 && count_stop < 25
        run_car_following(CAV_1, v_cfm)
    else
        println("Changed lane")
        break
    end
    if abs(CAV_1.st[2] - yc_f) < 1e-2
        count_stop += 1
    end
end


# Plot the results
nn = size(CAV_1.X_hist)[2]
T_hist = [T*i for i in 1:nn] 

gr()
plot(CAV_1.X_hist[1,:], CAV_1.X_hist[2,:], ylims=(-5,10), color=:red)

plot()
for car in Cars
    car.Type == "CAV" ? c = :green : c = :red
    display(plot!(T_hist, car.X_hist[1,1:nn], color=c))
end

plot()
for car in Cars
    car.Type == "CAV" ? c = :green : c = :red
    display(plot!(T_hist, car.X_hist[4,1:nn], color=c))
end

plot(T_hist, CAV_1.X_hist[3,:])
dist = sqrt.((HDV_3.X_hist[1,1:nn]-CAV_1.X_hist[1,1:nn]).^2 + (HDV_3.X_hist[2,1:nn]-CAV_1.X_hist[2,1:nn]).^2)
plot(T_hist, dist)
minimum(dist)

dist = sqrt.((HDV_2.X_hist[1,1:nn]-CAV_1.X_hist[1,1:nn]).^2 + (HDV_2.X_hist[2,1:nn]-CAV_1.X_hist[2,1:nn]).^2)
plot(T_hist, dist)
minimum(dist)
plot(T_hist, CAV_1.U_hist[2,:])
plot(t_comp)






## Look-up table to find the optimal control weights
grid_size = (9,9)
nodes = (Vector(LinRange(-2.0, 2.0, grid_size[1])), Vector(LinRange(-2.0, 2.0, grid_size[2])))
TABLE = readdlm("./input/Time-Energy/Dec24_weight_table_81_TE.csv", ',')

data = TABLE'
x = data[3:4,:];              #predictors
y1 = data[1,:];   #regressors
#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
logObsNoise = -5.0                        # log standard deviation of observation noise (this is optional)
gp1 = GP(x, y1, mZero, kern, logObsNoise)       #Fit the GP
GaussianProcesses.optimize!(gp1)                         # Optimize the hyperparameters
μ1, σ² = predict_y(gp1, x);
print(maximum(y1 - μ1))
# plot(heatmap(gp1); fmt=:png)

y2 = data[2,:];   #regressors
#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
logObsNoise = -5.0                        # log standard deviation of observation noise (this is optional)
gp2 = GP(x, y2, mZero, kern, logObsNoise)       #Fit the GP
GaussianProcesses.optimize!(gp2)                         # Optimize the hyperparameters
μ2, σ² = predict_y(gp2, x);
print(maximum(y2 - μ2))
# plot(heatmap(gp2); fmt=:png)

gp = (gp1, gp2)

## Load map and initialize a list of path objects
segment_dict, path_dict, node_dict = load_data()

path1 = Path(1, 100, 80)
path2 = Path(2, 100, 80)
paths = [path1, path2]


W_H = 10 .^[-1.0, -0.5]


## Run single simulation to get the trajectory
P_hist, V_hist, U_hist, comp_time, cost, safe = main_BO_GP(gp, paths, 10 .^[-1.0, -0.5];
                                                    p0 = 5.0, v0 = 9.0, solver = "Ipopt", H = 20)

# Plot the results
using Plots
plot(P_hist')
plot(V_hist')
plot(U_hist[1,:])
plot(comp_time)

# Save data for later animation in python
# prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
# file_name = string("./results/", prefix, "_trajectory.csv")
# U_hist = hcat(U_hist, [NaN, NaN])
# DATA = vcat(P_hist, V_hist, U_hist)'
# writedlm(file_name, DATA)

## Comparison between BayOpt and SVO
BO_stats = Dict("SAFE" => [], "COST" => [])
SVO_stats = Dict("SAFE" => [], "COST" => [])
seed = 111; Random.seed!(seed)
n_simulation = 200

for i in 1:n_simulation
    println("Simulation ", i)
    p0 = rand(Uniform(-10.0, 10.0))
    v0 = rand(Uniform(5.0, 12.0))
    W_A = 10 .^ [rand(Uniform(-2.0, 2.0)), rand(Uniform(-2.0, 2.0))]
    _, _, _, _, cost1, _ = main_BO_GP(gp, paths, W_A; p0 = p0, v0 = v0, solver = "Ipopt", H = 10)
    _, _, _, _, cost2, _ = main_SVO(paths, W_A; p0 = p0, v0 = v0, solver = "Ipopt", H = 10)

    push!(BO_stats["COST"], cost1)
    push!(SVO_stats["COST"], cost2)

    if i%10 == 0
        # Number of simulations with safe guarantee
        println("Number of safe simumations: ", count(BO_stats["COST"] .< 1e3), ", ", count(SVO_stats["COST"] .< 1e3))

        idx = [i for i in 1:length(BO_stats["COST"]) if BO_stats["COST"][i] < 1e3 && SVO_stats["COST"][i] < 1e3]

        # Number of simulations with improvements
        n_improv = count(SVO_stats["COST"][idx] .> BO_stats["COST"][idx])/length(idx)
        println("Percentages of improvements: ", n_improv)

        # Average percentage of improvements
        aver_improv = mean((SVO_stats["COST"][idx] - BO_stats["COST"][idx])./SVO_stats["COST"][idx])
        println("Average improvements: ", aver_improv)
    end
end

# Number of simulations with safe guarantee
println("Number of safe simumations: ", count(BO_stats["COST"] .< 1e3), ", ", count(SVO_stats["COST"] .< 1e3))

idx = [i for i in 1:length(BO_stats["COST"]) if BO_stats["COST"][i] < 1e3 && SVO_stats["COST"][i] < 1e3]

# Number of simulations with improvements
n_improv = count(SVO_stats["COST"][idx] .> BO_stats["COST"][idx])/length(idx)
println("Percentages of improvements: ", n_improv)

# Average percentage of improvements
aver_improv = mean((SVO_stats["COST"][idx] - BO_stats["COST"][idx])./SVO_stats["COST"][idx])
println("Average improvements: ", aver_improv)

println("THAT IS THE END !!!")


## Save data for later animation in python
# using Dates
# prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
# file_name = string("./results/", prefix, "_statistics_", seed, ".csv")
# STATS = hcat(BO_stats["COST"], SVO_stats["COST"])
# writedlm(file_name, STATS)



coeffs = [1, -2, -2, 1]
@elapsed roots(Polynomial(coeffs))
@elapsed find_zero(Polynomial(coeffs), (-1e3, 1e3), Bisection())[end]


# To build GUROBI
ENV["GUROBI_HOME"] = "/Library/gurobi950/mac64"
import Pkg
Pkg.add("Gurobi")
Pkg.build("Gurobi")