"Some neccesary functions"
function sigmoid(x)
    return 1/(1 + exp(-x))
end

function sigmoid_der(x)
    return sigmoid(x)*(1-sigmoid(x))
end

function distance(car_i::Car, car_j::Car)
    return sqrt((car_i.st[1]-car_j.st[1])^2 + (car_i.st[2]-car_j.st[2])^2)
end

# Function to generate random entering time
function randomTimeGen(number::Int64; Timestep = 0.1, mu = 5.0, sigma = 1.0, offset = 3.0)
    t_diff = rand(Normal(mu, sigma), number) 
    t_diff = max.(t_diff, offset)
    tt = cumsum(t_diff)
    return Timestep*ceil.(tt/Timestep)
end