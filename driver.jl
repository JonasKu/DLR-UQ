include("settings.jl")
include("SGSolver.jl")
include("DLRSolver.jl")
include("plotting.jl")

using NPZ;

close("all");

s = Settings();
N = s.N;
s.NCons = 0;
s.iCons = 0;

# trun on optimized used of stabilizing terms
optimized = false;
#s.N = s.r;

#r = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]
r = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16]
errorExpBack = zeros(length(r));
errorVarBack = zeros(length(r));
errorExpFor = zeros(length(r));
errorVarFor = zeros(length(r));
errorExpSG = zeros(length(r));
errorVarSG = zeros(length(r));

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = false;
        s.useStabilizingTermsL = true;
    end
    solver = DLRSolver(s);
    @time tEnd, X, S, W = SolveBackward(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpBack[k], errorVarBack[k] = L2ErrorExpVar(plotSolution,Array((X*S*W')'))
    println("rank ",r[k]," : ",errorExpBack[k]," ",errorVarBack[k])
end
println("-> DLR projector-splitting integrator DONE.")

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = true;
        s.useStabilizingTermsL = false;
    end
    solver = DLRSolver(s);
    @time tEnd, X, S, W = SolveForward(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpFor[k], errorVarFor[k] = L2ErrorExpVar(plotSolution,Array((X*S*W')'))
    println("rank ",r[k]," : ",errorExpFor[k]," ",errorVarFor[k])
end
println("-> DLR unconventional integrator DONE.")

for k = 1:length(r)
    s.N = r[k];
    solver = Solver(s);
    @time tEnd, u = Solve(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpSG[k], errorVarSG[k] = L2ErrorExpVar(plotSolution,u)
    println("rank ",r[k]," : ",errorExpSG[k]," ",errorVarSG[k])
end
println("-> SG DONE.")

###### plot results ######

fig = figure("Figure6a",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorExpSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorExpBack, "g-.<", linewidth=2, label="DLRA", alpha=1.0)
ax.plot(r,errorExpFor, "m:>", linewidth=2, label="unconventional DLRA", alpha=1.0)
ylabel("Error Expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

fig = figure("Figure6b",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorVarBack, "g-.<", linewidth=2, label="DLRA", alpha=1.0)
ax.plot(r,errorVarFor, "m:>", linewidth=2, label="unconventional DLRA", alpha=1.0)
ylabel("Error Variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

##### driver filter ######
s.lambda = 0.00001;
s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = false;
        s.useStabilizingTermsL = true;
    end
    solver = DLRSolver(s);
    @time tEnd, X, S, W = SolveBackward(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpBack[k], errorVarBack[k] = L2ErrorExpVar(plotSolution,Array((X*S*W')'))
    println("rank ",r[k]," : ",errorExpBack[k]," ",errorVarBack[k])
end
println("-> fDLR projector-splitting integrator DONE.")

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = true;
        s.useStabilizingTermsL = false;
    end
    solver = DLRSolver(s);
    @time tEnd, X, S, W = SolveForward(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpFor[k], errorVarFor[k] = L2ErrorExpVar(plotSolution,Array((X*S*W')'))
    println("rank ",r[k]," : ",errorExpFor[k]," ",errorVarFor[k])
end
println("-> fDLR unconventional integrator DONE.")

for k = 1:length(r)
    s.N = r[k];
    solver = Solver(s);
    @time tEnd, u = Solve(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpSG[k], errorVarSG[k] = L2ErrorExpVar(plotSolution,u)
    println("rank ",r[k]," : ",errorExpSG[k]," ",errorVarSG[k])
end
println("-> fSG DONE.")

fig = figure("Figure6c",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorFilterExpSG, "k--o", linewidth=2, label="fSG", alpha=1.0)
ax.plot(r,errorExpBack, "g-.<", linewidth=2, label="fDLR", alpha=1.0)
ax.plot(r,errorExpFor, "m:>", linewidth=2, label="unconventional fDLR", alpha=1.0)
ylabel("Error Expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

fig = figure("Figure6d",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="fSG", alpha=1.0)
ax.plot(r,errorVarBack, "g-.<", linewidth=2, label="fDLRA", alpha=1.0)
ax.plot(r,errorVarFor, "m:>", linewidth=2, label="unconventional fDLRA", alpha=1.0)
ylabel("Error Variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
