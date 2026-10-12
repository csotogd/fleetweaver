"""Classical baselines for the SAME min-max, heterogeneous, multi-trip problem."""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def greedy(inst, balanced=True):
    xy=np.vstack([inst["depot"],inst["locs"]]).astype(np.float64); d=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    m=len(inst["speed"]); paths=[[0] for _ in range(m)]
    elapsed=np.zeros(m); load=np.zeros(m); remaining=set(range(1,len(xy)))
    while remaining:
        options=[]
        for v in range(m):
            for j in remaining:
                refill=load[v]+inst["demand"][j-1]>inst["capacity"][v]
                travel=(d[paths[v][-1],0]+d[0,j]) if refill else d[paths[v][-1],j]
                time=elapsed[v]+(travel+d[j,0])/inst["speed"][v]
                score=time if balanced else travel/inst["speed"][v]
                options.append((score,travel,v,j,refill))
        _,travel,v,j,refill=min(options)
        if refill:
            paths[v].append(0); load[v]=0
        paths[v].append(j); load[v]+=inst["demand"][j-1]
        elapsed[v]+=travel/inst["speed"][v]; remaining.remove(j)
    for path in paths:
        if path[-1]!=0:path.append(0)
    return paths

def two_opt(inst, paths):
    """Deterministic intra-trip 2-opt; capacity and vehicle assignments stay fixed."""
    xy=np.vstack([inst["depot"],inst["locs"]]).astype(np.float64); d=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    result=[]
    for path in paths:
        route=list(path)
        depots=[i for i,j in enumerate(route) if j==0]
        for lo,hi in zip(depots,depots[1:]):
            changed=True
            while changed:
                changed=False
                for a in range(lo+1,hi-1):
                    for b in range(a+1,hi):
                        delta=d[route[a-1],route[b]]+d[route[a],route[b+1]]-d[route[a-1],route[a]]-d[route[b],route[b+1]]
                        if delta < -1e-8:
                            route[a:b+1]=route[a:b+1][::-1]; changed=True
        result.append(route)
    return result

def ortools(inst, seconds=1.0, warm_start=True):
    """Routing solver with explicit depot reload copies and a pure min-max objective.

    N-1 reload copies per vehicle are sufficient for any useful solution (at most N
    non-empty trips). No restriction to a single trip or an aggregate-distance
    surrogate. Integer travel times are rounded at 1e5 precision; independently
    recompute the reported objective in float coordinates.
    """
    n=len(inst["locs"]); m=len(inst["speed"]); scale=100000
    xy=np.vstack([inst["depot"],inst["locs"]]).astype(np.float64)
    base=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    reloads=max(0,n-1)
    nodes=1+n+m*reloads
    manager=pywrapcp.RoutingIndexManager(nodes,m,0)
    routing=pywrapcp.RoutingModel(manager)
    def original(node): return node if node<=n else 0
    time_callbacks=[]; demand_callbacks=[]
    # Keep Python callbacks alive until the solver returns.
    callbacks=[]
    for v in range(m):
        speed=float(inst["speed"][v]); capacity=int(inst["capacity"][v])
        def transit(a,b,speed=speed):
            return int(round(base[original(manager.IndexToNode(a)),original(manager.IndexToNode(b))]/speed*scale))
        def demand(a,capacity=capacity):
            j=manager.IndexToNode(a)
            return int(inst["demand"][j-1]) if 1<=j<=n else (-capacity if j>n else 0)
        callbacks.extend([transit,demand])
        time_callbacks.append(routing.RegisterTransitCallback(transit))
        demand_callbacks.append(routing.RegisterUnaryTransitCallback(demand))
    routing.AddDimensionWithVehicleTransits(time_callbacks,0,10**10,True,"Time")
    routing.GetDimensionOrDie("Time").SetGlobalSpanCostCoefficient(1)
    capacities=[int(x) for x in inst["capacity"]]
    routing.AddDimensionWithVehicleTransitAndCapacity(demand_callbacks,max(capacities),capacities,True,"Load")
    load=routing.GetDimensionOrDie("Load")
    for node in range(1,n+1):load.SlackVar(manager.NodeToIndex(node)).SetValue(0)
    for v in range(m):
        load.SlackVar(routing.Start(v)).SetValue(0)
        for k in range(reloads):
            node=n+1+v*reloads+k; idx=manager.NodeToIndex(node)
            routing.VehicleVar(idx).SetValues([-1,v])
            routing.AddDisjunction([idx],0)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(max(1,int(seconds*1000)))
    assignment=None
    if warm_start:
        warm=two_opt(inst,greedy(inst))
        encoded=[]
        for v,path in enumerate(warm):
            k=0; route=[]
            for j in path[1:-1]:
                if j==0: route.append(n+1+v*reloads+k); k+=1
                else: route.append(j)
            encoded.append(route)
        assignment=routing.ReadAssignmentFromRoutes(encoded,True)
    solution=(routing.SolveFromAssignmentWithParameters(assignment,params) if assignment
              else routing.SolveWithParameters(params))
    if solution is None:return None
    paths=[]
    for v in range(m):
        idx=routing.Start(v); path=[0]
        while not routing.IsEnd(idx):
            idx=solution.Value(routing.NextVar(idx)); path.append(original(manager.IndexToNode(idx)))
        paths.append(path)
    return paths
