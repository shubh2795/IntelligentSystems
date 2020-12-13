from cs5600_6600_f20_hw03 import *
ensembledTuple=load_nets(r"A:\USU\Assignments\IntelligentSystems\hw03\pck_nets")
net_ensemble=[]
print("accuracy of individual networks is below")
for individualNetork in ensembledTuple:
    net_ensemble.append(individualNetork[1])
    print(individualNetork[0])
    print(individualNetork[1].evaluate(valid_d))

print("accuracy of ensmbled network is below")
print(evaluate_net_ensemble(net_ensemble,valid_d))
