from run_many_vs_many import run_many_vs_many   
from run_1_vs_many import run_1_vs_many


def main():
    print("RUNNING 1 EVADER VS MANY PURSUERS SIMULATIONS")
    print("==============================================")
    run_1_vs_many(True) #SET HUMAN RENDER MODE TO TRUE to VISUALIZE
    print("==============================================")
    print("DONE RUNNING 1 EVADER VS MANY PURSUERS SIMULATIONS")
    print("==============================================")
    print("RUNNING MANY PURSUERS VS MANY EVADERS SIMULATIONS")
    run_many_vs_many(False) #SET HUMAN RENDER MODE TO TRUE to VISUALIZE 
    print("==============================================")
    print("DONE RUNNING MANY PURSUERS VS MANY EVADERS SIMULATIONS")



# Run the main function to generate data
main()