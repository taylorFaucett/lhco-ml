import os
import sys

def generate_event(bbx):
    DelphesPythia8 = "/pub/tfaucett/semi-visible-jets-ml/generation/gen/HEPTools/MG5/Delphes/DelphesPythia8"
    if "BlackBox1" in bbx:
        DelphesCard = "process_files/delphes/delphes_card_BlackBox1.dat"
    elif "BlackBox2" in bbx:
        DelphesCard = "process_files/delphes/delphes_card_BlackBox2.dat"
    elif "BlackBox3" in bbx:
        DelphesCard = "process_files/delphes/delphes_card_BlackBox3.dat"
    else:
        print("No Delphes Card selected. Ending early")
        sys.exit()
    PythiaCard = "process_files/pythia/pythia_%s.cmnd" %(bbx)
    RootFile = "../data/%s.root" %(bbx)
    LogFile = "logs/%s.log" %(bbx)
    run_command = "nohup %s %s %s %s > %s 2>&1 &" %(DelphesPythia8, DelphesCard, PythiaCard, RootFile, LogFile)
    print("Follow progress using the command: ")
    print("tail -f %s" %LogFile)
    if not os.path.exists(RootFile):
        os.system(run_command)
    else:
        print("Root file already exists")
        print(RootFile)
        sys.exit()

if __name__ == "__main__":
    bbxs = {
        1: "BlackBox1_Z_XY_qq",
        2: "BlackBox1_qcd",
        3: "BlackBox3_KKg2qq",
        4: "BlackBox3_KKg2gr",
        5: "BlackBox3_qcd"
    }
    print("Select a dataset from the options below")
    print(bbxs)
    generate_event(bbxs[input("Selection: ")])