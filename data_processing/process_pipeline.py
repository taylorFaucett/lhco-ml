import sys
from trim_towers import trim_towers
from generate_extra_from_rotations import generate_extra_from_rotations
from generate_HL import generate_HL
from generate_EFPs import generate_EFPs
from generate_jet_images import generate_jet_images

def process_pipeline(bbx):
    # Download sample data 
    print("RUNNING -> trim_towers")
    trim_towers(bbx)
    
    # Generate extra events to balance dataset by random rotations of signal
    print("RUNNING -> generate_extra_from_rotations")
    generate_extra_from_rotations(bbx)

    # Generate HL Jet Substructure Observables from trimmed jets file
    print("RUNNING -> generate_HL")
    generate_HL(bbx)

    # Generate Jet Images from trimmed jets
    print("RUNNING -> generate_jet_images")
    generate_jet_images(bbx)    

    # Generate EFPs from trimmed jets
    print("RUNNING -> generate_EFPs")
    generate_EFPs(bbx)

if __name__ == "__main__":
    bbx = str(sys.argv[1])
    process_pipeline(bbx)
