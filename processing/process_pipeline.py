from download_samples import download_samples
from generate_HL import generate_HL
from generate_EFPs import generate_EFPs
from generate_jet_images import generate_jet_images

def process_pipeline():
    # Download sample data 
    #print("RUNNING -> download_samples")
    #download_samples()

    # Generate HL Jet Substructure Observables from trimmed jets file
    print("RUNNING -> generate_extra_with_rotations")
    generate_extra_with_rotations()    
    
    # Generate HL Jet Substructure Observables from trimmed jets file
    print("RUNNING -> generate_HL")
    generate_HL()

    # Generate EFPs from trimmed jets
    print("RUNNING -> generate_EFPs")
    generate_EFPs()
    
    # Generate Jet Images from trimmed jets
    print("RUNNING -> generate_jet_images")
    generate_jet_images()

if __name__ == "__main__":
    process_pipeline()
