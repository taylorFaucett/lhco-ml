import uproot
import pathlib

path = pathlib.Path.cwd()


def read_data(filename):
    # Read data files
    file = uproot.open(filename)
    events = file["Delphes"]
    array = events["FatJet"].array()
    print(array.show())
    #x = file["Delphes"]["FatJet"].arrays("FatJet/FatJet.PT")
    


if __name__ == "__main__":
    data_path = path / "data"
    for file in data_path.glob("*"):
        read_data(file)
        break