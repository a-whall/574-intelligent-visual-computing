import argparse
import numpy
from pathlib import Path



def find_input_csv_file_by_name(file):

    expected_file = f'**/{file}.csv'

    matching_file = list(Path('.').glob(expected_file))

    if len(matching_file) == 0:
        return None

    return matching_file[0]



def main(args):

    in_path = find_input_csv_file_by_name(args.file)

    if in_path is None:
        print('File not found.')
        exit()

    out_dir = Path(args.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f'{args.file}.pts'

    points = numpy.loadtxt(in_path, delimiter=',', skiprows=1)[:, 2:8]
    original_len = len(points)

    points = numpy.unique(points, axis=0) # Remove duplicate points
    unique_len = len(points)

    print(f'Removed {original_len-unique_len} duplicate points')

    numpy.savetxt(out_path, points, fmt='%.6f %.6f %.6f %.6f %.6f %.6f', delimiter=' ')
    print(f'Successfully saved {unique_len} points. See {out_path}.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTS-file-extractor")
    parser.add_argument(
        "--file",
        required=True,
        help="Must give the program the path to the mesh-file containing vertices and normals.\nNote that the vertex and normal data must be in csv format and exist in indices [2,6]."
    )
    parser.add_argument(
        "--output_directory",
        default="data",
        help="Name of directory to store the generated pts file."
    )
    main(parser.parse_args())