import obspy
from obspy.io.seisan.core import _read_seisan
import sys
import argparse

# Function to read and print .hst file contents
def read_hst_file(file_path):
    try:
        # Validate file existence and accessibility
        import os
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            sys.exit(1)
        if not os.path.isfile(file_path) or not file_path.endswith('.hst'):
            print(f"Error: '{file_path}' is not a valid .hst file.")
            sys.exit(1)

        # Read the .hst file using ObsPy's SEISAN reader
        stream = _read_seisan(file_path)
        
        # Check if stream is None or empty
        if stream is None or len(stream) == 0:
            print(f"Error: No valid seismic data found in '{file_path}'.")
            sys.exit(1)
        
        # Print stream summary
        print(f"Stream contains {len(stream)} trace(s):")
        print(stream)
        
        # Iterate through each trace and print header details
        for trace in stream:
            print("\nTrace Header:")
            print(f"  Station: {trace.stats.station}")
            print(f"  Channel: {trace.stats.channel}")
            print(f"  Start time: {trace.stats.starttime}")
            print(f"  End time: {trace.stats.endtime}")
            print(f"  Sampling rate: {trace.stats.sampling_rate} Hz")
            print(f"  Number of samples: {trace.stats.npts}")
            print(f"  Data preview (first 5 samples): {trace.data[:5]}")
    
    except Exception as e:
        print(f"Error reading .hst file '{file_path}': {e}")
        sys.exit(1)

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read .hst file')
    parser.add_argument('--file', type=str, help='Path to the .hst file')
    args = parser.parse_args()
    
    # Call the function to read and print the file contents
    read_hst_file(args.file)
    