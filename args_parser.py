import argparse
import os
import sys

# Function to parse cli arguments, based on the caller script
def parser():

    # Read which script called the parser
    script = os.path.basename(sys.argv[0]).lower()

    parser = argparse.ArgumentParser()

    if 'nlsh_build' in script:
        # Required arguments (common in build and search)
        parser.add_argument('-d', required=True)
        parser.add_argument('-i', required=True)
        parser.add_argument('-type', required=True, choices=['sift', 'mnist', 'protein'])
        # Optional arguments with default values
        parser.add_argument('--knn', type=int, default=10)
        parser.add_argument('-m', type=int, default=100)
        parser.add_argument('--imbalance', type=float, default=0.03)
        parser.add_argument('--kahip_mode', type=int, choices=[0, 1, 2], default=2)
        parser.add_argument('--layers', type=int, default=3)
        parser.add_argument('--nodes', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--seed', type=int, default=1)
    elif 'nlsh_search' in script:
        # Required arguments (common in build and search)
        parser.add_argument('-d', required=True)
        parser.add_argument('-i', required=True)
        parser.add_argument('-type', required=True, choices=['sift', 'mnist', 'protein'])
        # Optional arguments with default values
        parser.add_argument('-q', required=True)
        parser.add_argument('-o', required=True)
        parser.add_argument('-N', type=int, default=1)
        parser.add_argument('-R', type=float)
        parser.add_argument('-T', type=int, default=5)
        parser.add_argument('-range', default='true', choices=['true', 'false'])
    elif 'protein_embed' in script:
        parser.add_argument('-i', required=True)
        parser.add_argument('-o', required=True)
    elif 'protein_search' in script:
        parser.add_argument('-d', required=True)
        parser.add_argument('-q', required=True)
        parser.add_argument('-o', required=True)
        parser.add_argument('-method', required=True, choices=['all', 'lsh', 'hypercube', 'neural', 'ivfflat', 'ivfpq'])

    args = parser.parse_args()
    validate_args(args, parser, script)
    return args


# Funciton to validate given cli arguments
def validate_args(args, parser, script):
    
    if 'nlsh_build' in script:
        # Required file validations
        if not os.path.isfile(args.d):
            parser.error(f"Input file not found: {args.d}")
        
        # Create a directory from user input
        if not os.path.isdir(args.i) and '.' not in os.path.basename(args.i):
            index_dir = args.i
        else:
            index_dir = os.path.dirname(os.path.abspath(args.i))
        if index_dir and not os.path.isdir(index_dir):
            try:
                os.makedirs(index_dir, exist_ok=True)
            except Exception as e:
                parser.error(f"Cannot create directory for index path: {index_dir} ({e})")
        # Numeric validations
        if args.knn <= 0:
            parser.error("-knn must be positive")
        if args.m <= 0:
            parser.error("-m must be positive")
        if not (0 <= args.imbalance <= 1):
            parser.error("--imbalance must be between 0 and 1")
        if args.layers <= 0:
            parser.error("--layers must be positive")
        if args.nodes <= 0:
            parser.error("--nodes must be positive")
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        if args.batch_size <= 0:
            parser.error("--batch_size must be positive")
        if args.lr <= 0:
            parser.error("--lr must be positive")
        if args.seed < 0:
            parser.error("--seed must be non-negative")

    elif 'nlsh_search' in script:
        # Required file validations
        if not os.path.isfile(args.d):
            parser.error(f"Input file not found: {args.d}")
        
        # Create a directory from user input
        if not os.path.isdir(args.i) and '.' not in os.path.basename(args.i):
            index_dir = args.i
        else:
            index_dir = os.path.dirname(os.path.abspath(args.i))
        if index_dir and not os.path.isdir(index_dir):
            try:
                os.makedirs(index_dir, exist_ok=True)
            except Exception as e:
                parser.error(f"Cannot create directory for index path: {index_dir} ({e})")
        # Default R value
        if args.R is None:
            if args.type == 'mnist':
                args.R = 2000.0
            else:
                args.R = 2800.0
        
        args.range = args.range.lower() == 'true' # Make flag boolean

        output_dir = os.path.dirname(os.path.abspath(args.o))
        if output_dir and not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                parser.error(f"Cannot create directory for index path: {output_dir} ({e})")
        if args.N <= 0:
            parser.error("-N must be positive")
        if args.R <= 0:
            parser.error("-R must be positive")
        if args.T <= 0:
            parser.error("-T must be positive")

    elif 'protein_embed' in script:
        # Required file validation
        if not os.path.isfile(args.i):
            parser.error(f"Input file not found: {args.i}")
    
    elif 'protein_search' in script:
        # Required file validations
        if not os.path.isfile(args.d):
            parser.error(f"Input file not found: {args.d}")
        if not os.path.isfile(args.q):
            parser.error(f"Query file not found: {args.q}")