import sys
import argparse

# Simulate command line args
sys.argv = ['benchmark.py', '--batch-sizes', '512', '--num-requests', '512']

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:8000')
parser.add_argument('--batch', type=int)
parser.add_argument('--batch-sizes', type=str)
parser.add_argument('--requests', type=int, default=100)
parser.add_argument('--num-requests', type=int)
parser.add_argument('--max-tokens', type=int, default=100)
parser.add_argument('--test-all-batches', action="store_true")
parser.add_argument('--save')
parser.add_argument('--output')

args = parser.parse_args()

num_requests = args.num_requests if args.num_requests else args.requests

if args.batch_sizes:
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
elif args.test_all_batches:
    batch_sizes = [1, 4, 16, 64, 256, 512]
elif args.batch:
    batch_sizes = [args.batch]
else:
    batch_sizes = [1, 64, 256]

print(f"Parsed batch_sizes: {batch_sizes}")
print(f"Parsed num_requests: {num_requests}")
