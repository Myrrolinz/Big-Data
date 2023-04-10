from config import *
from utils import compare_results, load_result

def get_baseline(length):
    return {k + 1 : 1/length for k in range(length)}

if __name__ == "__main__":
    r1 = load_result(BASIC_OUT)
    r2 = load_result(STRIPE_OUT)
    compare_results(r1, r2, NORM)