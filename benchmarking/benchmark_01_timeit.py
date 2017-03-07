import pprint
import timeit

result = timeit.repeat('benchmark_running()', setup='from benchmark_01 import benchmark_running', number=1, repeat=1)
pprint.pprint(result)