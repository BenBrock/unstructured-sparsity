
all: test

test: test.cu
	nvcc test.cu -o test -std=c++20 -O3 -arch=sm_80

clean:
	rm -fv test