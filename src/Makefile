
all: test

test: test.cu
	nvcc test.cu -o test -std=c++20 -O3 -arch=sm_80 -I$(HOME)/pkg/fmt-2023-03-01/include -L$(HOME)/pkg/fmt-2023-03-01/lib -lfmt

clean:
	rm -fv test
