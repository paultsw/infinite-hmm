all:
	cython ./ihmm/dp/dp.pyx
	python setup.py build_ext

clean:
	rm -rf ./ihmm/build ./ihmm/*.so
