clean:
	rm -rf output/*
	rm -rf output.*
tobin:clean
	python my_utils/tobin.py
.PHONY:clean tobin
