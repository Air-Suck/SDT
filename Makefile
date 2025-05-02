clean:
	rm -rf output/*
	rm -rf output.*
tobin:clean
	python my_utils/tobin.py
totxt:
	python 	my_utils/totxt.py
test:clean
	python test.py --pretrained_model model_zoo/checkpoint-iter199999.pth --store_type online --sample_size 500 --dir temp
.PHONY:clean tobin totxt
