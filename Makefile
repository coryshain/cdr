naacl19:
	python3 -m dtsr.bin.train experiments_naacl19/natstor_naacl19.ini -m DTSR.*
	python3 -m dtsr.bin.predict experiments_naacl19/natstor_naacl19.ini -p dev-test -m DTSR.*
	python3 -m dtsr.bin.pt experiments_naacl19/natstor_naacl19.ini -a -p dev-test -M loglik
	python3 -m dtsr.bin.train experiments_naacl19/dundee_naacl19.ini -m DTSR.*
	python3 -m dtsr.bin.predict experiments_naacl19/dundee_naacl19.ini -p dev-test -m DTSR.*
	python3 -m dtsr.bin.pt experiments_naacl19/dundee_naacl19.ini -a -p dev-test -M loglik
	python3 -m dtsr.bin.train experiments_naacl19/ucl_naacl19.ini -m DTSR.*
	python3 -m dtsr.bin.predict experiments_naacl19/ucl_naacl19.ini -p dev-test -m DTSR.*
	python3 -m dtsr.bin.pt experiments_naacl19/ucl_naacl19.ini -a -p dev-test -M loglik
	python3 -m dtsr.bin.pt experiments_naacl19/natstor_naacl19.ini experiments_naacl19/dundee_naacl19.ini experiments_naacl19/ucl_naacl19.ini -P -p dev-test -M loglik

