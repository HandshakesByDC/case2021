

data: data/20210312 data/20210312_test

data/20210312:
	curl -L 'https://www.dropbox.com/sh/i44patmfz2lbqmk/AAD8ixhwCxVoujU6OBgBe1fia?dl=1' -o casedata.zip
	unzip casedata.zip -x / -d data
	rm -rf casedata.zip

data/20210312_test:
	curl -L 'https://www.dropbox.com/sh/6n6zuhmvzaw0cak/AADCL6ldXP0azeR5UoXPp1rga?dl=1' -o casedata_test.zip
	unzip casedata_test.zip -x / -d test_data
	mv test_data/20210312 data/20210312_test
	rm -rf casedata_test.zip test_data

task4: subtask4/data

subtask4/data:
	cd subtask4 && mkdir -p data
	cp data/20210312/subtask4-token/en-train.txt subtask4/data/en-orig.txt
	cp data/20210312/subtask4-token/es-train.txt subtask4/data/es-orig.txt
	cp data/20210312/subtask4-token/pr-train.txt subtask4/data/pt-orig.txt
	cp data/20210312_test/english/subtask4-Token/test.txt subtask4/data/en-test.txt
	cp data/20210312_test/spanish/subtask4-Token/test.txt subtask4/data/es-test.txt
	cp data/20210312_test/portuguese/subtask4-Token/test.txt subtask4/data/pt-test.txt
