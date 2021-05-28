

data: data/20210312 data/20210312_test data/task3-raw-data

data/20210312:
	curl -L 'https://www.dropbox.com/sh/i44patmfz2lbqmk/AAD8ixhwCxVoujU6OBgBe1fia?dl=1' -o casedata.zip
	unzip casedata.zip -x / -d data
	rm -rf casedata.zip

data/20210312_test:
	curl -L 'https://www.dropbox.com/sh/6n6zuhmvzaw0cak/AADCL6ldXP0azeR5UoXPp1rga?dl=1' -o casedata_test.zip
	unzip casedata_test.zip -x / -d test_data
	mv test_data/20210312 data/20210312_test
	rm -rf casedata_test.zip test_data

subtask4: subtask4/data

subtask4/data:
	cd subtask4 && mkdir -p data
	cp data/20210312/subtask4-token/en-train.txt subtask4/data/en-orig.txt
	cp data/20210312/subtask4-token/es-train.txt subtask4/data/es-orig.txt
	cp data/20210312/subtask4-token/pr-train.txt subtask4/data/pt-orig.txt
	cp data/20210312_test/english/subtask4-Token/test.txt subtask4/data/en-test.txt
	cp data/20210312_test/spanish/subtask4-Token/test.txt subtask4/data/es-test.txt
	cp data/20210312_test/portuguese/subtask4-Token/test.txt subtask4/data/pt-test.txt

data/task3-raw-data:
	curl -L 'https://www.dropbox.com/sh/3t3a202foz78jep/AAC8nHrnHSegu3szMdL72sKDa?dl=1' -o task3-raw-data.zip
	unzip task3-raw-data.zip -x / -d data/task3-raw-data
	rm -rf task3-raw-data.zip

task3: task3/data

task3/data:
	unzip data/task3-raw-data/05-2020.zip -d task3/data
	unzip data/task3-raw-data/06-2020.zip -d task3/data
	gzip -c -d data/task3-raw-data/NYTimes-2020-5.gz > task3/data/NYTimes-2020-5
	gzip -c -d data/task3-raw-data/NYTimes-2020-6.gz > task3/data/NYTimes-2020-6
