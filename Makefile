

data: data/20210312

data/20210312:
	curl -L 'https://www.dropbox.com/sh/i44patmfz2lbqmk/AAD8ixhwCxVoujU6OBgBe1fia?dl=1' -o casedata.zip
	unzip casedata.zip -x / -d data
	rm -rf casedata.zip
