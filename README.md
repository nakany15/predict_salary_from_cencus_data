# Income prediction API
## Environment Set up
* Download and install conda if you don’t have it already.
    * conda create -n [envname] "python=3.8"
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.
* Install required libraries.
    * pip install -r requirements.txt

Initialize DVC.

```
dvc init
```

Download data from [UCI Machine Learning Repositry](https://archive.ics.uci.edu/ml/datasets/census+income) and place it in starter/data.<div>
Track the data using DVC.<div>
```
dvc add starter/data/census.csv
```

Set up DVC remote to store data in AWS S3 bucket.
```
dvc remote add -d storage s3://<name-of-s3-bucket>
```

Run dvc pipeline by `dvc repro` command.

## Deploy the project to heroku
Create a new heroku application.
```
heroku create <your-application-name> --buildpack heroku/python
```
set git remote heroku.
```
heroku git:remote --app <your-application-name>
```
Add extra buildpack.
```
heroku buildpacks:add --index 1 heroku-community/apt
```

Git push heroku main to deploy.
```
git push heroku main
```

Add AWS configuration keys
```
heroku config:set AWS_ACCESS_KEY_ID=<your-access-key-id> AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
```