# ml-project-1-ml_ssg

Firstly, you need to make sure to insert the train.csv and test.csv file in the correct place. 
So go at the following two links and donwload the files:
- https://drive.google.com/file/d/1GlgHTsIrML1Mls04R8IQ8E-wWbz4YHF5/view?usp=sharing
- https://drive.google.com/file/d/1VrkRo8mkOWQokTq4rRU0vkp1atgsX-rj/view?usp=sharing

Place the test.csv and train.csv files into the /code folder.

To run the code, make sure numpy and matplotlib are installed.

From command line go to the /code folder and execute the run.py file. You will receive an output_logistic.csv file in the same directory of the run.py file, with all the prediction made using the most accurate model.

The implementations.py file contains all the respective machine learning models, with the addition of the polynomial expansion creation and the cross-validation.

The proj1_helpers.py contains helping function as to load the data, predict the output and do the submission, as well as preprocessing functions as standardization and outlier removal.
