# Expected Goals Model

The code here is for creating a model that predicts the probability of an unblocked shot being a goal in a National
Hockey League (NHL) game. The data used for the model spans from the 2007 season until the 2016 season for all regular
season and playoff games. All the data used in this project was scraped using my scraper that can be found
[here.](https://github.com/HarryShomer/Hockey-Scraper) (and is available as a python package as "hockey_scraper").


Three different classifiers were built using the same data and features: A logistic regression, a Random Forest Classifer
and a Gradient Boosting Classifier. The one that best predicted out of sample data (as determined by roc-auc and log loss)
and was chosen as my final model was the one built using a Gradient Boosting Classifier.
 

If you are looking to run this yourself you will need a few things (besides for the proper dependencies).
First you need the data. You can use my scraper or any other way of getting it (getting it another way would obviously
entail making changes to the cleaning portions of the code). Second, the way I determine handedness and position among players is by using my
is by my own database of players. You can either find a way to work around it or just get rid of those features in the model.
Assuming both of those things are taken care of you can run the xg_model.py file which will run everything. With that
said, all together it could take a long time to run so maybe run it model by model (it'll still take some time though).


