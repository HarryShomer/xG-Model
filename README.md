# NHL Expected Goals Model

The code here is for creating a model that predicts the probability that an unblocked shot will be a goal in a National
Hockey League (NHL) game. The data used for the model spans from the 2007 season until the 2016 season for all regular
season and playoff games. All the data used in this project was scraped using my scraper that can be found
[here.](https://github.com/HarryShomer/Hockey-Scraper) (and is available on pip as "hockey_scraper").

## Model Features

The features used to built each model are the following:

1. Distance: Distance of shot from the net
2. Angle: Angle of shot
3. Shot Type: Slap Shot, Snap Shot, Wrist Shot, Deflected, Tip-In, Wrap-Around, Backhand
4. Off Wing: If the player took the shot from his off wing
5. Empty Net: If the net was empty
6. Strength: 5v5, 4x5, 5x5, 3x3...etc. for the shooting team
7. Score Category: Score differential for the shooting team. It spans from -3+ to 3+ (I just bin everything above 3 and below -3)
8. Is Forward: If the shooter is a forward
9. Is Home: If the shooter plays for the home team
10. Distance Change: Distance from previous event
11. Time Elapsed: The difference in time from the last event
12. Angle Change: The change in angle if it's a rebound shot (last event was an SOG  <= 2 seconds ago)
13. Previous Event & Team: Whether the previous event was a Fac, Sog, Block/Miss, or a Take/Hit (I changed gives to
takes for the other team) and for which team. This is represented by eight dummy variables (the four choices for both teams).

## Training and Testing

Three different classifiers were built using the same data and features: A Logistic regression, a Random Forest Classifer
and a Gradient Boosting Classifier. All three were built with the same exact training data, and tested on the same
testing data as well.

The data used here is the regular season and playoff data from the 2007-2016 seasons. I shuffled the data and used 80%
of the data for training the model (so the training and testing sets are both random subsets of the total dataset). Then,
for each model, I did 10 fold cross validation on the training set to tune the hyperparameters and create each model.

I then tested each model on the test set. Using each of my three new models, I calculated the probability of each shot
in the test set of being a goal. To evaluate these predictions, I calculated both the area underneath the ROC Curve (AUC)
and the Log Loss for each model's predictions on the test set. Both of these are shown below:

![](https://github.com/HarryShomer/xG-Model/blob/master/ROC_xG.png)

Log Loss:

* LR:   0.213
* RF:  0.209
* GBM:  0.207

The AUC score can take values from 0 to 1 with higher being better. For Log Loss they range from 0 to infinity with 0
being perfect. If we created a model that classified each shot randomly we would get an AUC score of .5 and a log loss
of .693. So that means each model is definitely better than random.

In terms of comparison, they are each very similar to each other with the one built using Gradient Boosting narrowly
leading the pack in both metrics. Due to the fact that the GBM best predicted out of sample data (as determined by both
measures) I chose it as my final model.

## Running it yourself
If you are looking to run this yourself you will need a few things. First, you'll need the proper dependencies. This
includes python 3 (I'm running 3.6), pandas, and scikit-learn. Second you need the data. You can use my scraper or any other way of getting it (getting it another way would obviously
entail making changes to the cleaning portions of the code). Third, the way I determine handedness and position among players
is by using my own database of players. You can either find a way to work around it or just get rid of those features
in the model.

Assuming both of those things are taken care of you can run the xg_model.py file which will run everything.
With that said, all together it could take a long time to run so maybe run it model by model (it'll still take some time
though).


